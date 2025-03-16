import scipy.io
import torch
from pyDOE import lhs
from globalf import *

class Data:
    def __init__(self,filename) -> None:
        self.lables_0={
            'u':0
            }
        self.lables_b={
            'u':0
            }
        self.inputs_0=[]
        self.inputs_f=[]
        self.inputs_b={
            'u':[]
        }
        self.load(filename)

    def load(self,filename):
        data = scipy.io.loadmat(filename)

        tt = data['t'].T
        t = tt.flatten()[:,None]

        x = data['x'].T.flatten()[:,None]
        self.Exact = data['Exact']#.T

        self.Exact_u = self.Exact.real.T
        X, T = np.meshgrid(x, t)
        self.x=x
        self.t=t
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        u_star_ori=self.Exact_u
        self.u_star=u_star_ori#.flatten()[:,None]

        # Domain bounds
        lb = self.X_star.min(0)
        ub = self.X_star.max(0)

        idx_x = np.random.choice(x.shape[0], N0, replace=False)
        self.idx_x=idx_x
        x0 = x[idx_x,:]

        self.u0 = torch.tensor(self.Exact_u[idx_x, 0:1], dtype=torch.float32).cuda()
 
        idx_t = np.random.choice(t.shape[0], Nb, replace=False)
        self.idx_t=idx_t
        tb = t[idx_t,:]
        
        X_f = lb + (ub-lb)*lhs(2, self.N_f)
        self.x_f = torch.tensor(X_f[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_f = torch.tensor(X_f[:, 1:2]).float().requires_grad_(True).cuda()
        
        X0 = np.concatenate((x0, 0*x0 + t[0]), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.x0 = torch.tensor(X0[:, 0:1]).float().requires_grad_(True).cuda()
        self.t0 = torch.tensor(X0[:, 1:2]).float().requires_grad_(True).cuda()
        
        self.x_lb = torch.tensor(X_lb[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_lb = torch.tensor(X_lb[:, 1:2]).float().requires_grad_(True).cuda()
        self.x_ub = torch.tensor(X_ub[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_ub = torch.tensor(X_ub[:, 1:2]).float().requires_grad_(True).cuda()

        u_lb_all = self.u_star[0,  :].flatten()[:,None]
        u_ub_all = self.u_star[-1, :].flatten()[:,None]
        u_lb= u_lb_all[idx_t]
        u_ub= u_ub_all[idx_t]
        self.u_lb = torch.tensor(u_lb).float().requires_grad_(True).cuda()
        self.u_ub = torch.tensor(u_ub).float().requires_grad_(True).cuda()
        self.lables_0={
            'u':u0
        }
        self.lables_b={
            'u':[self.u_lb,self.u_ub]
        }
        self.inputs_0=[self.x0,self.t0]
        self.inputs_f=[self.x_f,self.t_f]
        self.inputs_b={
            'ub':[self.x_ub,self.t_ub],
            'lb':[self.x_lb,self.t_lb]
        }
