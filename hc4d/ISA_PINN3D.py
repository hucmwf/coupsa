from Data import Data
from ISA_PINN import *

class Data3D(Data):
    def __init__(self, filename):
        self.load(filename)

    def load(self, filename):
        data = scipy.io.loadmat(filename)
        t = data['t'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        y = data['y'].flatten()[:, None]
        z = data['z'].flatten()[:, None]
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.Exact_u=data["Exact"] .real
        self.u_star = self.Exact_u.flatten()[:, None]
        X, Y,Z, T = np.meshgrid(x, y, z, t)
        self.X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None],Z.flatten()[:, None], T.flatten()[:, None]))
        lb = np.array([x.min(0), y.min(0),z.min(0), t.min(0)]).T
        ub = np.array([x.max(0), y.max(0),z.max(0), t.max(0)]).T
        X_f = lb + ((ub - lb) * lhs(4, n_f))

        self.X_f = torch.tensor(X_f, dtype=torch.float32)
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32)
        self.y_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32)
        self.z_f = torch.tensor(X_f[:, 2:3], dtype=torch.float32)
        self.t_f = torch.tensor(X_f[:, 3:4], dtype=torch.float32)
        X_Star_index = (np.where((self.X_star[:, 3:4] >= 0) & (self.X_star[:, 3:4] <= 1))[0]).reshape(-1, 1)
        self.X_star=self.X_star[X_Star_index][:,0]
        self.u_star=self.u_star[X_Star_index].flatten()[:,None]

        t_lb_index=(np.where(self.X_star[:,3:4]==0))[0].reshape(-1, 1)
        X_0=self.X_star[t_lb_index][:,0]
        u0=self.u_star[t_lb_index].flatten()[:,None]
        selected_indices=np.random.choice(X_0.shape[0],N0,replace=False)
        u0=u0[selected_indices].flatten()[:,None]
        X_0=X_0[selected_indices]
        
        x_lb_index=(np.where(self.X_star[:,0:1]==X.min()))[0].reshape(-1, 1)
        x_ub_index=(np.where(self.X_star[:,0:1]==X.max()))[0].reshape(-1, 1)
        y_lb_index=(np.where(self.X_star[:,1:2]==Y.min()))[0].reshape(-1, 1)
        y_ub_index=(np.where(self.X_star[:,1:2]==Y.max()))[0].reshape(-1, 1)
        z_lb_index=(np.where(self.X_star[:,2:3]==Z.min()))[0].reshape(-1, 1)
        z_ub_index=(np.where(self.X_star[:,2:3]==Z.max()))[0].reshape(-1, 1)
        selected_indices_x_lb = np.random.choice(x_lb_index.shape[0], Nb, replace=False)
        selected_indices_x_ub = np.random.choice(x_ub_index.shape[0], Nb, replace=False)
        selected_indices_y_lb = np.random.choice(y_lb_index.shape[0], Nb, replace=False)
        selected_indices_y_ub = np.random.choice(y_ub_index.shape[0], Nb, replace=False)
        selected_indices_z_lb = np.random.choice(z_lb_index.shape[0], Nb, replace=False)
        selected_indices_z_ub = np.random.choice(z_ub_index.shape[0], Nb, replace=False)

        x_lb_index=x_lb_index[selected_indices_x_lb]
        x_ub_index=x_ub_index[selected_indices_x_ub]
        y_lb_index=y_lb_index[selected_indices_y_lb]
        y_ub_index=y_ub_index[selected_indices_y_ub]
        z_lb_index=z_lb_index[selected_indices_z_lb]
        z_ub_index=z_ub_index[selected_indices_z_ub]  

        self.X_0=torch.tensor(X_0, dtype=torch.float32)
        self.x0=self.X_0[:,0:1]
        self.x_lb=torch.tensor(self.X_star[x_lb_index][:,0], dtype=torch.float32)
        self.y_lb=torch.tensor(self.X_star[y_lb_index][:,0], dtype=torch.float32)
        self.z_lb=torch.tensor(self.X_star[z_lb_index][:,0], dtype=torch.float32)
        self.x_ub=torch.tensor(self.X_star[x_ub_index][:,0], dtype=torch.float32)
        self.y_ub=torch.tensor(self.X_star[y_ub_index][:,0], dtype=torch.float32)
        self.z_ub=torch.tensor(self.X_star[z_ub_index][:,0], dtype=torch.float32)

        u_x_lb=self.u_star[x_lb_index].flatten()[:, None]
        u_y_lb=self.u_star[y_lb_index].flatten()[:, None]
        u_z_lb=self.u_star[z_lb_index].flatten()[:, None]
        u_x_ub=self.u_star[x_ub_index].flatten()[:, None]
        u_y_ub=self.u_star[y_ub_index].flatten()[:, None]
        u_z_ub=self.u_star[z_ub_index].flatten()[:, None]
        
        self.u0=torch.tensor(u0, dtype=torch.float32)
        u_x_lb=torch.tensor(u_x_lb, dtype=torch.float32)
        u_y_lb=torch.tensor(u_y_lb, dtype=torch.float32)
        u_z_lb=torch.tensor(u_z_lb, dtype=torch.float32)
        u_x_ub=torch.tensor(u_x_ub, dtype=torch.float32)
        u_y_ub=torch.tensor(u_y_ub, dtype=torch.float32)
        u_z_ub=torch.tensor(u_z_ub, dtype=torch.float32)

        self.lables_0={
            'u':self.u0,
        }
        self.lables_b={
            'u':{'x_lb':u_x_lb,'x_ub':u_x_ub,'y_lb':u_y_lb,'y_ub':u_y_ub,'z_lb':u_z_lb,'z_ub':u_z_ub}
        }
        self.inputs_0=[self.X_0[:,0:1],self.X_0[:,1:2],self.X_0[:,2:3],self.X_0[:,3:4]]
        self.inputs_f=[self.x_f,self.y_f,self.z_f,self.t_f]
        self.inputs_b={
            'x_lb':[self.x_lb[:,0:1],self.x_lb[:,1:2],self.x_lb[:,2:3],self.x_lb[:,3:4]],
            'x_ub':[self.x_ub[:,0:1],self.x_ub[:,1:2],self.x_ub[:,2:3],self.x_ub[:,3:4]],
            'y_lb':[self.y_lb[:,0:1],self.y_lb[:,1:2],self.y_lb[:,2:3],self.y_lb[:,3:4]],
            'y_ub':[self.y_ub[:,0:1],self.y_ub[:,1:2],self.y_ub[:,2:3],self.y_ub[:,3:4]],
            'z_lb':[self.z_lb[:,0:1],self.z_lb[:,1:2],self.z_lb[:,2:3],self.z_lb[:,3:4]],
            'z_ub':[self.z_ub[:,0:1],self.z_ub[:,1:2],self.z_ub[:,2:3],self.z_ub[:,3:4]]
        }

class ISAPINN3D(ISA_PINN):
    def __init__(self, data: Data, layers: list, adam_iter: int, newton_iter: int, f_model, Loss=..., lbfgs_lr=0.8, N_f=10000, checkPointPath="./checkPoint"):
        super().__init__(data, layers, adam_iter, newton_iter, f_model, Loss, lbfgs_lr, N_f, checkPointPath)
        self.SA_weights['u_b_weights']=nn.Parameter(torch.full((self.data.x_lb.shape[0], 1), 100.0, device=device))

    def u_model(self, x,y,z, t):
        return self.model(torch.cat([x,y,z, t], dim=1))

    def error_u(self):
        u_pred,f_u_pred=self.predict()
        u_star=self.data.u_star
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        return error_u
    
    def predict(self):
        X_star = torch.tensor(self.data.X_star, dtype=torch.float32, device=device, requires_grad=True)
        u_star = self.model(X_star)
        X_star = X_star.clone().detach().requires_grad_(True)
        f_u_star = self.f_model(self, X_star[:, 0:1], X_star[:, 1:2],X_star[:, 2:3],X_star[:, 3:4])
        u_star = u_star.detach().cpu().numpy()
        f_u_star = f_u_star.detach().cpu().numpy()
        return u_star, f_u_star
