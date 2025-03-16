import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import pickle
import os
import datetime
from pyDOE import lhs
from globalf import *

class ISA_PINN_3D:

    def __init__(self, mat_filename, layers, tf_iter, newton_iter, f_model, ux_model=None, Loss=None, lbfgs_lr=0.8, N_f=10000, checkPointPath="./checkPoint"):
        self.N_f = N_f
        self.__Loadmat(mat_filename)
        self.layers = layers
        self.sizes_w = []
        self.sizes_b = []
        self.lbfgs_lr = lbfgs_lr

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        self.col_weights = torch.full((N_f, 1), 100.0, device=device, requires_grad=True)
        self.col_v_weights = torch.full((N_f, 1), 100.0, device=device, requires_grad=True)
        self.u_weights = torch.full((self.u0.shape[0], 1), 100.0, device=device, requires_grad=True)
        self.ub_weights = torch.full((self.u_x_lb.shape[0], 1), 100.0, device=device, requires_grad=True)
        self.v_weights = torch.full((self.v0.shape[0], 1), 100.0, device=device, requires_grad=True)
        self.vb_weights = torch.full((self.v_x_lb.shape[0], 1), 100.0, device=device, requires_grad=True)

        class NeuralNet(nn.Module):
            def __init__(self, layer_sizes):
                super(NeuralNet, self).__init__()
                layers = []
                input_size = layer_sizes[0]
                for output_size in layer_sizes[1:-1]:
                    layers.append(nn.Linear(input_size, output_size))
                    layers.append(nn.Tanh())
                    input_size = output_size
                layers.append(nn.Linear(input_size, layer_sizes[-1]))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)

        self.u_model = NeuralNet(self.layers)
        print(self.u_model)

        self.u_model = self.u_model.cuda()

        self.Loss = Loss
        self.tf_iter = tf_iter
        self.newton_iter = newton_iter
        self.f_model = f_model

    def load(self, path):
        self.u_model.load_state_dict(torch.load(path))
        self.u_model.eval()

    def uv_model(self, x, y, t):
        x = x.requires_grad_(True).to(device).float()
        y = y.requires_grad_(True).to(device).float()
        t = t.requires_grad_(True).to(device).float()
    
        uv = self.u_model(torch.cat([x, y, t], dim=1))
        u = uv[:,0:1]
        v = uv[:,1:2]
        return u, v

    def g_grad(self, model, x_f, y_f, t_f, t0, u0, v0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, v_x_lb, v_x_ub, v_y_lb, v_y_ub, XY, XT, YT, col_weights, u_weights, ub_weights, col_v_weights, v_weights, vb_weights):
        self.u_model.train()
    
        x_f = x_f.requires_grad_(True).float()
        y_f = y_f.requires_grad_(True).float()
        t_f = t_f.requires_grad_(True).float()
        t0 = t0.requires_grad_(True).float()
        u0 = u0.requires_grad_(True).float()
        v0 = v0.requires_grad_(True).float()
        x_lb = x_lb.requires_grad_(True).float()
        x_ub = x_ub.requires_grad_(True).float()
        y_lb = y_lb.requires_grad_(True).float()
        y_ub = y_ub.requires_grad_(True).float()
        u_x_lb = u_x_lb.requires_grad_(True).float()
        u_x_ub = u_x_ub.requires_grad_(True).float()
        u_y_lb = u_y_lb.requires_grad_(True).float()
        u_y_ub = u_y_ub.requires_grad_(True).float()
        v_x_lb = v_x_lb.requires_grad_(True).float()
        v_x_ub = v_x_ub.requires_grad_(True).float()
        v_y_lb = v_y_lb.requires_grad_(True).float()
        v_y_ub = v_y_ub.requires_grad_(True).float()
        XY = XY.requires_grad_(True).float()
        XT = XT.requires_grad_(True).float()
        YT = YT.requires_grad_(True).float()
   
#        col_weights = col_weights.to(device).requires_grad_(True)
#        u_weights = u_weights.to(device).requires_grad_(True)

        model.zero_grad()

        loss_value, mse_0, mse_b, mse_f = self.Loss(model, x_f, y_f, t_f, t0, u0, v0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, v_x_lb, v_x_ub, v_y_lb, v_y_ub, XY, XT, YT, col_weights, u_weights, ub_weights, col_v_weights, v_weights, vb_weights)

        loss_value.backward(retain_graph=True)
        grads = [param.grad.clone() for param in model.parameters()]
        model.zero_grad()
        loss_value.backward(retain_graph=True)
        grads_col = self.col_weights.grad.clone()
        grads_u = self.u_weights.grad.clone()
        grads_ub = self.ub_weights.grad.clone()
        grads_col_v = self.col_v_weights.grad.clone()
        grads_v = self.v_weights.grad.clone()
        grads_vb = self.vb_weights.grad.clone()

        return loss_value.item(), mse_0.item(), mse_b.item(), mse_f.item(), grads, grads_col, grads_u, grads_ub, grads_col_v, grads_v, grads_vb


    def fit(self):
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
        start_time = time.time()

        optimizer = optim.Adam(self.u_model.parameters(), lr=0.005, betas=(0.99, 0.999))
        optimizer_col_weights = optim.Adam([self.col_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_u_weights = optim.Adam([self.u_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_ub_weights = optim.Adam([self.ub_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_col_v_weights = optim.Adam([self.col_v_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_v_weights = optim.Adam([self.v_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_vb_weights = optim.Adam([self.vb_weights], lr=0.005, betas=(0.99, 0.999))
    
        print("starting Adam training")

        for epoch in range(self.tf_iter):
            for i in range(n_batches):
                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, grads_ub, grads_col_v, grads_v, grads_vb= self.g_grad(self.u_model, self.x_f, self.y_f, self.t_f, self.t0, self.u0, self.v0, self.x_lb, self.x_ub, self.y_lb, self.y_ub, self.u_x_lb, self.u_x_ub, self.u_y_lb, self.u_y_ub, self.v_x_lb, self.v_x_ub, self.v_y_lb, self.v_y_ub, self.XY, self.XT, self.YT, self.col_weights, self.u_weights, self.ub_weights, self.col_v_weights, self.v_weights, self.vb_weights)

                # Apply gradients to the model
                optimizer.zero_grad()
                for param, grad in zip(self.u_model.parameters(), grads):
                    param.grad = grad
                optimizer.step()
    
                # Apply gradients to col_weights and u_weights
                optimizer_col_weights.zero_grad()
                self.col_weights.grad = -grads_col
                optimizer_col_weights.step()
    
                optimizer_u_weights.zero_grad()
                self.u_weights.grad = -grads_u
                optimizer_u_weights.step()
    
                optimizer_ub_weights.zero_grad()
                self.ub_weights.grad = -grads_ub
                optimizer_ub_weights.step()
    
                optimizer_col_v_weights.zero_grad()
                self.col_v_weights.grad = -grads_col_v
                optimizer_col_v_weights.step()
    
                optimizer_v_weights.zero_grad()
                self.v_weights.grad = -grads_v
                optimizer_v_weights.step()
    
                optimizer_vb_weights.zero_grad()
                self.vb_weights.grad = -grads_vb
                optimizer_vb_weights.step()
    
            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                error_u_value, error_v_value = self.error_u()
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, Error: %.4e, Error v: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value, error_v_value))
                start_time = time.time()


    def fit_lbfgs(self):
    
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.LBFGS(self.u_model.parameters(), lr=0.8, tolerance_grad=1e-05, tolerance_change=1e-09)
    
        print("starting L-BFGS training")
   
        lossl_value = [] 
        for epoch in range(self.newton_iter):
            for i in range(n_batches):
                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, grads_ub, grads_col_v, grads_v, grads_vb = self.g_grad(self.u_model, self.x_f, self.y_f, self.t_f, self.t0, self.u0, self.v0, self.x_lb, self.x_ub, self.y_lb, self.y_ub, self.u_x_lb, self.u_x_ub, self.u_y_lb, self.u_y_ub, self.v_x_lb, self.v_x_ub, self.v_y_lb, self.v_y_ub, self.XY, self.XT, self.YT, self.col_weights, self.u_weights, self.ub_weights, self.col_v_weights, self.v_weights, self.vb_weights)
    
                def closure():
                    optimizer.zero_grad()
                    for param, grad in zip(self.u_model.parameters(), grads):
                        param.grad = grad
    
                    return loss_value       
                
                optimizer.step(closure)
    
    
            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                error_u_value, error_v_value = self.error_u()
    
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, Error: %.4e, Error v: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value, error_v_value))
    
                start_time = time.time()
                a1= lossl_value[-1]
                a2= lossl_value[-2]
                a3= lossl_value[-3]
                a4= lossl_value[-4]
                if epoch>10000 and abs((a1-a2)/a2)<1e-10 and abs((a2-a3)/a3)<1e-10 and abs((a3-a4)/a4)<1e-10:
                    break

            lossl_value.append(loss_value)#.item())
 

    def __Loadmat(self, fileName):
        data = scipy.io.loadmat(fileName)

        tt = data['t']
        t = tt.flatten()[:,None]
        x = data['x'].flatten()[:, None]
        y = data['y'].flatten()[:, None]
        self.x = x
        self.y = y
        self.t = t

        X, Y, T = np.meshgrid(x, y, t)
        self.X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))

        self.u_star_all=data["Exact_u"]
        self.v_star_all=data["Exact_v"]
        self.refer_u=data["Exact_u"] 
        self.refer_v=data["Exact_v"]

        u_star_ori=data["Exact_u"]
        v_star_ori=data["Exact_v"]

        self.u_star = u_star_ori.flatten()[:, None]
        self.v_star = v_star_ori.flatten()[:,None]

        lb = np.array([x.min(0), y.min(0), t.min(0)]).T
        ub = np.array([x.max(0), y.max(0), t.max(0)]).T
        X_f = lb + ((ub - lb) * lhs(3, self.N_f))

        XY_X, XY_Y = np.meshgrid(x, y)  
        XT_X, XT_T = np.meshgrid(x, t)
        YT_Y, YT_T = np.meshgrid(y, t)
        XY = np.hstack((XY_X.flatten()[:, None], XY_Y.flatten()[:, None]))
        XT = np.hstack((XT_X.flatten()[:, None], XT_T.flatten()[:, None]))
        YT = np.hstack((YT_Y.flatten()[:, None], YT_T.flatten()[:, None]))

        selected_indices = np.random.choice(XY.shape[0], N0, replace=False)
        selected_indicesxt = np.random.choice(XT.shape[0], Nb, replace=False)
        selected_indicesyt = np.random.choice(YT.shape[0], Nb, replace=False)
        XY = XY[selected_indices, :]
        YT = YT[selected_indicesyt, :]
        XT = XT[selected_indicesxt, :]

        u0all_ori= u_star_ori[:,:,0]
        u0all=u0all_ori.flatten()[:,None]
        u0 = u0all[selected_indices]

        u_x_lball_ori= u_star_ori[:,0,:].T
        u_x_uball_ori= u_star_ori[:,-1,:].T
        u_y_lball_ori= u_star_ori[0, :,:].T
        u_y_uball_ori= u_star_ori[-1,:,:].T

        u_x_lball =u_x_lball_ori.flatten()[:,None]
        u_x_uball =u_x_uball_ori.flatten()[:,None]
        u_y_lball =u_y_lball_ori.flatten()[:,None]
        u_y_uball =u_y_uball_ori.flatten()[:,None]

        v0all_ori= v_star_ori[:,:,0]
        v0all=v0all_ori.flatten()[:,None]
        v0 = v0all[selected_indices]

        v_x_lball_ori= v_star_ori[:,0,:].T
        v_x_uball_ori= v_star_ori[:,-1,:].T
        v_y_lball_ori= v_star_ori[0, :,:].T
        v_y_uball_ori= v_star_ori[-1,:,:].T

        v_x_lball =v_x_lball_ori.flatten()[:,None]
        v_x_uball =v_x_uball_ori.flatten()[:,None]
        v_y_lball =v_y_lball_ori.flatten()[:,None]
        v_y_uball =v_y_uball_ori.flatten()[:,None]

        u_x_lb = u_x_lball[selected_indicesyt]
        u_x_ub = u_x_uball[selected_indicesyt]
        u_y_lb = u_y_lball[selected_indicesxt]
        u_y_ub = u_y_uball[selected_indicesxt]

        v_x_lb = v_x_lball[selected_indicesyt]
        v_x_ub = v_x_uball[selected_indicesyt]
        v_y_lb = v_y_lball[selected_indicesxt]
        v_y_ub = v_y_uball[selected_indicesxt]

        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u_x_lb = torch.tensor(u_x_lb, dtype=torch.float32)
        self.u_x_ub = torch.tensor(u_x_ub, dtype=torch.float32)
        self.u_y_lb = torch.tensor(u_y_lb, dtype=torch.float32)
        self.u_y_ub = torch.tensor(u_y_ub, dtype=torch.float32)

        self.v0 = torch.tensor(v0, dtype=torch.float32)
        self.v_x_lb = torch.tensor(v_x_lb, dtype=torch.float32)
        self.v_x_ub = torch.tensor(v_x_ub, dtype=torch.float32)
        self.v_y_lb = torch.tensor(v_y_lb, dtype=torch.float32)
        self.v_y_ub = torch.tensor(v_y_ub, dtype=torch.float32)

        self.XY = torch.tensor(XY, dtype=torch.float32)
        self.XT = torch.tensor(XT, dtype=torch.float32)
        self.YT = torch.tensor(YT, dtype=torch.float32)
        self.X_f = torch.tensor(X_f, dtype=torch.float32)
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32)
        self.y_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32)
        self.t_f = torch.tensor(X_f[:, 2:3], dtype=torch.float32)
        self.t0 = torch.tensor(np.repeat(t.min(0), XY[:, 1:2].shape[0]).reshape((XY[:, 1:2].shape[0], -1)), dtype=torch.float32)
        self.x_lb = torch.tensor(np.repeat(x.min(0), YT[:, 1:2].shape[0]).reshape((YT[:, 1:2].shape[0], -1)), dtype=torch.float32)
        self.x_ub = torch.tensor(np.repeat(x.max(0), YT[:, 1:2].shape[0]).reshape((YT[:, 1:2].shape[0], -1)), dtype=torch.float32)
        self.y_lb = torch.tensor(np.repeat(y.min(0), XT[:, 1:2].shape[0]).reshape((XT[:, 1:2].shape[0], -1)), dtype=torch.float32)
        self.y_ub = torch.tensor(np.repeat(y.max(0), XT[:, 1:2].shape[0]).reshape((XT[:, 1:2].shape[0], -1)), dtype=torch.float32)


    def error_u(self):
        self.u_model.eval()
    
        X_star = torch.tensor(self.X_star, dtype=torch.float32, device=device)
    
        with torch.no_grad():
            u_pred, v_pred = self.uv_model(X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3])
    
        u_star = torch.tensor(self.u_star, dtype=torch.float32, device=device)
        v_star = torch.tensor(self.v_star, dtype=torch.float32, device=device)
    
        # Calculate the error
        error_u = torch.norm(u_star - u_pred, p=2) / torch.norm(u_star, p=2)
        error_v = torch.norm(v_star - v_pred, p=2) / torch.norm(v_star, p=2)
    
        return error_u.item(), error_v.item()

    def predict(self, x,y,t):
        self.u_model.eval() 
    
        X,Y,T=np.meshgrid(x,y,t)
        XX_star=np.hstack((X.flatten()[:, None],Y.flatten()[:, None],T.flatten()[:, None]))
        X_star = torch.tensor(XX_star, dtype=torch.float32, device=device, requires_grad=True)
    
        with torch.no_grad(): 
            u_star, v_star = self.uv_model(X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3])
    
        f_u_star, f_v_star = self.f_model(self.u_model, X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3])
    
        return u_star.detach().cpu().numpy(), f_u_star.detach().cpu().numpy(), v_star.detach().cpu().numpy(), f_v_star.detach().cpu().numpy()

