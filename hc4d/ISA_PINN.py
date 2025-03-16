import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import scipy.io
from pyDOE import lhs
import pickle
import os
from globalf import *
from Data import Data
import datetime

class ISA_PINN:
    def __DefaultLoss(self,x_f_batch, t_f_batch,
             x0, t0, u0,u_lb,u_ub, x_lb,
             t_lb, x_ub, t_ub, col_weights, u_weights):
        u0_pred = self.u_model(torch.cat([x0, t0], 1))
        u_lb_pred, u_x_lb_pred = self.u_x_model(self.model, x_lb, t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.model, x_ub, t_ub)
        f_u_pred = self.f_model(self.model, x_f_batch, t_f_batch)
        
        mse_0_u = torch.mean((u_weights * (u0 - u0_pred))**2)
        mse_b_u = torch.mean((u_lb_pred - u_ub_pred)**2) + torch.mean((u_x_lb_pred - u_x_ub_pred)**2)
        mse_f_u = torch.mean((col_weights * f_u_pred)**2)
        
        return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

    def u_model(self, x, t):
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        u = self.model(torch.cat([x, t], dim=1))
        return u

    def __init__(self,data:Data,layers:list,adam_iter:int,newton_iter:int,f_model,Loss=__DefaultLoss,lbfgs_lr=0.8,N_f=10000,checkPointPath="./checkPoint"):
        self.N_f=N_f
        self.data=data
        
        self.layers=layers
        self.sizes_w=[]
        self.sizes_b=[]
        self.lbfgs_lr=lbfgs_lr
        self.SA_weights={
            'col_weights':nn.Parameter(torch.full((N_f, 1), 100.0, device=device)),
            'u_weights':nn.Parameter(torch.full((self.data.x0.shape[0], 1), 100.0, device=device)),         
                         }

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

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

        self.model = NeuralNet(self.layers)
        print(self.model)
        self.model = self.model.cuda()

        self.Loss=Loss
        self.adam_iter=adam_iter
        self.newton_iter=newton_iter
        self.f_model=f_model

    def ggrad(self, model, inputs_f, inputs_0, lables_0, lables_b,inputs_b):
        inputs_f=[input.to(device).requires_grad_(True) for input in inputs_f]
        inputs_0=[input.to(device).requires_grad_(True) for input in inputs_0]

        for key,inputs in inputs_b.items():
            inputs_b[key]=[input.to(device).requires_grad_(True) for input in inputs]
        for key,lables in lables_b.items():
            for subkey,lable in lables.items():
                lables[subkey]=lable.to(device).requires_grad_(True)
            lables_b[key]=lables

        # x_f_batch = x_f_batch.to(device).requires_grad_(True)
        # t_f_batch = t_f_batch.to(device).requires_grad_(True)
        # x0_batch = x0_batch.to(device).requires_grad_(True)
        # t0_batch = t0_batch.to(device).requires_grad_(True)
        for key,lable in lables_0.items():
            lables_0[key]=lable.to(device).requires_grad_(True)

        for key,SA_weight in self.SA_weights.items():
            self.SA_weights[key]=SA_weight.to(device).requires_grad_(True)
        # x_lb = x_lb.to(device).requires_grad_(True)
        # t_lb = t_lb.to(device).requires_grad_(True)
        # x_ub = x_ub.to(device).requires_grad_(True)
        # t_ub = t_ub.to(device).requires_grad_(True)
    
        model.zero_grad()
        loss_value, mse_0, mse_b, mse_f = self.Loss(self, inputs_f, inputs_0, lables_0, lables_b, inputs_b, self.SA_weights)
        loss_value.backward(retain_graph=True)
        grads = [param.grad.clone() for param in model.parameters()]
        model.zero_grad()
    
        loss_value.backward(retain_graph=True)
        SA_grads=[SA_weight.grad.clone() for key,SA_weight in self.SA_weights.items()]
    
        return loss_value.item(), mse_0.item(), mse_b.item(), mse_f.item(), grads, SA_grads

    def fit(self):
    
        # Set batch size for collocation points
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.005, betas=(0.99, 0.999))
        SA_optimizers=[optim.Adam([SA_weight], lr=0.005, betas=(0.99, 0.999)) for key,SA_weight in self.SA_weights.items()]
        print("starting Adam training")
    
        # For mini-batch (if used)
        for epoch in range(self.adam_iter):
            for i in range(n_batches):
    
                # x0_batch = torch.tensor(self.x0, dtype=torch.float32)
                # t0_batch = torch.tensor(self.t0, dtype=torch.float32)
                # #u0_batch = torch.tensor(self.u0, dtype=torch.float32)
    
                # x_f_batch = torch.tensor(self.x_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
                # t_f_batch = torch.tensor(self.t_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
    
                loss_value, mse_0, mse_b, mse_f, grads, SA_grads = self.ggrad(self.model, self.data.inputs_f, self.data.inputs_0, self.data.lables_0, self.data.lables_b, self.data.inputs_b)
    
                optimizer.zero_grad()
                for param, grad in zip(self.model.parameters(), grads):
                    param.grad = grad
                optimizer.step()

                index=0
                for key,SA_weights in self.SA_weights.items(): 
                    SA_optimizers[index].zero_grad()
                    SA_weights.grad = -SA_grads[index]
                    SA_optimizers[index].step()
                    index+=1
    
            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value))
    
                start_time = time.time()

    def fit_lbfgs(self):
    
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8, tolerance_grad=1e-07, tolerance_change=1e-011)
    
        print("starting L-BFGS training")
    
        for epoch in range(self.newton_iter):
            for i in range(n_batches):
                loss_value, mse_0, mse_b, mse_f, grads, SA_grads = self.ggrad(self.model, self.data.inputs_f, self.data.inputs_0, self.data.lables_0, self.data.lables_b, self.data.inputs_b)
    
                def closure():
                    optimizer.zero_grad()
                    for param, grad in zip(self.model.parameters(), grads):
                        param.grad = grad
    
                    return loss_value       
                
                optimizer.step(closure)
    
            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value))
    
                start_time = time.time()

    def predict(self):
        X_star = torch.tensor(self.data.X_star, dtype=torch.float32, device=device, requires_grad=True)
        u_star = self.u_model(X_star[:, 0:1], X_star[:, 1:2])

        X_star = X_star.clone().detach().requires_grad_(True)
        f_u_star = self.f_model(self, X_star[:, 0:1], X_star[:, 1:2])

        u_star = u_star.detach().cpu().numpy()
        f_u_star = f_u_star.detach().cpu().numpy()

        return u_star, f_u_star

