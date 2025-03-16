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
from ISA_PINN import ISA_PINN
import torch.autograd as autograd

#torch.manual_seed(1234)
#np.random.seed(1234)

def loss(sapinn:ISA_PINN, x_f_batch, t_f_batch, x0, t0, u0, v0, x_lb, t_lb, x_ub, t_ub, u_lb, u_ub, v_lb, v_ub, col_weights, u_weights, u_lb_weights, u_ub_weights, col_v_weights, v_weights, v_lb_weights, v_ub_weights):
    u0_pred, v0_pred = sapinn.uv_model(x0,t0)#u_model(torch.cat([x0, t0], 1))
    u_lb_pred, v_lb_pred = sapinn.uv_model(x_lb, t_lb)#u_x_model(u_model, x_lb, t_lb)
    u_ub_pred, v_ub_pred = sapinn.uv_model(x_ub, t_ub)#u_x_model(u_model, x_ub, t_ub)
    f_u_pred,f_v_pred = f_model(sapinn, x_f_batch, t_f_batch)
    
    mse_0_u = torch.mean((u_weights * (u0 - u0_pred))**2)+ torch.mean((v_weights * (v0 - v0_pred))**2)
    mse_b_u = torch.mean((u_lb_weights*(u_lb - u_lb_pred))**2) + torch.mean((u_ub_weights*(u_ub - u_ub_pred))**2)
    mse_b_v = torch.mean((v_lb_weights*(v_lb - v_lb_pred))**2) + torch.mean((v_ub_weights*(v_ub - v_ub_pred))**2)
    mse_f_u = torch.mean((col_weights * f_u_pred)**2) + torch.mean((col_v_weights * f_v_pred)**2)
    
    return mse_0_u + mse_b_u + mse_b_v + mse_f_u, mse_0_u, mse_b_u + mse_b_v, mse_f_u

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(sapinn:ISA_PINN, x, t):
    u,v = sapinn.uv_model(x,t)
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    v_t = autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    
    f_u=u_t + v_xx + 2.*(u**2 + v**2)*v
    f_v=v_t - u_xx - 2.*(u**2 + v**2)*u
    
    return f_u,f_v

for id_t in range(Nm):
    model=ISA_PINN("data.mat",layer_sizes,tf_iter1,newton_iter1,f_model=f_model,Loss=loss,N_f= n_f)

    model.fit()
    model.fit_lbfgs()
    
    u_pred,v_pred, f_u_pred,f_v_pred = model.predict()
    
    U3_pred = u_pred.reshape((Nt, Nx)).T
    V3_pred = v_pred.reshape((Nt, Nx)).T
    f_U3_pred = f_u_pred.reshape((Nt, Nx)).T
    f_V3_pred = f_v_pred.reshape((Nt, Nx)).T

    Exact_u_i = model.Exact_u
    Exact_v_i = model.Exact_v
    perror_u  = np.linalg.norm((Exact_u_i - U3_pred).flatten(),2)
    perror_v  = np.linalg.norm((Exact_v_i - V3_pred).flatten(),2)

    perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)
    perror_vEx = np.linalg.norm(Exact_v_i.flatten(),2)
 
    error_u = perror_u/perror_uEx
    error_v = perror_v/perror_vEx
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
 
