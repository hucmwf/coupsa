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
from ISA_PINN3D import*
from torch import pi,cos,sin

#torch.manual_seed(1234)
#np.random.seed(1234)

def loss(isapinn:ISAPINN3D, inputs_f, inputs_0, lables_0, lables_b,inputs_b, SA_weights):
    u0_pred = isapinn.u_model(inputs_0[0],inputs_0[1],inputs_0[2],inputs_0[3])
    u_x_lb_pred = isapinn.u_model(inputs_b['x_lb'][0], inputs_b['x_lb'][1],inputs_b['x_lb'][2],inputs_b['x_lb'][3])
    u_x_ub_pred = isapinn.u_model(inputs_b['x_ub'][0], inputs_b['x_ub'][1],inputs_b['x_ub'][2],inputs_b['x_ub'][3])
    u_y_ub_pred = isapinn.u_model(inputs_b['y_ub'][0], inputs_b['y_ub'][1],inputs_b['y_ub'][2],inputs_b['y_ub'][3])
    u_y_lb_pred = isapinn.u_model(inputs_b['y_lb'][0], inputs_b['y_lb'][1],inputs_b['y_lb'][2],inputs_b['y_lb'][3])
    u_z_ub_pred = isapinn.u_model(inputs_b['z_ub'][0], inputs_b['z_ub'][1],inputs_b['z_ub'][2],inputs_b['z_ub'][3])
    u_z_lb_pred = isapinn.u_model(inputs_b['z_lb'][0], inputs_b['z_lb'][1],inputs_b['z_lb'][2],inputs_b['z_lb'][3])
    f_u_pred= f_model(isapinn, inputs_f[0], inputs_f[1],inputs_f[2],inputs_f[3])

    u0=lables_0['u']
    u_x_lb,u_x_ub,u_y_lb,u_y_ub,u_z_lb,u_z_ub=lables_b['u']['x_lb'],lables_b['u']['x_ub'],lables_b['u']['y_lb'],lables_b['u']['y_ub'],lables_b['u']['z_lb'],lables_b['u']['z_ub']

    mse_0_u = torch.mean((SA_weights['u_weights'] * (u0 - u0_pred))**2)
    mse_b_u = torch.mean((SA_weights['u_b_weights']*(u_x_lb_pred - u_x_lb))**2) +\
          torch.mean((SA_weights['u_b_weights']*(u_x_ub_pred - u_x_ub))**2)+\
          torch.mean((SA_weights['u_b_weights']*(u_y_lb_pred - u_y_lb))**2)+\
          torch.mean((SA_weights['u_b_weights']*(u_y_ub_pred - u_y_ub))**2)+\
          torch.mean((SA_weights['u_b_weights']*(u_z_lb_pred - u_z_lb))**2)+\
          torch.mean((SA_weights['u_b_weights']*(u_z_ub_pred - u_z_ub))**2)
    mse_f_u = torch.mean((SA_weights['col_weights'] * f_u_pred)**2)
    
    return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(isapinn:ISAPINN3D, x,y,z, t):
    u= isapinn.u_model(x,y,z,t)
    q= 5*pi*cos(5*pi*t+pi*x*y*z)-pi**2*(x**2*y**2+x**2*z**2+y**2*z**2)*(cos(10*pi*t+2*pi*x*y*z)-2*sin(5*pi*t +pi*x*y*z))
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_z = autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_zz = autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    f_u=u_t -u_x**2 -u*u_xx-u_y**2-u*u_yy-u_z**2-u*u_zz-q

    return f_u

for id_t in range(Nm):
    data=Data3D('hcdata.mat')
    model=ISAPINN3D(data,layer_sizes,tf_iter1,newton_iter1,f_model=f_model,Loss=loss,N_f= n_f)

    model.fit()
    model.fit_lbfgs()
   
    error_u_value = model.error_u()
    print('Error u: %e' % (error_u_value))

