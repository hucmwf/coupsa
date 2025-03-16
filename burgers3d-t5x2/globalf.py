import torch

n_f = 20000
lrate = 0.005
N0 = 7000
Nb = 3000

Nm = 1
Nx, Ny, Nt= 103, 105, 101
N_x, N_y, N_t= 103, 105, 101
    
tf_iter1=10000
newton_iter1=30000

num_layer=7
width= 48
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

