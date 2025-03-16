import torch

n_f = 20000
lrate = 0.005

Nm = 1
Nx = 257
Nt = 101

N0 = 200
Nb = 50

tf_iter1= 15000
newton_iter1= 60000

num_layer=4
width=80
layer_sizes=[2]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

