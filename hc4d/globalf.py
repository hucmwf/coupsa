import numpy as np
import torch

n_f = 20000 #[20000]
lrate = 0.005

Nm = 1
N0 = 3000
Nb = 1000

tf_iter1= 30000
newton_iter1= 30000

num_layer=5
width=48
layer_sizes=[4]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

