import csv
import os
import torch
import torch.nn.functional as F

# set default variables
seed = 0
data_type = torch.float32

torch.manual_seed(seed)
torch.set_default_dtype(data_type)

# generate input
H = 5
W = 6
C = 3
M = 1
K = 2

# transpose convolution
input = torch.rand((1, C, H, W))
kernel = torch.rand((C, M, K, K))
output = F.conv_transpose2d(input, kernel, stride=1)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, '../data/dim.csv'), 'w', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow([H, W, C, M, K])
    writer.writerow(input.numpy().flatten().tolist())
    writer.writerow(kernel.numpy().flatten().tolist())
    writer.writerow(output.numpy().flatten().tolist())
