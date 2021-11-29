import csv
import os
import torch
import torch.nn.functional as F

# set default variables
seed = 0
data_type = torch.float32

torch.manual_seed(seed)
torch.set_default_dtype(data_type)
torch.use_deterministic_algorithms(True)

# generate input
H = 5
W = 6
C = 3
M = 1
KH = 2
KW = 2
SW = 1
SH = 1
PH = 0
PW = 0

# transpose convolution
input = torch.rand((1, C, H, W))
kernel = torch.rand((C, M, KH, KW))
output = F.conv_transpose2d(input, kernel, padding=(PH, PW), stride=(SH, SW))

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

if not os.path.exists(os.path.join(__location__, '../data')):
    os.makedirs(os.path.join(__location__, '../data'))

with open(os.path.join(__location__, '../data/dim.csv'), 'w', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow([H, W, C, M, KH, KW, SW, SH, PH, PW])
    writer.writerow(input.numpy().flatten().tolist())
    writer.writerow(kernel.numpy().flatten().tolist())
    writer.writerow(output.numpy().flatten().tolist())
