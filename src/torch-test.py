import csv
import os
import datetime
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

# set cuda
cuda = torch.device('cuda')


# transpose convolution
input = torch.rand((1, C, H, W)).cuda()
kernel = torch.rand((C, M, KH, KW)).cuda()

start = datetime.datetime.now()
output = F.conv_transpose2d(input, kernel, padding=(PH, PW), stride=(SH, SW)).cuda()
end = datetime.datetime.now()
elapsed = end - start
print('time used by pytorch: {}ms'.format(elapsed.microseconds))