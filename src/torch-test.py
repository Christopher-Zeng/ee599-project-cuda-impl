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

# generate input
H = 5
W = 6
C = 3
M = 1
K = 2

# set cuda
cuda = torch.device('cuda')


# transpose convolution
input = torch.rand((1, C, H, W)).cuda()
kernel = torch.rand((C, M, K, K)).cuda()

start = datetime.datetime.now()
output = F.conv_transpose2d(input, kernel, stride=1).cuda()
end = datetime.datetime.now()
elapsed = end - start
print('time used by pytorch: {}ms'.format(elapsed.microseconds))