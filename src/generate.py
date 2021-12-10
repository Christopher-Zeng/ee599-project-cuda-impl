import os
from numpy.core.fromnumeric import reshape
import torch
from torch._C import ParameterDict
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime

# set default variables
seed = 0
data_type = torch.float32

torch.manual_seed(seed)
torch.set_default_dtype(data_type)
# torch.use_deterministic_algorithms(True)

params = {}
params["N"] = 64
params["H"] = 32
params["W"] = 64
params["C"] = 256
params["M"] = params["C"]
params["KH"] = 7
params["KW"] = params["KH"]
params["SH"] = 7
params["SW"] = params["SH"]
params["PH"] = params["SH"]
params["PW"] = params["PH"]
params["OH"] = params["SH"] * (params["H"] - 1) + params["KH"] - 2 * params["PH"]
params["OW"] = params["SW"] * (params["W"] - 1) + params["KW"] - 2 * params["PW"]

data_dir = "./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


kernel = torch.rand((params["C"], params["M"], params["KH"], params["KW"]))
pd.DataFrame(
    kernel.numpy().reshape((params["C"] * params["M"], params["KH"] * params["KW"]))
).to_csv(os.path.join(data_dir, "kernel.csv"), index=None)

execution_time = []

for n in range(params["N"]):

    print("Epoch {}.".format(n))
    input = torch.rand((1, params["C"], params["H"], params["W"]))
    pd.DataFrame(
        input.permute((0, 2, 3, 1))
        .numpy()
        .reshape((params["H"] * params["W"], params["C"])),
    ).to_csv(os.path.join(data_dir, "input_{}.csv".format(n)), index=None)

    start = datetime.datetime.now()
    with torch.cuda.device(0):
        input_cuda = input.cuda()
        kernel_cuda = kernel.cuda()
        output_cuda = F.conv_transpose2d(
            input,
            kernel,
            padding=(params["PH"], params["PW"]),
            stride=(params["SH"], params["SW"]),
        )
        output = output_cuda.cpu()
    end = datetime.datetime.now()

    pd.DataFrame(
        output.permute(0, 2, 3, 1)
        .numpy()
        .reshape((params["OH"] * params["OW"], params["M"]))
    ).to_csv(os.path.join(data_dir, "output_{}.csv".format(n)), index=None)
    execution_time.append(start - end)

avg_execution_time = sum([time.microseconds for time in execution_time]) / len(
    execution_time
)
print("Pytorch: Average time per sample: {}ms".format(avg_execution_time))

# 1504MiB
# Pytorch: Average time per sample: 930492.671875ms
