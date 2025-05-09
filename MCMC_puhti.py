#source /scratch/project_2010938/venv_mine/bin/activate

# import the libraries
import torch
import time

# Check CUDA availability
cuda_available = torch.cuda.is_available()
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "No CUDA device"

# Print to console
print("CUDA available:", cuda_available)
print("CUDA device name:", cuda_device_name)
print(torch.version.cuda)          # e.g., '11.7'
print(torch.backends.cudnn.enabled)  # Should be True

# Save to file
with open("cuda_status.txt", "w") as f:
    f.write(f"CUDA available: {cuda_available}\n")
    f.write(f"CUDA device name: {cuda_device_name}\n")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import time

# Check for CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("CUDA available:", use_cuda)
if use_cuda:
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# Load dataset
fitting_data = pd.read_csv("python_holisoils.csv")
print(fitting_data.shape)
print(fitting_data.columns)

# Extract relevant columns
N = len(fitting_data)
Tr = fitting_data['treatment'].max()
Pl = fitting_data['plot_id'].max()

# Ensure tensors are on the correct device
def to_tensor(col, device):
    return torch.tensor(fitting_data[col].values, dtype=torch.float32, device=device)

# Extract data and move to device
treatment = to_tensor('treatment', device)
plot_id = to_tensor('plot_id', device)
day_year = to_tensor('day_year', device)
temp = to_tensor('temp', device)
resp = to_tensor('resp', device)
M = to_tensor('M_observed', device)
Q10_range = torch.tensor([10.0, 20.0], device=device)

def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M, device):
    T_0 = 227.13
    sigma = pyro.sample('sigma', dist.Normal(torch.tensor(0.2712706, device=device), 0.05))

    A = pyro.sample('A', dist.Normal(torch.tensor(400.0, device=device), 100.0).expand([Pl]).to_event(1))
    Ea = pyro.sample('Ea', dist.Normal(torch.tensor(398.5, device=device), 50.0).expand([Tr]).to_event(1))
    a = pyro.sample('a', dist.Normal(torch.tensor(3.11, device=device), 0.25).expand([Tr]).to_event(1))
    b = pyro.sample('b', dist.Normal(torch.tensor(2.42, device=device), 0.25).expand([Tr]).to_event(1))
    amplitude = pyro.sample('amplitude', dist.Normal(torch.tensor(0.0, device=device), 0.25).expand([Tr]).to_event(1))
    peak_day = pyro.sample('peak_day', dist.Normal(torch.tensor(196.0, device=device), 50.0).expand([Tr]).to_event(1))

    tr_idx = (treatment - 1).long()
    pl_idx = (plot_id - 1).long()

    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_temp = A[pl_idx] * torch.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0))
    sine_wave = amplitude[tr_idx] * torch.cos((2 * torch.pi / 365) * day_year +
                                              (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) - torch.pi / 2)

    model_resp = sine_wave + (xi_temp * xi_moist)

    pyro.sample('obs', dist.Normal(model_resp, sigma), obs=resp)

    return model_resp


# Run MCMC with NUTS
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=200, num_chains=1)  # For CUDA, start with 1 chain

# Run and benchmark
start_time = time.time()
mcmc.run(Tr, Pl, treatment, plot_id, day_year, temp, resp, M, device)
end_time = time.time()

print(f"\nExecution time on device='{device}': {end_time - start_time:.2f} seconds")
