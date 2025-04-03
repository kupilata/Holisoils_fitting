#source tf-env/bin/activate  #to activate the virtual environment (from command line)

# import the numerical object library
import pandas as pd

# import the libraries for sampling
import torch
print(torch.backends.mps.is_available())  # Should return True if MPS is available
print(torch.backends.mps.is_built()) #True if PyTorch was built with Metal support.
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import time # for benchmarking

# load the dataset and explore it
fitting_data = pd.read_csv("python_holisoils.csv")
fitting_data.shape
fitting_data.columns
fitting_data.dtypes
fitting_data.describe()
fitting_data.head()
fitting_data.tail()

# Extract relevant columns from the data
N = len(fitting_data)
Tr = fitting_data['treatment'].max()
Pl = fitting_data['plot_id'].max()
treatment = torch.tensor(fitting_data['treatment'].values, dtype=torch.float32)
plot_id = torch.tensor(fitting_data['plot_id'].values, dtype=torch.float32)
day_year = torch.tensor(fitting_data['day_year'].values, dtype=torch.float32)
temp = torch.tensor(fitting_data['temp'].values, dtype=torch.float32)
resp = torch.tensor(fitting_data['resp'].values, dtype=torch.float32)
M = torch.tensor(fitting_data['M_observed'].values, dtype=torch.float32)
Q10_range = torch.tensor([10.0, 20.0])  # Example range, adjust as needed

# Check if MPS is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def model(N, Tr, Pl, treatment, plot_id, day_year, temp, resp, M, Q10_range, device):
    T_0 = 227.13
    sigma = 0.2712706

    # Sample parameters
    A = pyro.sample('A', dist.Normal(torch.tensor(400.0, device=device), torch.tensor(100.0, device=device)).expand([Pl]))
    Ea = pyro.sample('Ea', dist.Normal(torch.tensor(398.5, device=device), torch.tensor(50.0, device=device)).expand([Tr]))
    a = pyro.sample('a', dist.Normal(torch.tensor(3.11, device=device), torch.tensor(0.25, device=device)).expand([Tr]))
    b = pyro.sample('b', dist.Normal(torch.tensor(2.42, device=device), torch.tensor(0.25, device=device)).expand([Tr]))
    amplitude = pyro.sample('amplitude', dist.Normal(torch.tensor(0.0, device=device), torch.tensor(0.25, device=device)).expand([Tr]))
    peak_day = pyro.sample('peak_day', dist.Normal(torch.tensor(196.0, device=device), torch.tensor(50.0, device=device)).expand([Tr]))
    
    # Initialize response tensor
    model_resp = torch.zeros(N, device=device)
    
    for i in range(N):
        tr_idx = int(treatment[i] - 1)  # Convert to integer index
        pl_idx = int(plot_id[i] - 1)  # Convert to integer index
        
        # Compute xi_moist, xi_temp, and sine_wave
        xi_moist = a[tr_idx] * M[i] - b[tr_idx] * M[i]**2
        xi_temp = A[pl_idx] * torch.exp(-Ea[tr_idx] / ((temp[i] + 273.15) - T_0))
        sine_wave = amplitude[tr_idx] * torch.cos((2 * torch.pi / 365) * day_year[i] + (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) - torch.pi / 2)
        
        # Debug prints
        print(f"Index {i}: tr_idx={tr_idx}, pl_idx={pl_idx}")
        print(f"xi_moist[{i}]: {xi_moist}")
        print(f"xi_temp[{i}]: {xi_temp}")
        print(f"sine_wave[{i}]: {sine_wave}")
        
        # Compute modeled response
        model_resp[i] = sine_wave + (xi_temp * xi_moist)
    
    # Observe response
    pyro.sample('obs', dist.Normal(model_resp, sigma), obs=resp)
    
    return model_resp

# Ensure tensors are on the correct device
treatment = treatment.to(device)
plot_id = plot_id.to(device)
day_year = day_year.to(device)
temp = temp.to(device)
resp = resp.to(device)
M = M.to(device)
Q10_range = Q10_range.to(device)

# Run the model
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)

# Measure execution time for device="cpu"
# start_time_cpu = time.time()
# mcmc.run(N, Tr, Pl, treatment, plot_id, day_year, temp, resp, M, Q10_range, device="cpu")
# end_time_cpu = time.time()
# execution_time_cpu = end_time_cpu - start_time_cpu

# Measure execution time for device="mps"
start_time_mps = time.time()
mcmc.run(N, Tr, Pl, treatment, plot_id, day_year, temp, resp, M, Q10_range, device="mps")
end_time_mps = time.time()
execution_time_mps = end_time_mps - start_time_mps


with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    mcmc.run(N, Tr, Pl, treatment, plot_id, day_year, temp, resp, M, Q10_range, device="mps")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Print the execution times
# print(f"Execution time for device='cpu': {execution_time_cpu} seconds")
print(f"Execution time for device='mps': {execution_time_mps} seconds")

mcmc.summary()