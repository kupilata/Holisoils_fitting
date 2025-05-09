# source tf-env/bin/activate  #to activate the virtual environment 
# (from command line)

# import the numerical object library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

# import the libraries for sampling
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import time #for benchmarking

# Should return True if MPS is available
print(torch.backends.mps.is_available())
# True if PyTorch was built with Metal support. 
print(torch.backends.mps.is_built()) 

# load the dataset and explore it
fitting_data = pd.read_csv("python_holisoils.csv")
#fitting_data = fitting_data.sample(n=1000, random_state=42)  # Set random_state for reproducibility, resampling for testing!!!!!!
fitting_data.shape
fitting_data.columns
fitting_data.dtypes
fitting_data.describe()
fitting_data.head()
fitting_data.tail()

# Check if MPS is available
device = torch.device('cpu')

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

#Tr = Tr.to(device)
#Pl = Pl.to(device)
treatment = treatment.to(device)
plot_id = plot_id.to(device)
day_year = day_year.to(device)
temp = temp.to(device)
resp = resp.to(device)
M = M.to(device)

def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M, device):
    T_0 = 227.13
    sigma = pyro.sample('sigma', dist.Normal(0.2712706, 0.05))  # Allow sigma to vary

    # Sample parameters
    A = pyro.sample('A', dist.Normal(torch.tensor(400.0, device=device), torch.tensor(100.0, device=device)).expand([Pl]).to_event(1))
    Ea = pyro.sample('Ea', dist.Normal(torch.tensor(398.5, device=device), torch.tensor(50.0, device=device)).expand([Tr]).to_event(1))
    a = pyro.sample('a', dist.Normal(torch.tensor(3.11, device=device), torch.tensor(0.25, device=device)).expand([Tr]).to_event(1))
    b = pyro.sample('b', dist.Normal(torch.tensor(2.42, device=device), torch.tensor(0.25, device=device)).expand([Tr]).to_event(1))
    amplitude = pyro.sample('amplitude', dist.Normal(torch.tensor(0.0, device=device), torch.tensor(0.25, device=device)).expand([Tr]).to_event(1))
    peak_day = pyro.sample('peak_day', dist.Normal(torch.tensor(196.0, device=device), torch.tensor(50.0, device=device)).expand([Tr]).to_event(1))

    # Indexing
    tr_idx = (treatment - 1).long()  # Convert to integer index
    pl_idx = (plot_id - 1).long()  # Convert to integer index

    # Compute model response vectorized
    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_temp = A[pl_idx] * torch.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0))
    sine_wave = amplitude[tr_idx] * torch.cos((2 * torch.pi / 365) * day_year + (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) - torch.pi / 2)

    # Modeled response
    model_resp = sine_wave + (xi_temp * xi_moist)

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
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=200, num_chains=3)

# Measure execution time for device="mps"
start_time_cpu = time.time()
mcmc.run(Tr, Pl, treatment, plot_id, day_year, temp, resp, M, device="cpu")
end_time_cpu = time.time()
execution_time_cpu = end_time_cpu - start_time_cpu

# Print the execution times
print(f"Execution time for device='cpu': {execution_time_cpu} seconds")



# Extract the samples
samples = mcmc.get_samples()

# Calculate the summary statistics, including Rhat
summary = mcmc.summary()
samples = mcmc.get_samples()
sampled_parameters = list(samples.keys())

# Print the Rhat values
print("Rhat values for each parameter:")
for param, values in summary.items():
    print(f"{param}: Rhat = {values['r_hat']}")

param_samples_A = samples['A']
param_samples_Ea = samples['Ea']
param_samples_a = samples['a']
param_samples_b = samples['b']
param_samples_amplitude = samples['amplitude']
param_samples_peak_day = samples['peak_day']



# Convert the tensor to a NumPy array
param_samples_Ea_np = param_samples_Ea.cpu().numpy()

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=param_samples_Ea_np, width=0.5)

# Add labels and title
plt.xlabel('Parameter')
plt.ylabel('Sample Value')
plt.title('Boxplot of Sampled Parameters (Ea)')

# Show the plot
plt.show()



### Trying metal
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Ensure all variables are PyTorch tensors and move them to MPS
def to_tensor(var):
    return torch.tensor(var, dtype=torch.float32, device=device) if not isinstance(var, torch.Tensor) else var.to(device)

N = to_tensor(N)
Tr = to_tensor(Tr)
Pl = to_tensor(Pl)
treatment = to_tensor(treatment)
plot_id = to_tensor(plot_id)
day_year = to_tensor(day_year)
temp = to_tensor(temp)
resp = to_tensor(resp)
M = to_tensor(M)
Q10_range = to_tensor(Q10_range)

# Check device placement
for name, tensor in [("N", N), ("Tr", Tr), ("Pl", Pl), ("treatment", treatment), 
                     ("plot_id", plot_id), ("day_year", day_year), ("temp", temp), 
                     ("resp", resp), ("M", M), ("Q10_range", Q10_range)]:
    print(f"{name}: {tensor.device}")


use_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Move all input tensors to the specified device
Tr = Tr.to(use_device)
Pl = Pl.to(use_device)
treatment = treatment.to(use_device)
plot_id = plot_id.to(use_device)
day_year = day_year.to(use_device)
temp = temp.to(use_device)
resp = resp.to(use_device)
M = M.to(use_device)

#redefining the model to use mps
def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M, use_device):
 
    T_0 = 227.13
    sigma = pyro.sample('sigma', dist.Normal(torch.tensor(0.2712706, device=use_device), torch.tensor(0.05, device=use_device)))
    #print(f"sigma device: {sigma.device}")

    Pl_len = int(Pl.item()) if isinstance(Pl, torch.Tensor) else int(Pl)
    Tr_len = int(Tr.item()) if isinstance(Tr, torch.Tensor) else int(Tr)
    A = pyro.sample('A', dist.Normal(torch.tensor(400.0, device=use_device), torch.tensor(100.0, device=use_device)).expand([int(Pl_len)]).to_event(1))
    Ea = pyro.sample('Ea', dist.Normal(torch.tensor(398.5, device=use_device), torch.tensor(50.0, device=use_device)).expand([int(Tr_len)]).to_event(1))
    a = pyro.sample('a', dist.Normal(torch.tensor(3.11, device=use_device), torch.tensor(0.25, device=use_device)).expand([int(Tr_len)]).to_event(1))
    b = pyro.sample('b', dist.Normal(torch.tensor(2.42, device=use_device), torch.tensor(0.25, device=use_device)).expand([int(Tr_len)]).to_event(1))
    amplitude = pyro.sample('amplitude', dist.Normal(torch.tensor(0.0, device=use_device), torch.tensor(0.25, device=use_device)).expand([int(Tr_len)]).to_event(1))
    peak_day = pyro.sample('peak_day', dist.Normal(torch.tensor(196.0, device=use_device), torch.tensor(50.0, device=use_device)).expand([int(Tr_len)]).to_event(1))

    tr_idx = (treatment - 1).long()
    pl_idx = (plot_id - 1).long()
    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_temp = A[pl_idx] * torch.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0))
    sine_wave = amplitude[tr_idx] * torch.cos((2 * torch.pi / 365) * day_year + (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) - torch.pi / 2)
    model_resp = sine_wave + (xi_temp * xi_moist)

    pyro.sample('obs', dist.Normal(model_resp, sigma), obs=resp)
    return model_resp

# Ensure that the device is passed correctly when calling mcmc.run()
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=200, num_chains=3)

# Measure execution time for device="mps"
start_time_mps = time.time()
mcmc.run(Tr, Pl, treatment, plot_id, day_year, temp, resp, M, use_device=use_device)
end_time_mps = time.time()
execution_time_mps = end_time_mps - start_time_mps

print("Execution time for device='mps': {execution_time_mps} seconds")



def benchmark_matmul(tensor_size=10000):
    # Create a sample tensor
    temp_cpu = torch.rand(tensor_size, dtype=torch.float32, device='cpu')

    # Define the matmul operation
    def matmul_op(x):
        return torch.matmul(x.unsqueeze(1).T, x.unsqueeze(1))

    # --- CPU Benchmark ---
    start_cpu = time.time()
    result_cpu = matmul_op(temp_cpu)
    end_cpu = time.time()
    print(f"[CPU] Result shape cpu: {result_cpu.shape}, Time: {end_cpu - start_cpu:.6f} s")

    # Check MPS availability
    if torch.backends.mps.is_available():
        temp_mps = temp_cpu.to('mps')
        # Warm-up run for MPS
        _ = matmul_op(temp_mps)
        torch.mps.synchronize()  # Make sure everything finishes
        start_mps = time.time()
        result_mps = matmul_op(temp_mps)
        torch.mps.synchronize()  # Wait for MPS to finish
        end_mps = time.time()
        print(f"[MPS] Result shape mps: {result_mps.shape}, Time: {end_mps - start_mps:.6f} s")
    else:
        print("MPS not available on this system.")

# Run the benchmark
benchmark_matmul(tensor_size=10000)