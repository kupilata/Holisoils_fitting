#source /scratch/project_2010938/venv_mine/bin/activate

import time
import torch
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
# import arviz as az #for diagnostics on MCMC
import matplotlib.pyplot as plt
import seaborn as sns

torch.set_num_threads(10)  # or maybe 4, to try if CPUs are 20. 5 works well, it prevents the sampling to slow down over time for some reason

# Load data
fitting_data = pd.read_csv("python_holisoils.csv")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#fitting_data = fitting_data.sample(frac=0.25, random_state=42) #resampling 25% of the dataset. !!!!!!!!!!!!!! for testing, speeding up the runs, please remove it!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Tr = int(fitting_data['treatment'].max())
Pl = int(fitting_data['plot_id'].max())

# Extract data as CPU tensors
def to_tensor(col):
    return torch.tensor(fitting_data[col].values, dtype=torch.float32)

treatment = to_tensor('treatment')
plot_id = to_tensor('plot_id')
day_year = to_tensor('day_year')
temp = to_tensor('temp')
resp = to_tensor('resp')
M = to_tensor('M_observed')

# Convert tensors to contiguous format, suggested by Claude
treatment = to_tensor('treatment').contiguous()
plot_id = to_tensor('plot_id').contiguous()
day_year = to_tensor('day_year').contiguous()
temp = to_tensor('temp').contiguous()
resp = to_tensor('resp').contiguous()
M = to_tensor('M_observed').contiguous()


# Create a DataFrame for easier plotting
#resp_np = fitting_data['resp']
#df = pd.DataFrame({
#    'Response': resp_np,
#    'Treatment': treatment
#})

# Boplot of resp
#plt.figure(figsize=(10, 6))
#sns.boxplot(x='Treatment', y='Response', data=df)
#plt.title('Response by Treatment Group')
#plt.xlabel('Treatment')
#plt.ylabel('Response')
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.tight_layout()
#plt.show()


####################### MODEL

# Precompute shared terms
T_0 = 227.13 #value in K, internal to the model there is the conversion to C
#temp_K = temp + 273.15  # Compute once (converted temperature to Kelvin), not used since the function takes C
day_of_year = 2 * torch.pi / 365  # Constant factor for sine wave

# Pre-calculate indices
tr_idx = (treatment - 1).long()
pl_idx = (plot_id - 1).long()
pl_idx.shape
tr_idx.shape
max(pl_idx)

# Define initial values for the parameters for sampling the chains
initial_params = {
    "Ea": torch.tensor(398.5).expand([Tr]),
    "A": torch.tensor(400.0).expand([Pl]),
    "a": torch.tensor(3.11).expand([Tr]),
    "b": torch.tensor(2.42).expand([Tr]),
    "amplitude": torch.zeros(Tr),
    "peak_day": torch.tensor(196.0).expand([Tr]),
    "sigma": torch.tensor(0.1)
}

# Define the Pyro model with positive sigma
def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M):
    #print(f"Inside model: Pl = {Pl}")  # Verify Pl is 613
    
    # Sample treatment-level parameters
    Ea = pyro.sample("Ea", dist.Normal(398.5, 20.0).expand([Tr]).to_event(1))
    A = pyro.sample("A", dist.Normal(400.0, 50.0).expand([Pl]).to_event(1))
    a = pyro.sample("a", dist.Normal(3.11, 1.0).expand([Tr]).to_event(1))
    b = pyro.sample("b", dist.Normal(2.42, 1.0).expand([Tr]).to_event(1))
    amplitude = pyro.sample("amplitude", dist.Normal(0.0, 1.0).expand([Tr]).to_event(1))
    peak_day = pyro.sample("peak_day", dist.Normal(196.0, 10.0).expand([Tr]).to_event(1))


    # Build model components
    xi_temp = torch.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0)) #this takes temp in C, and converts it
    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_season = amplitude[tr_idx] * torch.cos(
        (2 * torch.pi / 365) * day_year +
        (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) -
        torch.pi / 2
    )
    model_resp = A[pl_idx] * xi_temp * xi_moist * xi_season

    # Observation likelihood
    sigma = pyro.sample('sigma', dist.HalfNormal(0.1)) 
    pyro.sample("obs", dist.Normal(model_resp, sigma), obs=resp)

# Run MCMC with multiple CPU chains (auto-parallelized)
nuts_kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=9, init_strategy=pyro.infer.autoguide.init_to_value(values=initial_params))
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=500, num_chains=4) #maybe 5 better as prime number but reserve also 25-26 CPUs

start_time = time.time()
mcmc.run(Tr, Pl, treatment, plot_id, day_year, temp, resp, M)
end_time = time.time()


print(f"\nAuto-parallel MCMC on CPU finished in {end_time - start_time:.2f} seconds.")

# Get posterior samples
posterior = mcmc.get_samples()

# Save the posterior samples as a torch file
torch.save(posterior, 'posterior_samples.pt')

print("Results saved to posterior_samples.pt")

#posterior_loaded = torch.load('posterior_samples.pt')
#print(posterior.keys())
#print(posterior_loaded['A'].shape)