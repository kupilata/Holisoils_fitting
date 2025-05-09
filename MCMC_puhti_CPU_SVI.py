#source /scratch/project_2010938/venv_mine/bin/activate
import time
import torch
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.infer import Predictive

# Load data
fitting_data = pd.read_csv("python_holisoils.csv")
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

# Precompute shared terms
T_0 = 227.13
temp_C = temp + 273.15  # Compute once (converted temperature to Kelvin)
day_of_year = 2 * torch.pi / 365  # Constant factor for sine wave

# Define the Pyro model with positive sigma
def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M):

    # Group indices
    tr_idx = (treatment - 1).long()
    pl_idx = (plot_id - 1).long()
    print(f"tr_idx: {tr_idx}, pl_idx: {pl_idx}")

    # Sample treatment-level parameters
    Ea = pyro.sample("Ea", dist.Normal(398.5, 20.0).expand([Tr]).to_event(1))
    A = pyro.sample("A", dist.Normal(400.0, 50.0).expand([Tr]).to_event(1))
    a = pyro.sample("a", dist.Normal(3.11, 1.0).expand([Tr]).to_event(1))
    b = pyro.sample("b", dist.Normal(2.42, 1.0).expand([Tr]).to_event(1))
    amplitude = pyro.sample("amplitude", dist.Normal(0.0, 1.0).expand([Tr]).to_event(1))
    peak_day = pyro.sample("peak_day", dist.Normal(196.0, 10.0).expand([Tr]).to_event(1))

    # Sample positive global noise
    sigma = pyro.sample('sigma', dist.HalfNormal(0.1))

    # Build model components
    temp_K = temp + 273.15
    xi_temp = torch.exp(-Ea[tr_idx] / 8.314 * (1.0 / temp_K - 1.0 / (T_0 + 273.15)))
    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_season = 1 + amplitude[tr_idx] * torch.sin(day_year * 2 * torch.pi / 365 - (peak_day[tr_idx] * 2 * torch.pi / 365))

    model_resp = A[tr_idx] * xi_temp * xi_moist * xi_season
    print(f"Model response: {model_resp}")

    # Expand sigma to match response shape
    sigma_expanded = sigma.expand(resp.shape)

    # Define plate for observations (batch dimension for resp)
    with pyro.plate("data", resp.shape[0]):
        pyro.sample("obs", dist.Normal(model_resp, sigma_expanded), obs=resp)


def guide(Tr, Pl, treatment, plot_id, day_year, temp, resp, M):
    # Guide distributions (these should match the ones in your model)
    
    # Ea
    Ea_loc = pyro.param("Ea_loc", torch.ones(Tr) * 398.5)
    Ea_scale = pyro.param("Ea_scale", torch.ones(Tr) * 20.0, constraint=torch.distributions.constraints.positive)
    pyro.sample("Ea", dist.Normal(Ea_loc, Ea_scale).expand([Tr]).to_event(1))
    
    # A
    A_loc = pyro.param("A_loc", torch.ones(Tr) * 400.0)
    A_scale = pyro.param("A_scale", torch.ones(Tr) * 50.0, constraint=torch.distributions.constraints.positive)
    pyro.sample("A", dist.Normal(A_loc, A_scale).expand([Tr]).to_event(1))
    
    # a
    a_loc = pyro.param("a_loc", torch.ones(Tr) * 3.11)
    a_scale = pyro.param("a_scale", torch.ones(Tr) * 1.0, constraint=torch.distributions.constraints.positive)
    pyro.sample("a", dist.Normal(a_loc, a_scale).expand([Tr]).to_event(1))
    
    # b
    b_loc = pyro.param("b_loc", torch.ones(Tr) * 2.42)
    b_scale = pyro.param("b_scale", torch.ones(Tr) * 1.0, constraint=torch.distributions.constraints.positive)
    pyro.sample("b", dist.Normal(b_loc, b_scale).expand([Tr]).to_event(1))
    
    # amplitude
    amplitude_loc = pyro.param("amplitude_loc", torch.zeros(Tr))
    amplitude_scale = pyro.param("amplitude_scale", torch.ones(Tr), constraint=torch.distributions.constraints.positive)
    pyro.sample("amplitude", dist.Normal(amplitude_loc, amplitude_scale).expand([Tr]).to_event(1))
    
    # peak_day
    peak_day_loc = pyro.param("peak_day_loc", torch.ones(Tr) * 196.0)
    peak_day_scale = pyro.param("peak_day_scale", torch.ones(Tr) * 10.0, constraint=torch.distributions.constraints.positive)
    pyro.sample("peak_day", dist.Normal(peak_day_loc, peak_day_scale).expand([Tr]).to_event(1))
    
    # sigma (positive global noise)
    sigma_scale = pyro.param("sigma_scale", torch.ones(1) * 0.1, constraint=torch.distributions.constraints.positive)
    pyro.sample('sigma', dist.HalfNormal(sigma_scale))



# Optimizer setup
optimizer = ClippedAdam({"lr": 0.01})

# Define the loss function (Trace_ELBO is commonly used)
svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

# Training loop
num_steps = 3000
for step in range(num_steps):
    loss = svi.step(Tr, Pl, treatment, plot_id, day_year, temp, resp, M)
    if step % 100 == 0:
        print(f"Step {step}:\tLoss = {loss:.2f}")



import matplotlib.pyplot as plt
import seaborn as sns
import torch


#Assuming Ea_loc and Ea_scale are already defined after training
# and Ea is a sample of 20 values drawn from the posterior
Ea_loc = pyro.param("Ea_loc")  # Mean of Ea
Ea_scale = pyro.param("Ea_scale")  # Standard deviation (uncertainty) of Ea

# Sample 20 values from the posterior distribution
samples_Ea = dist.Normal(Ea_loc, Ea_scale).expand([20]).to_event(1).sample()

# Convert the samples to a numpy array for plotting
samples_Ea_np = samples_Ea.detach().numpy()

# Assuming that Ea_scale gives the same uncertainty for all samples
# If Ea_scale varies per sample, we can modify this step
uncertainties_Ea = Ea_scale.detach().numpy()  # The uncertainties (std dev)

# Create a boxplot to show the 20 samples with their uncertainties
plt.figure(figsize=(10, 6))

# Scatter plot showing the samples
plt.scatter(range(1, 21), samples_Ea_np, color='blue', label='Samples')

# Error bars representing the uncertainty (standard deviation) for each sample
plt.errorbar(range(1, 21), samples_Ea_np, yerr=uncertainties_Ea, fmt='o', color='red', label='Uncertainty')

plt.title('Ea Samples with Associated Uncertainties')
plt.xlabel('Sample Number')
plt.ylabel('Ea Value')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Store the loss over time
losses = []

optimizer = ClippedAdam({"lr": 0.001})

# Define the loss function (Trace_ELBO is commonly used)
svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

# Training loop
num_steps = 10000
for step in range(num_steps):
    loss = svi.step(Tr, Pl, treatment, plot_id, day_year, temp, resp, M)
    losses.append(loss)
    if step % 100 == 0:
        print(f"Step {step}:\tLoss = {loss:.2f}")

# Convert losses to numpy array for easier manipulation
losses = np.array(losses)

# Define a threshold to clip outliers (e.g., 95th percentile or specific value)
threshold = np.percentile(losses, 90)  # For example, clip the top 5% losses

# Clip losses at the threshold
clipped_losses = np.clip(losses, None, threshold)

# Plot clipped losses
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (Clipped)')
plt.title('Loss Over Time During Training (Clipped)')
plt.show()