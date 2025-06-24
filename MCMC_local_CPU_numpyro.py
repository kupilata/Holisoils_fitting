
#source jax-env/bin/activate

import os
# Set XLA flags to use more than one CPUs. These should be virtual devices, I guess, so it might still be using all allocated resources
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=25"

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd
import torch #just to save the torch object

########## Block to start monitoring CPU, writes in cpu_usage.log #########################
import psutil
import time
import threading

# Function to monitor and log CPU usage periodically
def monitor_cpu_usage(interval=30, log_file="cpu_usage.log"):
    with open(log_file, "w") as f:
        f.write(f"CPU monitoring started. SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}\n")
        f.write(f"Total CPUs available according to psutil: {psutil.cpu_count()}\n")
        f.write(f"JAX devices configured: {os.environ.get('XLA_FLAGS', 'Not set')}\n\n")
        
        f.write("Time, CPU Usage (%), Number of Threads\n")
        
        while True:
            cpu_percent = psutil.cpu_percent(interval=interval)
            process = psutil.Process(os.getpid())
            num_threads = len(process.threads())
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            log_line = f"{timestamp}, {cpu_percent:.1f}%, {num_threads}\n"
            f.write(log_line)
            f.flush()  # Make sure data is written immediately
            
            time.sleep(interval)


# Start monitoring in a background thread
print("Starting CPU monitoring thread...")
monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
monitor_thread.start()

########## End of bloc to start monitoring CPU #########################


#chech the CPU count detected
print("Available devices:", jax.local_device_count())

devices = jax.local_devices()
print("Detected devices:", devices)

numpyro.set_host_device_count(25)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Match this to the XLA FLAGS line

# Load the data
print("Loading data")
fitting_data = pd.read_csv("python_holisoils.csv")
Tr = int(fitting_data['treatment'].max())
Pl = int(fitting_data['plot_id'].max())
print("Treatments:", Tr, "Plots:", Pl)


print("Convert data to jax.numpy arrays")
treatment = jnp.array(fitting_data['treatment'].values)
plot_id = jnp.array(fitting_data['plot_id'].values)
day_year = jnp.array(fitting_data['day_year'].values)
temp = jnp.array(fitting_data['temp'].values)
resp = jnp.array(fitting_data['resp'].values)
M = jnp.array(fitting_data['M_observed'].values)
treatment_name = np.array(fitting_data['treatment_name'])

print("Printing treatment array extract: ", treatment[1:100])
print("Printing plot_id array extract: ", plot_id[1:100])
print("Printing day_year array extract: ", day_year[1:100])
print("Printing temp array extract: ", temp[1:100])
print("Printing resp array extract: ", resp[1:100])
print("Printing M array extract: ", M[1:100])

# Define the model
def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M):
    # Priors
    Ea = numpyro.sample("Ea", dist.Normal(398.5, 20.0).expand([Tr]))
    A = numpyro.sample("A", dist.Normal(400.0, 50.0).expand([Pl]))
    a = numpyro.sample("a", dist.Normal(3.11, 1.0).expand([Tr]))
    b = numpyro.sample("b", dist.Normal(2.42, 1.0).expand([Tr]))
    amplitude = numpyro.sample("amplitude", dist.Normal(0.0, 1.0).expand([Tr]))
    peak_day = numpyro.sample("peak_day", dist.Normal(196.0, 10.0).expand([Tr]))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # Constants
    T_0 = 227.13  # constant like in your Pyro model

    # Derived indices
    tr_idx = treatment.astype(int) - 1
    pl_idx = plot_id.astype(int) - 1

    # Process model
    xi_temp = jnp.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0))
    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_season = amplitude[tr_idx] * jnp.cos(
        (2 * jnp.pi / 365) * day_year +
        (2 * jnp.pi / 365) * (peak_day[tr_idx] - 1) -
        jnp.pi / 2
    )

    model_resp = A[pl_idx] * xi_temp * xi_moist * xi_season

    # Likelihood
    numpyro.sample("obs", dist.Normal(model_resp, sigma), obs=resp)

# Initialize NUTS kernel
nuts_kernel = NUTS(model, target_accept_prob=0.9)

print("Configure MCMC")
mcmc = MCMC(
    nuts_kernel,
    num_samples=15000,
    num_warmup=2000,
    num_chains=4,                  # Fully parallel
    chain_method="parallel",
    progress_bar=True
)

print("# Run MCMC")
rng_key = PRNGKey(0) #initialize the random number generators
mcmc.run(rng_key, Tr, Pl, treatment, plot_id, day_year, temp, resp, M)

# Extract posterior
posterior_samples = mcmc.get_samples()

# Save the posterior samples as a torch file
torch.save(posterior_samples, 'posterior_samples.pt')
print("Results saved to posterior_samples.pt")



# find the categories
# Create mapping from treatment to treatment_name
treatment_map = dict(zip(fitting_data['treatment'], fitting_data['treatment_name']))

# Get unique treatments and their corresponding names
unique_treatments = np.unique(fitting_data['treatment'])
unique_treatments_names = [treatment_map[t] for t in unique_treatments]