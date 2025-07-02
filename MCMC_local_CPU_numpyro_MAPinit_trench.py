
#source jax-env/bin/activate

import os
# Set XLA flags to use more than one CPUs. These should be virtual devices, I guess, so it might still be using all allocated resources
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=25"

# More aggressive CPU utilization
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=25 --xla_cpu_multi_thread_eigen=true"
os.environ["OMP_NUM_THREADS"] = "25"  
os.environ["MKL_NUM_THREADS"] = "25"
os.environ["NUMEXPR_NUM_THREADS"] = "25"


import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

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


# for MAP init
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro.optim as optim



# Function to find MAP using SVI
def find_MAP_svi(model, data_dict, num_steps=3000):
    """Find MAP estimate using Stochastic Variational Inference"""
    print("Finding MAP estimate using SVI...")
    
    # Use AutoNormal guide (mean-field approximation)
    guide = AutoNormal(model)
    
    # Set up SVI with Adam optimizer
    svi = SVI(model, guide, optim.Adam(0.01), Trace_ELBO())
    
    # Run optimization
    rng_key = PRNGKey(42)
    svi_result = svi.run(rng_key, num_steps, **data_dict)
    
    # Get MAP estimate (median of the guide distribution)
    map_estimate = guide.median(svi_result.params)
    
    print("MAP estimation completed!")
    return map_estimate

def init_around_map(map_estimate, num_chains=4, noise_scale=0.05):
    """Initialize chains around MAP with small noise"""
    print("Creating initialization around MAP...")
    
    rng_key = PRNGKey(123)
    init_params = {}
    
    for param_name, map_value in map_estimate.items():
        # Add small random noise to MAP estimate for each chain
        keys = jax.random.split(rng_key, num_chains)
        
        # Create noise with same shape as parameter
        noise = jax.vmap(
            lambda k: jax.random.normal(k, map_value.shape) * noise_scale
        )(keys)
        
        # Stack MAP estimate for each chain and add noise
        init_params[param_name] = map_value[None, ...] + noise
        rng_key = jax.random.split(rng_key, 1)[0]
    
    return init_params


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


#### Filter out Saint Mitre, temporary hopefully. There seems to be a unit conversion error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(fitting_data['site'].unique())
fitting_data = fitting_data[fitting_data['site'] != 'Saint Mitre']
print(fitting_data['site'].unique())



print("Convert data to jax.numpy arrays")
treatment = jnp.array(fitting_data['treatment'].values)
plot_id = jnp.array(fitting_data['plot_id'].values)
day_year = jnp.array(fitting_data['day_year'].values)
temp = jnp.array(fitting_data['temp'].values)
resp = jnp.array(fitting_data['resp'].values)
M = jnp.array(fitting_data['M_observed'].values)
treatment_name = np.array(fitting_data['treatment_name'])
trenched = jnp.array(fitting_data['trenched'].values, dtype=bool)

print("Printing treatment array extract: ", treatment[1:100])
print("Printing plot_id array extract: ", plot_id[1:100])
print("Printing day_year array extract: ", day_year[1:100])
print("Printing temp array extract: ", temp[1:100])
print("Printing resp array extract: ", resp[1:100])
print("Printing M array extract: ", M[1:100])

### this model has a linear boolean multiplier added, for trenched values
def model(Tr, Pl, treatment, trenched, plot_id, day_year, temp, resp, M):
    # Priors
    Ea = numpyro.sample("Ea", dist.Normal(398.5, 20.0).expand([Tr]))
    A = numpyro.sample("A", dist.Normal(400.0, 50.0).expand([Pl]))
    a = numpyro.sample("a", dist.Normal(3.11, 1.0).expand([Tr]))
    b = numpyro.sample("b", dist.Normal(2.42, 1.0).expand([Tr]))
    amplitude = numpyro.sample("amplitude", dist.Normal(0.0, 1.0).expand([Tr]))
    peak_day = numpyro.sample("peak_day", dist.Normal(196.0, 10.0).expand([Tr]))
    
    # New parameter: linear multiplier for non-trenched plots
    linear_mult = numpyro.sample("linear_mult", dist.Normal(1.0, 0.5))
    
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    # Constants
    T_0 = 227.13
    
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
    
    base_resp = A[pl_idx] * xi_temp * xi_moist * xi_season
    
    # Apply linear multiplier only when trenched = False
    model_resp = jnp.where(trenched, base_resp, base_resp * linear_mult)
    
    # Likelihood
    numpyro.sample("obs", dist.Normal(model_resp, sigma), obs=resp)


# Initialize NUTS kernel
nuts_kernel = NUTS(model, target_accept_prob=0.9)

# Initialize NUTS kernel, deeper
#nuts_kernel = NUTS(
#    model,
#    #step_size=0.01,        # Smaller step size
#    adapt_step_size=True,
#    adapt_mass_matrix=True,
#    max_tree_depth=8,      # Moderately deep (default is 10)
#    target_accept_prob=0.8 # Good balance (default is 0.8)
#)

print("Configure MCMC")
mcmc = MCMC(
    nuts_kernel,
    num_samples=15000,
    num_warmup=5000,
    num_chains=24,                  # Fully parallel
    chain_method="parallel",
    progress_bar=True
)

# Prepare data dictionary for SVI
data_dict = {
    'Tr': Tr,
    'Pl': Pl,
    'treatment': treatment,
    'trenched': trenched,  # Add this line
    'plot_id': plot_id,
    'day_year': day_year,
    'temp': temp,
    'resp': resp,
    'M': M
}

# Find MAP estimate
map_estimate = find_MAP_svi(model, data_dict, num_steps=3000)

# Create initialization around MAP - FIXED VERSION
init_params = init_around_map(map_estimate, num_chains=24, noise_scale=0.05)  # Changed from 4 to 24

print("# Run MCMC with MAP initialization")
rng_key = PRNGKey(0)
mcmc.run(rng_key, init_params=init_params, **data_dict)


#Check convergence diagnostics
print("\n=== Convergence Diagnostics ===")
print(f"Divergences: {mcmc.get_extra_fields()['diverging'].sum()}")


# Extract posterior
posterior_samples = mcmc.get_samples()

# Save the posterior samples as a torch file
torch.save(posterior_samples, 'posterior_samples.pt')
print("Results saved to posterior_samples.pt")


