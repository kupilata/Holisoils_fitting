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


# Function to check for invalid values in arrays
def check_array_validity(arr, name):
    """Check if array contains NaN or infinite values"""
    if jnp.any(jnp.isnan(arr)):
        print(f"WARNING: {name} contains NaN values")
        return False
    if jnp.any(jnp.isinf(arr)):
        print(f"WARNING: {name} contains infinite values")
        return False
    print(f"{name}: OK (min={jnp.min(arr):.3f}, max={jnp.max(arr):.3f})")
    return True


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

# Check data validity
print("\n=== Data Validation ===")
check_array_validity(treatment, "treatment")
check_array_validity(plot_id, "plot_id")
check_array_validity(day_year, "day_year")
check_array_validity(temp, "temp")
check_array_validity(resp, "resp")
check_array_validity(M, "M")
print(f"trenched: {jnp.sum(trenched)} True values out of {len(trenched)} total")

# Additional checks
print(f"\nData ranges:")
print(f"Temperature range: {jnp.min(temp):.2f} to {jnp.max(temp):.2f}")
print(f"Moisture range: {jnp.min(M):.2f} to {jnp.max(M):.2f}")
print(f"Response range: {jnp.min(resp):.2f} to {jnp.max(resp):.2f}")

# Check for negative or zero values that might cause issues
if jnp.any(M <= 0):
    print("WARNING: M contains non-positive values which might cause issues in M^2 term")
if jnp.any(resp <= 0):
    print("WARNING: resp contains non-positive values")

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
    
    # Process model with more robust calculations
    # Temperature component
    temp_kelvin = temp + 273.15
    temp_diff = temp_kelvin - T_0
    
    # Add small epsilon to avoid division by zero or very small denominators
    epsilon = 1e-6
    temp_diff = jnp.where(jnp.abs(temp_diff) < epsilon, 
                         jnp.sign(temp_diff) * epsilon, 
                         temp_diff)
    
    xi_temp = jnp.exp(-Ea[tr_idx] / temp_diff)
    
    # Moisture component - ensure M is positive and add bounds
    M_safe = jnp.clip(M, 0.01, 1.0)  # Clip to reasonable range
    xi_moist = a[tr_idx] * M_safe - b[tr_idx] * M_safe**2
    
    # Seasonal component
    xi_season = amplitude[tr_idx] * jnp.cos(
        (2 * jnp.pi / 365) * day_year +
        (2 * jnp.pi / 365) * (peak_day[tr_idx] - 1) -
        jnp.pi / 2
    )
    
    # Base respiration
    base_resp = A[pl_idx] * xi_temp * xi_moist * xi_season
    
    # Apply linear multiplier only when trenched = False
    # Ensure linear_mult is positive to avoid negative respiration
    linear_mult_safe = jnp.exp(linear_mult)  # Use exp to ensure positivity
    model_resp = jnp.where(trenched, base_resp, base_resp * linear_mult_safe)
    
    # Add a small positive constant to avoid issues with very small values
    model_resp = jnp.maximum(model_resp, 1e-6)
    
    # Check for invalid values before likelihood
    numpyro.deterministic("model_resp_check", model_resp)
    
    # Likelihood with more robust sigma
    sigma_safe = jnp.maximum(sigma, 1e-3)  # Ensure sigma is not too small
    numpyro.sample("obs", dist.Normal(model_resp, sigma_safe), obs=resp)


# Test the model with some sample parameters first
print("\n=== Testing model with sample parameters ===")
def test_model():
    # Create some reasonable parameter values for testing
    test_params = {
        "Ea": jnp.full(Tr, 398.5),
        "A": jnp.full(Pl, 400.0),
        "a": jnp.full(Tr, 3.11),
        "b": jnp.full(Tr, 2.42),
        "amplitude": jnp.full(Tr, 0.0),
        "peak_day": jnp.full(Tr, 196.0),
        "linear_mult": 0.0,  # exp(0) = 1
        "sigma": 1.0
    }
    
    # Test model calculation
    try:
        # Simulate the model calculations
        T_0 = 227.13
        tr_idx = treatment.astype(int) - 1
        pl_idx = plot_id.astype(int) - 1
        
        temp_kelvin = temp + 273.15
        temp_diff = temp_kelvin - T_0
        epsilon = 1e-6
        temp_diff = jnp.where(jnp.abs(temp_diff) < epsilon, 
                             jnp.sign(temp_diff) * epsilon, 
                             temp_diff)
        
        xi_temp = jnp.exp(-test_params["Ea"][tr_idx] / temp_diff)
        
        M_safe = jnp.clip(M, 0.01, 1.0)
        xi_moist = test_params["a"][tr_idx] * M_safe - test_params["b"][tr_idx] * M_safe**2
        
        xi_season = test_params["amplitude"][tr_idx] * jnp.cos(
            (2 * jnp.pi / 365) * day_year +
            (2 * jnp.pi / 365) * (test_params["peak_day"][tr_idx] - 1) -
            jnp.pi / 2
        )
        
        base_resp = test_params["A"][pl_idx] * xi_temp * xi_moist * xi_season
        linear_mult_safe = jnp.exp(test_params["linear_mult"])
        model_resp = jnp.where(trenched, base_resp, base_resp * linear_mult_safe)
        model_resp = jnp.maximum(model_resp, 1e-6)
        
        print("Model calculation test passed!")
        check_array_validity(xi_temp, "xi_temp")
        check_array_validity(xi_moist, "xi_moist")
        check_array_validity(xi_season, "xi_season")
        check_array_validity(base_resp, "base_resp")
        check_array_validity(model_resp, "model_resp")
        
        return True
        
    except Exception as e:
        print(f"Model calculation test failed: {e}")
        return False

if not test_model():
    print("Model test failed! Stopping execution.")
    exit(1)

# Initialize NUTS kernel
nuts_kernel = NUTS(model, target_accept_prob=0.9)

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
    'trenched': trenched,
    'plot_id': plot_id,
    'day_year': day_year,
    'temp': temp,
    'resp': resp,
    'M': M
}

print("\n=== Starting MAP estimation ===")
try:
    # Find MAP estimate
    map_estimate = find_MAP_svi(model, data_dict, num_steps=3000)
    print("MAP estimation successful!")
    
    # Create initialization around MAP
    init_params = init_around_map(map_estimate, num_chains=24, noise_scale=0.05)
    
    print("# Run MCMC with MAP initialization")
    rng_key = PRNGKey(0)
    mcmc.run(rng_key, init_params=init_params, **data_dict)
    
except Exception as e:
    print(f"MAP estimation failed: {e}")
    print("Falling back to default initialization...")
    
    # Run without MAP initialization
    rng_key = PRNGKey(0)
    mcmc.run(rng_key, **data_dict)

#Check convergence diagnostics
print("\n=== Convergence Diagnostics ===")
print(f"Divergences: {mcmc.get_extra_fields()['diverging'].sum()}")

# Extract posterior
posterior_samples = mcmc.get_samples()

# Save the posterior samples as a torch file
torch.save(posterior_samples, 'posterior_samples.pt')
print("Results saved to posterior_samples.pt")