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
import pandas as pd
import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import torch #just to save the torch object

########## Block to start monitoring CPU, writes in cpu_usage.log #########################
import psutil
import time
import threading

# for MAP init
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro.optim as optim
from numpyro.infer import Predictive

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

# Create site mapping
print("\n=== Creating Site Mapping ===")
sites = fitting_data['site'].unique()
n_sites = len(sites)
site_to_idx = {site: idx for idx, site in enumerate(sites)}
fitting_data['site_idx'] = fitting_data['site'].map(site_to_idx)

print(f"Found {n_sites} unique sites:")
for i, site in enumerate(sites):
    site_data = fitting_data[fitting_data['site'] == site]
    n_obs = len(site_data)
    n_missing_M = site_data['M_observed'].isna().sum()
    n_missing_temp = site_data['temp'].isna().sum()
    print(f"  {i}: {site} - {n_obs} obs, {n_missing_M} missing M, {n_missing_temp} missing temp")

# Create site x treatment combinations and filter sparse ones
print("\n=== Creating Site × Treatment Combinations ===")
site_treatment_combinations = fitting_data.groupby(['site_idx', 'treatment']).size().reset_index(name='count')
print(f"Total possible site×treatment combinations: {n_sites * Tr}")
print(f"Observed site×treatment combinations: {len(site_treatment_combinations)}")

# Filter combinations with sufficient data (adjust threshold as needed)
min_obs_threshold = 5  # Minimum observations per site×treatment combination
valid_combinations = site_treatment_combinations[site_treatment_combinations['count'] >= min_obs_threshold]
print(f"Valid combinations (≥{min_obs_threshold} obs): {len(valid_combinations)}")

# Create mapping for valid site×treatment combinations
valid_combinations = valid_combinations.reset_index(drop=True)
valid_combinations['site_treat_idx'] = valid_combinations.index
n_site_treat_combinations = len(valid_combinations)

print("Valid site×treatment combinations:")
for idx, row in valid_combinations.iterrows():
    site_name = sites[row['site_idx']]
    print(f"  {idx}: Site {row['site_idx']} ({site_name}) × Treatment {row['treatment']} - {row['count']} obs")

# Create mapping dictionary for fast lookup
site_treat_to_idx = {}
for idx, row in valid_combinations.iterrows():
    key = (row['site_idx'], row['treatment'])
    site_treat_to_idx[key] = idx

# Add site×treatment index to the main dataframe
def get_site_treat_idx(row):
    key = (row['site_idx'], row['treatment'])
    return site_treat_to_idx.get(key, -1)  # -1 for invalid combinations

fitting_data['site_treat_idx'] = fitting_data.apply(get_site_treat_idx, axis=1)

# Remove rows with invalid site×treatment combinations
print(f"Removing {(fitting_data['site_treat_idx'] == -1).sum()} observations with sparse site×treatment combinations")
fitting_data = fitting_data[fitting_data['site_treat_idx'] != -1].reset_index(drop=True)

print(f"Final dataset size: {len(fitting_data)} observations")
print(f"Using {n_site_treat_combinations} site×treatment combinations")

# Handle missing data with site-level approach
print("\n=== Site-Level Missing Data Analysis ===")

# Check for missing M values
M_missing_mask = fitting_data['M_observed'].isna()
n_missing_M_total = M_missing_mask.sum()
n_total = len(fitting_data)
print(f"Total missing M values: {n_missing_M_total} out of {n_total} ({n_missing_M_total/n_total*100:.1f}%)")

# Check for missing temperature values
temp_missing_mask = fitting_data['temp'].isna()
n_missing_temp_total = temp_missing_mask.sum()
print(f"Total missing temp values: {n_missing_temp_total} out of {n_total} ({n_missing_temp_total/n_total*100:.1f}%)")

# Calculate site-level missing data statistics
print("\nSite-level missing data patterns:")
for site in sites:
    site_mask = fitting_data['site'] == site
    site_M_missing = (M_missing_mask & site_mask).sum()
    site_temp_missing = (temp_missing_mask & site_mask).sum()
    site_total = site_mask.sum()
    if site_total > 0:  # Only print if site has data after filtering
        print(f"  {site}: {site_M_missing}/{site_total} missing M ({site_M_missing/site_total*100:.1f}%), "
              f"{site_temp_missing}/{site_total} missing temp ({site_temp_missing/site_total*100:.1f}%)")

# For missing values, use placeholder values (will be estimated by the model)
M_observed_with_placeholders = fitting_data['M_observed'].fillna(0.5)  # Neutral placeholder
temp_with_placeholders = fitting_data['temp'].fillna(fitting_data['temp'].median())  # Use median as placeholder

print("Convert data to jax.numpy arrays")
treatment = jnp.array(fitting_data['treatment'].values)
plot_id = jnp.array(fitting_data['plot_id'].values)
day_year = jnp.array(fitting_data['day_year'].values)
temp_with_placeholders_jax = jnp.array(temp_with_placeholders.values)
resp = jnp.array(fitting_data['resp'].values)
M_with_placeholders = jnp.array(M_observed_with_placeholders.values)
M_missing_mask_jax = jnp.array(M_missing_mask.values)
temp_missing_mask_jax = jnp.array(temp_missing_mask.values)
site_idx = jnp.array(fitting_data['site_idx'].values)
site_treat_idx = jnp.array(fitting_data['site_treat_idx'].values)
treatment_name = np.array(fitting_data['treatment_name'])
trenched = jnp.array(fitting_data['trenched'].values, dtype=bool)

print("Printing array extracts:")
print("treatment array extract: ", treatment[1:100])
print("site_idx array extract: ", site_idx[1:100])
print("site_treat_idx array extract: ", site_treat_idx[1:100])
print("temp array extract: ", temp_with_placeholders_jax[1:100])
print("M array extract: ", M_with_placeholders[1:100])

# Check data validity
print("\n=== Data Validation ===")
check_array_validity(treatment, "treatment")
check_array_validity(plot_id, "plot_id")
check_array_validity(day_year, "day_year")
check_array_validity(temp_with_placeholders_jax, "temp_with_placeholders")
check_array_validity(resp, "resp")
check_array_validity(M_with_placeholders, "M_with_placeholders")
check_array_validity(site_idx, "site_idx")
check_array_validity(site_treat_idx, "site_treat_idx")
print(f"trenched: {jnp.sum(trenched)} True values out of {len(trenched)} total")
print(f"Missing M values: {jnp.sum(M_missing_mask_jax)} out of {len(M_missing_mask_jax)} total")
print(f"Missing temp values: {jnp.sum(temp_missing_mask_jax)} out of {len(temp_missing_mask_jax)} total")
print(f"Sites: {n_sites} unique sites")
print(f"Site×Treatment combinations: {n_site_treat_combinations}")

# Memory estimate for new approach
print(f"\n=== Memory Estimate ===")
old_memory_gb = 24 * (n_missing_M_total + n_missing_temp_total) * 15000 * 4 / (1024**3)
new_memory_gb = 24 * (n_sites * 4 + n_site_treat_combinations * 6) * 15000 * 4 / (1024**3)  # Site params + site×treat params
print(f"Old approach memory: {old_memory_gb:.2f} GB")
print(f"New site×treatment approach memory: {new_memory_gb:.2f} GB")
print(f"Memory reduction: {old_memory_gb/new_memory_gb:.1f}x")

# Create mapping arrays for the model
site_treat_to_site = jnp.array([valid_combinations.iloc[i]['site_idx'] for i in range(n_site_treat_combinations)])
site_treat_to_treatment = jnp.array([valid_combinations.iloc[i]['treatment'] for i in range(n_site_treat_combinations)])

### Site×Treatment Hierarchical Bayesian Model
def model(Tr, Pl, n_sites, n_site_treat_combinations, treatment, trenched, plot_id, day_year, 
          temp_with_placeholders, resp, M_with_placeholders, M_missing_mask, temp_missing_mask, 
          site_idx, site_treat_idx, site_treat_to_site, site_treat_to_treatment):
    
    # HIERARCHICAL STRUCTURE: Treatment -> Site×Treatment -> Plots
    
    # Treatment-level hyperpriors (global parameters)
    Ea_mu = numpyro.sample("Ea_mu", dist.Normal(398.5, 20.0))
    Ea_sigma = numpyro.sample("Ea_sigma", dist.Exponential(1.0))
    
    a_mu = numpyro.sample("a_mu", dist.Normal(3.11, 1.0))
    a_sigma = numpyro.sample("a_sigma", dist.Exponential(1.0))
    
    b_mu = numpyro.sample("b_mu", dist.Normal(2.42, 1.0))
    b_sigma = numpyro.sample("b_sigma", dist.Exponential(1.0))
    
    amplitude_mu = numpyro.sample("amplitude_mu", dist.Normal(0.0, 1.0))
    amplitude_sigma = numpyro.sample("amplitude_sigma", dist.Exponential(1.0))
    
    peak_day_mu = numpyro.sample("peak_day_mu", dist.Normal(196.0, 10.0))
    peak_day_sigma = numpyro.sample("peak_day_sigma", dist.Exponential(1.0))
    
    # Site×Treatment level parameters (hierarchical)
    Ea_site_treat = numpyro.sample("Ea_site_treat", 
                                   dist.Normal(Ea_mu, Ea_sigma).expand([n_site_treat_combinations]))
    a_site_treat = numpyro.sample("a_site_treat", 
                                  dist.Normal(a_mu, a_sigma).expand([n_site_treat_combinations]))
    b_site_treat = numpyro.sample("b_site_treat", 
                                  dist.Normal(b_mu, b_sigma).expand([n_site_treat_combinations]))
    amplitude_site_treat = numpyro.sample("amplitude_site_treat", 
                                          dist.Normal(amplitude_mu, amplitude_sigma).expand([n_site_treat_combinations]))
    peak_day_site_treat = numpyro.sample("peak_day_site_treat", 
                                         dist.Normal(peak_day_mu, peak_day_sigma).expand([n_site_treat_combinations]))
    
    # Plot-level parameters (as before)
    A = numpyro.sample("A", dist.Normal(400.0, 50.0).expand([Pl]))
    
    # Linear multiplier for non-trenched plots
    linear_mult = numpyro.sample("linear_mult", dist.Normal(0.0, 0.5))  # log-scale
    
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    # SITE-LEVEL MISSING DATA MODEL FOR MOISTURE
    # Sample site-level moisture parameters
    site_M_mean = numpyro.sample("site_M_mean", dist.Beta(2.0, 2.0).expand([n_sites]))
    site_M_std = numpyro.sample("site_M_std", dist.Exponential(2.0).expand([n_sites]))
    
    # Scale site means to reasonable moisture range
    site_M_mean_scaled = 0.1 + 0.8 * site_M_mean  # Scale to [0.1, 0.9]
    site_M_std_safe = jnp.maximum(site_M_std * 0.1, 0.01)  # Reasonable std, minimum 0.01
    
    # Create complete M array using site-level distributions
    M_complete = jnp.where(
        M_missing_mask,
        # For missing values: sample from site-specific distribution
        site_M_mean_scaled[site_idx],  # Use site mean as imputed value
        # For observed values: use actual observations
        M_with_placeholders
    )
    
    # SITE-LEVEL MISSING DATA MODEL FOR TEMPERATURE  
    # Sample site-level temperature parameters
    site_temp_mean = numpyro.sample("site_temp_mean", dist.Normal(15.0, 10.0).expand([n_sites]))
    site_temp_std = numpyro.sample("site_temp_std", dist.Exponential(1.0).expand([n_sites]))
    
    # Ensure reasonable temperature std
    site_temp_std_safe = jnp.maximum(site_temp_std, 1.0)  # Minimum 1°C std
    
    # Create complete temperature array using site-level distributions
    temp_complete = jnp.where(
        temp_missing_mask,
        # For missing values: use site-specific mean
        site_temp_mean[site_idx],
        # For observed values: use actual observations
        temp_with_placeholders
    )
    
    # Constants
    T_0 = 227.13
    
    # Derived indices
    pl_idx = plot_id.astype(int) - 1
    
    # Process model using SITE×TREATMENT specific parameters
    # Temperature component
    temp_kelvin = temp_complete + 273.15
    temp_diff = temp_kelvin - T_0
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    temp_diff = jnp.where(jnp.abs(temp_diff) < epsilon, 
                         jnp.sign(temp_diff) * epsilon, 
                         temp_diff)
    
    xi_temp = jnp.exp(-Ea_site_treat[site_treat_idx] / temp_diff)
    
    # Moisture component - ensure M is positive and add bounds
    M_safe = jnp.clip(M_complete, 0.01, 1.0)
    xi_moist = a_site_treat[site_treat_idx] * M_safe - b_site_treat[site_treat_idx] * M_safe**2
    
    # Seasonal component
    xi_season = amplitude_site_treat[site_treat_idx] * jnp.cos(
        (2 * jnp.pi / 365) * day_year +
        (2 * jnp.pi / 365) * (peak_day_site_treat[site_treat_idx] - 1) -
        jnp.pi / 2
    )
    
    # Base respiration
    base_resp = A[pl_idx] * xi_temp * xi_moist * xi_season
    
    # Apply linear multiplier only when trenched = False
    linear_mult_safe = jnp.exp(linear_mult)  # Ensure positivity
    model_resp = jnp.where(trenched, base_resp, base_resp * linear_mult_safe)
    
    # Ensure positive respiration
    model_resp = jnp.maximum(model_resp, 1e-6)
    
    # Store for diagnostics
    numpyro.deterministic("M_complete", M_complete)
    numpyro.deterministic("temp_complete", temp_complete)
    numpyro.deterministic("site_M_mean_scaled", site_M_mean_scaled)
    numpyro.deterministic("site_temp_mean_est", site_temp_mean)
    numpyro.deterministic("xi_temp", xi_temp)
    numpyro.deterministic("xi_moist", xi_moist)
    numpyro.deterministic("xi_season", xi_season)
    numpyro.deterministic("model_resp", model_resp)
    
    # Store aggregated treatment-level parameters for post-processing
    numpyro.deterministic("Ea_by_treatment", Ea_site_treat[site_treat_to_treatment - 1])
    numpyro.deterministic("a_by_treatment", a_site_treat[site_treat_to_treatment - 1])
    numpyro.deterministic("b_by_treatment", b_site_treat[site_treat_to_treatment - 1])
    
    # Likelihood
    sigma_safe = jnp.maximum(sigma, 1e-3)
    numpyro.sample("obs", dist.Normal(model_resp, sigma_safe), obs=resp)


# Test the model with some sample parameters first
print("\n=== Testing model with sample parameters ===")
def test_model():
    try:
        # Test with a small subset of data first
        test_size = min(100, len(treatment))
        test_data_dict = {
            'Tr': Tr,
            'Pl': Pl,
            'n_sites': n_sites,
            'n_site_treat_combinations': n_site_treat_combinations,
            'treatment': treatment[:test_size],
            'trenched': trenched[:test_size],
            'plot_id': plot_id[:test_size],
            'day_year': day_year[:test_size],
            'temp_with_placeholders': temp_with_placeholders_jax[:test_size],
            'resp': resp[:test_size],
            'M_with_placeholders': M_with_placeholders[:test_size],
            'M_missing_mask': M_missing_mask_jax[:test_size],
            'temp_missing_mask': temp_missing_mask_jax[:test_size],
            'site_idx': site_idx[:test_size],
            'site_treat_idx': site_treat_idx[:test_size],
            'site_treat_to_site': site_treat_to_site,
            'site_treat_to_treatment': site_treat_to_treatment
        }
        
        # Initialize model to check for issues
        rng_key = PRNGKey(42)
        
        # Try to trace the model
        from numpyro.handlers import trace, seed
        model_trace = trace(seed(model, rng_key)).get_trace(**test_data_dict)
        
        print("Model test passed!")
        print(f"Test performed on {test_size} data points")
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if not test_model():
    print("Model test failed! Please check the error above.")
    exit(1)

# Initialize NUTS kernel
nuts_kernel = NUTS(model, target_accept_prob=0.9)

print("Configure MCMC")
mcmc = MCMC(
    nuts_kernel,
    num_samples=15000,
    num_warmup=5000,
    num_chains=24,
    chain_method="parallel",
    progress_bar=True,
    thinning=2  # Store every 2nd sample
)

# Prepare data dictionary for SVI and MCMC
data_dict = {
    'Tr': Tr,
    'Pl': Pl,
    'n_sites': n_sites,
    'n_site_treat_combinations': n_site_treat_combinations,
    'treatment': treatment,
    'trenched': trenched,
    'plot_id': plot_id,
    'day_year': day_year,
    'temp_with_placeholders': temp_with_placeholders_jax,
    'resp': resp,
    'M_with_placeholders': M_with_placeholders,
    'M_missing_mask': M_missing_mask_jax,
    'temp_missing_mask': temp_missing_mask_jax,
    'site_idx': site_idx,
    'site_treat_idx': site_treat_idx,
    'site_treat_to_site': site_treat_to_site,
    'site_treat_to_treatment': site_treat_to_treatment
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

# Check convergence diagnostics
print("\n=== Convergence Diagnostics ===")
divergences = mcmc.get_extra_fields()['diverging'].sum()
print(f"Divergences: {divergences}")

if divergences > 0:
    print(f"Warning: {divergences} divergent transitions detected!")
    print("Consider increasing target_accept_prob or reducing step size")

# Extract posterior and predictions
posterior_samples = mcmc.get_samples()

# Clear MCMC object to free memory before post-processing
del mcmc
import gc; gc.collect()

# Save the posterior samples as a torch file (redundant, just in case)
torch.save(posterior_samples, 'posterior_samples.pt')
print("Posteriors saved to posterior_samples.pt")

# Get predictions from the posterior
print("Generating model predictions...")
thinned_posterior = {}
thin_factor = 10  # Use every 10th sample for predictions
for key, value in posterior_samples.items():
    thinned_posterior[key] = value[::thin_factor]

predictive = Predictive(model, thinned_posterior)

data_dict_pred = data_dict.copy()
data_dict_pred['resp'] = None  # Don't condition on observations
predictions_dict = predictive(rng_key, **data_dict_pred)

# Save everything together
complete_results = {
    **posterior_samples,  # All parameters
    'model_resp': predictions_dict['model_resp'],
    'M_complete': predictions_dict['M_complete'],
    'temp_complete': predictions_dict['temp_complete'],
    'site_M_mean_scaled': predictions_dict['site_M_mean_scaled'],
    'site_temp_mean_est': predictions_dict['site_temp_mean_est'],
    'xi_temp': predictions_dict['xi_temp'],
    'xi_moist': predictions_dict['xi_moist'],
    'xi_season': predictions_dict['xi_season'],
    'Ea_by_treatment': predictions_dict['Ea_by_treatment'],
    'a_by_treatment': predictions_dict['a_by_treatment'],
    'b_by_treatment': predictions_dict['b_by_treatment'],
    # Store the mapping information for post-processing
    'site_treat_to_site': site_treat_to_site,
    'site_treat_to_treatment': site_treat_to_treatment,
    'sites': sites,
    'treatment_names': treatment_name,
    'valid_combinations': valid_combinations.to_dict('records')
}
# Print site×treatment level results
print(f"\n=== Site×Treatment Level Results ===")
print("Site×Treatment parameter estimates:")
for idx, row in valid_combinations.iterrows():
    site_name = sites[row['site_idx']]
    Ea_mean = jnp.mean(posterior_samples['Ea_site_treat'][:, idx])
    a_mean = jnp.mean(posterior_samples['a_site_treat'][:, idx])
    b_mean = jnp.mean(posterior_samples['b_site_treat'][:, idx])
    print(f"  {site_name} × T{row['treatment']}: Ea={Ea_mean:.1f}, a={a_mean:.2f}, b={b_mean:.2f}")

# Print site-level missing data results
print(f"\n=== Site-Level Missing Data Results ===")
site_M_means_posterior = predictions_dict['site_M_mean_scaled']
site_temp_means_posterior = predictions_dict['site_temp_mean_est']

print("Site-level moisture estimates:")
for i, site in enumerate(sites):
    mean_est = jnp.mean(site_M_means_posterior[:, i])
    std_est = jnp.std(site_M_means_posterior[:, i])
    print(f"  {site}: M = {mean_est:.3f} ± {std_est:.3f}")

print("\nSite-level temperature estimates:")
for i, site in enumerate(sites):
    mean_est = jnp.mean(site_temp_means_posterior[:, i])
    std_est = jnp.std(site_temp_means_posterior[:, i])
    print(f"  {site}: Temp = {mean_est:.2f} ± {std_est:.2f}°C")

# Save the complete results as a torch file
torch.save(complete_results, 'posterior_with_predictions.pt')
print("Posteriors+results saved to posterior_with_predictions.pt")

print(f"\n=== Final Summary ===")
print(f"Successfully completed MCMC with {mcmc.num_samples} samples per chain")
print(f"Hierarchical structure: Treatment-level → Site×Treatment-level → Plot-level")
print(f"Site×Treatment combinations: {n_site_treat_combinations} (filtered from {n_sites * Tr} possible)")
print(f"Site-level approach for missing data: {n_sites} sites instead of {n_missing_M_total + n_missing_temp_total} individual missing values")
print(f"Memory reduction: ~{old_memory_gb/new_memory_gb:.1f}x")
print(f"Divergences: {divergences}")

# Additional aggregation utilities for post-processing
print(f"\n=== Aggregation Information ===")
print("Available aggregation levels:")
print("  1. Site×Treatment level: Use 'site_treat_idx' with parameters like 'Ea_site_treat'")
print("  2. Site level: Use 'site_idx' with site-level parameters")
print("  3. Treatment level: Aggregate 'Ea_site_treat' by 'site_treat_to_treatment'")
print("  4. Global level: Use hyperpriors like 'Ea_mu'")

# Example aggregation code (commented out for now)
"""
# Example: Aggregate site×treatment parameters to treatment level
def aggregate_to_treatment_level(posterior_samples, param_name, site_treat_to_treatment, n_treatments):
    # Get site×treatment level parameter
    site_treat_param = posterior_samples[param_name]  # shape: (n_samples, n_site_treat_combinations)
    
    # Initialize treatment-level aggregation
    treatment_param = jnp.zeros((site_treat_param.shape[0], n_treatments))
    
    for t in range(1, n_treatments + 1):  # treatments are 1-indexed
        mask = site_treat_to_treatment == t
        if jnp.sum(mask) > 0:
            treatment_param = treatment_param.at[:, t-1].set(jnp.mean(site_treat_param[:, mask], axis=1))
    
    return treatment_param

# Example usage:
# Ea_by_treatment = aggregate_to_treatment_level(posterior_samples, 'Ea_site_treat', site_treat_to_treatment, Tr)
"""