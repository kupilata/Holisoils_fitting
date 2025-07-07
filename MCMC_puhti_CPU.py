import multiprocessing
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import time
import pandas as pd

# Set the start method for multiprocessing
# Try these settings for your cluster
#multiprocessing.set_start_method('fork', force=True)  # Often better for Linux clusters
#torch.set_num_threads(22)  # Match your cluster's thread allocation
torch.set_num_threads(4)  # Use 4 threads per chain, x 5 chains
#torch.set_num_interop_threads(1)  # Reduce thread contention

fitting_data = pd.read_csv("python_holisoils.csv")
Tr = int(fitting_data['treatment'].max())
Pl = int(fitting_data['plot_id'].max())

# Define the model function at the top level
def model(Tr, Pl, treatment, plot_id, day_year, temp, resp, M):
    Ea = pyro.sample("Ea", dist.Normal(398.5, 20.0).expand([Tr]).to_event(1))
    A = pyro.sample("A", dist.Normal(400.0, 50.0).expand([Pl]).to_event(1))
    a = pyro.sample("a", dist.Normal(3.11, 1.0).expand([Tr]).to_event(1))
    b = pyro.sample("b", dist.Normal(2.42, 1.0).expand([Tr]).to_event(1))
    amplitude = pyro.sample("amplitude", dist.Normal(0.0, 1.0).expand([Tr]).to_event(1))
    peak_day = pyro.sample("peak_day", dist.Normal(196.0, 10.0).expand([Tr]).to_event(1))

    xi_temp = torch.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0))
    xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
    xi_season = amplitude[tr_idx] * torch.cos(
        (2 * torch.pi / 365) * day_year +
        (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) -
        torch.pi / 2
    )
    model_resp = A[pl_idx] * xi_temp * xi_moist * xi_season

    sigma = pyro.sample('sigma', dist.HalfNormal(0.1))
    pyro.sample("obs", dist.Normal(model_resp, sigma), obs=resp)


# Extract data as CPU tensors
def to_tensor(col):
    return torch.tensor(fitting_data[col].values, dtype=torch.float32)

# Your data and initial parameters
treatment = to_tensor('treatment').contiguous()
plot_id = to_tensor('plot_id').contiguous()
day_year = to_tensor('day_year').contiguous()
temp = to_tensor('temp').contiguous()
resp = to_tensor('resp').contiguous()
M = to_tensor('M_observed').contiguous()
T_0 = 227.13
tr_idx = (treatment - 1).long()
pl_idx = (plot_id - 1).long()

initial_params = {
    "Ea": torch.tensor(398.5).expand([Tr]),
    "A": torch.tensor(400.0).expand([Pl]),
    "a": torch.tensor(3.11).expand([Tr]),
    "b": torch.tensor(2.42).expand([Tr]),
    "amplitude": torch.zeros(Tr),
    "peak_day": torch.tensor(196.0).expand([Tr]),
    "sigma": torch.tensor(0.1)
}

if __name__ == '__main__':
    mcmc = None  # Define mcmc outside the try block
    try:
        # Initialize the NUTS kernel and MCMC
        nuts_kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=9, 
                           init_strategy=pyro.infer.autoguide.init_to_value(values=initial_params),
                    adapt_step_size=True,
                    adapt_mass_matrix=True)
        mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=500, num_chains=5)
        
        # Run the MCMC
        start_time = time.time()
        mcmc.run(Tr, Pl, treatment, plot_id, day_year, temp, resp, M)
        end_time = time.time()
        print(f"\nAuto-parallel MCMC on CPU finished in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
    
    # Only attempt to get samples if mcmc was successfully created and run
    if mcmc is not None:
        try:
            # Get posterior samples
            posterior = mcmc.get_samples()
            # Save the posterior samples as a torch file
            torch.save(posterior, 'posterior_samples.pt')
            print("Results saved to posterior_samples.pt")
        except Exception as e:
            print(f"Error saving results: {e}")