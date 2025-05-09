# source tf-env/bin/activate  #to activate the virtual environment 
# (from command line)
# wifwu7-kejtys-foZwet
# cd /scratch/project_2010938
# sbatch MCMC_multiGPU_run.sh #to run a task
# squeue -l -u $USER #to check the queue
# csc-projects

#local folder /Users/ilmenichetti/Library/CloudStorage/OneDrive-Valtion/Holisoils/Holisoils_fitting

# import the libraries
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# ---------------------------------------
# LOAD AND PREP DATA
# ---------------------------------------

fitting_data = pd.read_csv("python_holisoils.csv")
print("Data loaded:", fitting_data.shape)

# Metadata (Bayesian stratification layers)
Tr = int(fitting_data['treatment'].max()) # number of treatments
Pl = int(fitting_data['plot_id'].max()) # number of plots

# Convert data to tensors (still on CPU here)
def to_tensor(col):
    return torch.tensor(fitting_data[col].values, dtype=torch.float32)

treatment = to_tensor('treatment')
plot_id = to_tensor('plot_id')
day_year = to_tensor('day_year')
temp = to_tensor('temp')
resp = to_tensor('resp')
M = to_tensor('M_observed')

# ---------------------------------------
# MODEL DEFINITION
# ---------------------------------------

class RespirationModel(nn.Module): # set up the model class to be used by pyro, inheriting from the nn class (basyc Pytorch class for neural networks)
    def __init__(self, Tr, Pl, device):  #defining the constructor method
        # these objects are the same across all the N iterations
        super().__init__() #calling the constructor for nn models, inherited from nn.Module. Needed for Pytorch to work.
        self.Tr = Tr
        self.Pl = Pl
        self.device = device
        # although the calculation is vectorized and it runs just once whenever it's called, the terms
        # Tr, Pl and device do not change even across the sampler iterations, so they are initialized. THe sampler needs this.

    def forward(self, treatment, plot_id, day_year, temp, resp, M): #the forward part defines the computations repeated at every cycle, so iterated over N
        T_0 = 227.13
        
        #sigma = pyro.sample('sigma', dist.Normal(torch.tensor(0.2712706, device=self.device), 0.05))
        #A = pyro.sample('A', dist.Normal(torch.tensor(400.0, device=self.device), 100.0).expand([self.Pl]).to_event(1))
        #Ea = pyro.sample('Ea', dist.Normal(torch.tensor(398.5, device=self.device), 50.0).expand([self.Tr]).to_event(1))
        #a = pyro.sample('a', dist.Normal(torch.tensor(3.11, device=self.device), 0.25).expand([self.Tr]).to_event(1))
        #b = pyro.sample('b', dist.Normal(torch.tensor(2.42, device=self.device), 0.25).expand([self.Tr]).to_event(1))
        #amplitude = pyro.sample('amplitude', dist.Normal(torch.tensor(0.0, device=self.device), 0.25).expand([self.Tr]).to_event(1))
        #peak_day = pyro.sample('peak_day', dist.Normal(torch.tensor(196.0, device=self.device), 50.0).expand([self.Tr]).to_event(1))

        #trying to optimise moving all constantrs to the GPU explicitly
        sigma = pyro.sample('sigma', dist.Normal(
            torch.tensor(0.2712706, device=self.device), 
            torch.tensor(0.05, device=self.device)))

        A = pyro.sample('A', dist.Normal(
            torch.tensor(400.0, device=self.device), 
            torch.tensor(100.0, device=self.device)).expand([self.Pl]).to_event(1))

        Ea = pyro.sample('Ea', dist.Normal(
            torch.tensor(398.5, device=self.device), 
            torch.tensor(50.0, device=self.device)).expand([self.Tr]).to_event(1))

        a = pyro.sample('a', dist.Normal(
            torch.tensor(3.11, device=self.device), 
            torch.tensor(0.25, device=self.device)).expand([self.Tr]).to_event(1))

        b = pyro.sample('b', dist.Normal(
            torch.tensor(2.42, device=self.device), 
            torch.tensor(0.25, device=self.device)).expand([self.Tr]).to_event(1))

        amplitude = pyro.sample('amplitude', dist.Normal(
            torch.tensor(0.0, device=self.device), 
            torch.tensor(0.25, device=self.device)).expand([self.Tr]).to_event(1))

        peak_day = pyro.sample('peak_day', dist.Normal(
            torch.tensor(196.0, device=self.device), 
            torch.tensor(50.0, device=self.device)).expand([self.Tr]).to_event(1))

        tr_idx = (treatment - 1).long()
        pl_idx = (plot_id - 1).long()

        xi_moist = a[tr_idx] * M - b[tr_idx] * M**2
        xi_temp = A[pl_idx] * torch.exp(-Ea[tr_idx] / ((temp + 273.15) - T_0))
        sine_wave = amplitude[tr_idx] * torch.cos((2 * torch.pi / 365) * day_year +
                                                  (2 * torch.pi / 365) * (peak_day[tr_idx] - 1) - torch.pi / 2)

        model_resp = sine_wave + (xi_temp * xi_moist)
        pyro.sample('obs', dist.Normal(model_resp, sigma), obs=resp)
        return model_resp

# ---------------------------------------
# WORKER FUNCTION FOR EACH GPU
# this part routes the jobs to each of the multiple devices
# ---------------------------------------

def run_chain_on_device(device_id, treatment, plot_id, day_year, temp, resp, M, Tr, Pl, return_dict):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Move tensors to the device (CUDA)
    treatment = treatment.to(device)
    plot_id = plot_id.to(device)
    day_year = day_year.to(device)
    temp = temp.to(device)
    resp = resp.to(device)
    M = M.to(device)

    # Set seed 
    pyro.set_rng_seed(1234 + device_id)

    print(f"Starting MCMC on GPU {device_id}...")

    model = RespirationModel(Tr, Pl, device)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=2000, warmup_steps=500, num_chains=1)
    mcmc.run(treatment, plot_id, day_year, temp, resp, M)

    samples = mcmc.get_samples()
    print(f"Finished MCMC on GPU {device_id}.")
    return_dict[device_id] = samples #line to write results in the dictionary shared across devices

# ---------------------------------------
# MAIN: PARALLEL EXECUTION
# ---------------------------------------

def run_multi_gpu_mcmc():
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")

    manager = mp.Manager() # creates a multi-process manager to share the data among them
    return_dict = manager.dict() # shared "dictionary", each process stores results here
    processes = [] #list to store the processes

    for gpu_id in range(num_gpus):
        p = mp.Process(target=run_chain_on_device, args=(
            gpu_id, treatment, plot_id, day_year, temp, resp, M, Tr, Pl, return_dict
        )) # create a process
        p.start() #starts it 
        processes.append(p) #store the process in the list processes

    for p in processes:
        p.join() # this for loop waits for all processes to be completed

    print("All GPU chains completed.")
    return return_dict #output the common dictionary once they all run

# ---------------------------------------
# ENTRY POINT, running the actual sampling
# ---------------------------------------

if __name__ == '__main__': # only executed when the script is run directly, not imported as module
    #the if above is for reusability, in case one would want to import the functions above as modules. Supposedly can avoid recursions.
    
    # this sets up the method for starting new processes, for safe multiprocessing
    mp.set_start_method('spawn', force=True) #spawn means that new processes are start from scratches on a new python interpreter. "fork" would copy them, definitely not safe.
    start = time.time()
    all_samples = run_multi_gpu_mcmc()
    end = time.time()

    print(f"\nTotal execution time: {end - start:.1f} seconds")

    # Optional: combine samples
    # combined_Ea = torch.cat([all_samples[i]['Ea'] for i in sorted(all_samples)], dim=0)
    # print(f"Combined Ea samples shape: {combined_Ea.shape}")

    ### UNTESTED BELOW, NEVER RUN IT YET!!!!!!!!!

    # Combine samples for all parameters
    combined_samples = {}
    
    # For each parameter, concatenate the samples across devices (GPUs)
    for param_name in ['Ea', 'A', 'a', 'b', 'amplitude', 'peak_day', 'sigma']:
        combined_param = torch.cat([all_samples[i][param_name] for i in sorted(all_samples)], dim=0)
        combined_samples[param_name] = combined_param
        print(f"Combined {param_name} samples shape: {combined_param.shape}")

    # Save the combined samples to disk as PyTorch files
    torch.save(combined_samples, 'combined_samples.pt')
    print("Samples saved to 'combined_samples.pt'.")
