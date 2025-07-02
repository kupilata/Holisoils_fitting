
# import the libraries
#source tf-env/bin/activate

import torch
import seaborn as sns
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt




posterior = torch.load("posterior_samples.pt", weights_only=False)
print("Keys in the dictionary:", posterior.keys())
posterior_Ea = posterior["Ea"]  # shape: [num_samples, Tr]
posterior_A = posterior["A"]  # shape: [num_samples, Pl]
posterior_a = posterior["a"]  # shape: [num_samples, Tr]
posterior_b = posterior["b"]  # shape: [num_samples, Tr]
posterior_amplitude = posterior["amplitude"]  # shape: [num_samples, Tr]
posterior_peak_day = posterior["peak_day"]  # shape: [num_samples, Tr]
posterior_sigma = posterior["sigma"]  # shape: [num_samples, Tr]
posterior_linear_mult = posterior["linear_mult"]  # shape: [num_samples, Tr]


#load the data, in case this is just an analysis run
fitting_data = pd.read_csv("python_holisoils.csv")
Tr = int(fitting_data['treatment'].max())
Pl = int(fitting_data['plot_id'].max())
# Create mapping from treatment to treatment_name
treatment_map = dict(zip(fitting_data['treatment'], fitting_data['treatment_name']))
# Get unique treatments and their corresponding names
unique_treatments = np.unique(fitting_data['treatment'])
unique_treatments_names = [treatment_map[t] for t in unique_treatments]


# Print R-hat for key parameters
idata = az.from_numpyro(mcmc)
rhat = az.rhat(idata)
print(f"Max R-hat: {max([rhat[param].max().values for param in rhat.data_vars]):.3f}")





## Bit of MCMC diagnostics
# Correct number of chains and draws
num_chains = 4
num_draws = 15000

# Reshape the tensors to (num_chains, num_draws, *shape)
posterior_reshaped = {
    key: value.reshape(num_chains, num_draws, -1) for key, value in posterior.items()
}

# Convert tensors to NumPy arrays for use with ArviZ
#posterior_dict = {key: value.numpy() for key, value in posterior_reshaped.items()}
posterior_dict = {key: np.array(value) for key, value in posterior_reshaped.items()}

# Create an InferenceData object with the correct dimensions
idata = az.from_dict(posterior=posterior_dict, coords={"chain": range(num_chains), "draw": range(num_draws)})

# Trace plots
plt.figure(figsize=(12, 6))
az.plot_trace(idata)
plt.show()

# Autocorrelation plots
#az.plot_autocorr(idata, combined=True)
#plt.show()

# Rank plots
#az.plot_rank(idata)
#plt.show()

rhat = az.rhat(idata)
ess = az.ess(idata)

max_rhat = max(np.max(rhat[var_name].values) for var_name in rhat.data_vars)


# Summary statistics, including R-hat and ESS
summary = az.summary(idata, kind='stats')
print(summary)





## Working on the chains
# Assuming posterior_Ea is your tensor
# Convert the tensor to a Pandas DataFrame for easier plotting with Seaborn
df_Ea = pd.DataFrame(np.array(posterior_Ea), columns=[f'Cat {i+1}' for i in range(posterior_Ea.shape[1])])
df_A = pd.DataFrame(np.array(posterior_A), columns=[f'Cat {i+1}' for i in range(posterior_A.shape[1])])
df_a = pd.DataFrame(np.array(posterior_a), columns=[f'Cat {i+1}' for i in range(posterior_a.shape[1])])
df_b = pd.DataFrame(np.array(posterior_b), columns=[f'Cat {i+1}' for i in range(posterior_b.shape[1])])
df_peak_day = pd.DataFrame(np.array(posterior_peak_day), columns=[f'Cat {i+1}' for i in range(posterior_peak_day.shape[1])])
df_amplitude = pd.DataFrame(np.array(posterior_amplitude), columns=[f'Cat {i+1}' for i in range(posterior_amplitude.shape[1])])
# For linear_mult (scalar parameter - single column)
df_linear_mult = pd.DataFrame(np.array(posterior_linear_mult), columns=['linear_mult'])



# Melt the DataFrame for Seaborn
df_Ea_melted = pd.melt(df_Ea, var_name='Category', value_name='Ea Value')
df_A_melted = pd.melt(df_A, var_name='Category', value_name='A Value')
df_a_melted = pd.melt(df_a, var_name='Category', value_name='a Value')
df_b_melted = pd.melt(df_b, var_name='Category', value_name='b Value')
df_peak_day_melted = pd.melt(df_peak_day, var_name='Category', value_name='Peak Day Value')
df_amplitude_melted = pd.melt(df_amplitude, var_name='Category', value_name='Amplitude Value')
df_linear_mult_melted = pd.melt(df_linear_mult, var_name='Category', value_name='Trench linear multiplier')


# Create a mapping from 'Cat X' to treatment names
category_mapping = {f'Cat {i+1}': name for i, name in enumerate(unique_treatments_names)}

# Apply the mapping
df_Ea_melted['Category'] = df_Ea_melted['Category'].map(category_mapping)
df_a_melted['Category'] =  df_a_melted['Category'].map(category_mapping)
df_b_melted['Category'] =  df_b_melted['Category'].map(category_mapping)
df_peak_day_melted['Category'] =  df_peak_day_melted['Category'].map(category_mapping)
df_amplitude_melted['Category'] =  df_amplitude_melted['Category'].map(category_mapping)


# Create the boxplot of Ea
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Ea Value', data=df_Ea_melted)
plt.ylabel('$E_0$ Value')
plt.xlabel('') # Remove x-axis label
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ea_boxplot.png')  # Add this line to save the first plot
plt.show()

# Create the plot of the linear multiplier
plt.figure(figsize=(8, 6))
plt.hist(df_linear_mult['linear_mult'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No effect (1.0)')
plt.xlabel('Linear Multiplier Value')
plt.ylabel('Frequency')
plt.title('Posterior Distribution of Trench Linear Multiplier')
plt.legend()
plt.savefig('linear_multiplier.png')  # Add this line to save the first plot
plt.show()

# Create a multi-panel layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Plot each boxplot in a separate panel
sns.boxplot(x='Category', y='a Value', data=df_a_melted, ax=axes[0, 0])
sns.boxplot(x='Category', y='b Value', data=df_b_melted, ax=axes[0, 1])
sns.boxplot(x='Category', y='Peak Day Value', data=df_peak_day_melted, ax=axes[1, 0])
sns.boxplot(x='Category', y='Amplitude Value', data=df_amplitude_melted, ax=axes[1, 1])
# Set titles for each subplot
axes[0, 0].set_title('Posterior a')
axes[0, 1].set_title('Posterior b')
axes[1, 0].set_title('Posterior Peak Day')
axes[1, 1].set_title('Posterior Amplitude')
# Remove "Category" labels and rotate x-axis labels for better readability
for ax in axes.flatten():
    ax.set_xlabel('') # Remove "Category" label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('other_params.png')  # This saves the second plot
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='A Value', data=df_A_melted)
plt.title('Posterior A')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('A.png')
plt.show()
