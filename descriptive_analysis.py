
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import data_pretreatment
wholedb_co2 = data_pretreatment.wholedb_co2

# Get number of rows
print(wholedb_co2.shape[0])

# Get number of columns
print(wholedb_co2.shape[1])

# Get all column names
print(wholedb_co2.columns)

# Overview of the dataframe
print(wholedb_co2.info())


# Replace DumbravitaTrench with Dumbravita
# The "Trenched" boolean takes care of that
wholedb_co2['siteid'] = wholedb_co2['siteid'].replace('DumbravitaTrench', 'Dumbravita')


###############Outlier removal

z_scores = np.abs(stats.zscore(wholedb_co2['merged_flux'].dropna()))

# Remove points with z-score > 3 (or 2.5 for stricter)
threshold = 3
outlier_mask = z_scores > threshold

# Apply to original dataframe (handling NaN values)
wholedb_co2_clean = wholedb_co2[~((np.abs(stats.zscore(wholedb_co2['merged_flux'], nan_policy='omit')) > threshold) & 
                                  (~wholedb_co2['merged_flux'].isna()))]
print(f"Original data: {len(wholedb_co2)} rows")
print(f"After removing outliers: {len(wholedb_co2_clean)} rows")
print(f"Removed: {len(wholedb_co2) - len(wholedb_co2_clean)} rows")


# Remove negative values
wholedb_co2_positive = wholedb_co2_clean[wholedb_co2_clean['merged_flux'] >= 0]
print(f"Before removing negatives: {len(wholedb_co2_clean)} rows")
print(f"After removing negatives: {len(wholedb_co2_positive)} rows")
print(f"Removed negative values: {len(wholedb_co2_clean) - len(wholedb_co2_positive)} rows")      

# remove values above 5
wholedb_co2_lowpass = wholedb_co2_positive[wholedb_co2_positive['merged_flux'] <= 5]
print(f"Before removing negatives: {len(wholedb_co2_positive)} rows")
print(f"After removing negatives: {len(wholedb_co2_lowpass)} rows")
print(f"Removed negative values: {len(wholedb_co2_positive) - len(wholedb_co2_lowpass)} rows")      

# Filter out rows where merged_flux is NaN
wholedb_co2_lowpass = wholedb_co2_lowpass.dropna(subset=['merged_flux'])

# Filter out rows where 'treatment' is NA
wholedb_co2_noNA = wholedb_co2_lowpass[~wholedb_co2_lowpass['treatment'].isna()]

# Check dimensions
print(wholedb_co2_lowpass.shape)
print(wholedb_co2_noNA.shape)

####################################################################################################
##### store the data for running the analysis on Puhti cluster
M_observed = wholedb_co2_noNA['tsmoisture'].copy()

# M_observed[is.na(M_observed)] <- mean(M_observed, na.rm = TRUE)
M_observed = M_observed.fillna(M_observed.mean())

# Convert date column to datetime if it's not already
wholedb_co2_noNA['date'] = pd.to_datetime(wholedb_co2_noNA['date'])

python_holisoils = pd.DataFrame({
    'treatment': wholedb_co2_noNA['treatment'].astype('category').cat.codes,  # Convert to numeric codes
    'plot_id': wholedb_co2_noNA['point'].astype('category').cat.codes,  # Convert plot IDs to numeric codes
    'day_year': wholedb_co2_noNA['date'].dt.dayofyear,
    'temp': wholedb_co2_noNA['soil_temp_5cm'],
    'resp': wholedb_co2_noNA['merged_flux'],
    'M_observed': M_observed,
    'M_missing': wholedb_co2_noNA['tsmoisture'].isna().astype(int),
    'year': wholedb_co2_noNA['date'].dt.year,
    'treatment_name': wholedb_co2_noNA['treatment'],
    'trenched': wholedb_co2_noNA['Trenched'],
    'site': wholedb_co2_noNA['siteid']
})
python_holisoils.to_csv("python_holisoils.csv", index=False)
####################################################################################################



# Create treatment mappings BEFORE any filtering
# Get unique treatments from the full dataset
all_treatments = wholedb_co2_lowpass['treatment'].unique()
n_treatments = len(all_treatments)

# Create color and marker mappings for all treatments
colors = sns.color_palette("tab10", n_treatments)
base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'X']
markers = (base_markers * ((n_treatments // len(base_markers)) + 1))[:n_treatments]

# Create treatment style mapping
treatment_style = {}
for i, treatment in enumerate(all_treatments):
    treatment_style[treatment] = (colors[i], markers[i])




############### Environmental variables plotting

print("=== NA Summary Before Filtering ===")
print("Trenched Data:")
trenched_na_summary = trenched_data.groupby('siteid')[['soil_temp_5cm', 'tsmoisture', 'merged_flux', 'treatment']].apply(lambda x: x.isna().sum())
print(trenched_na_summary)

print("\nUntrenched Data:")
untrenched_na_summary = untrenched_data.groupby('siteid')[['soil_temp_5cm', 'tsmoisture', 'merged_flux', 'treatment']].apply(lambda x: x.isna().sum())
print(untrenched_na_summary)

# Show total counts by site before filtering
print("\n=== Total Rows by Site Before Filtering ===")
print("Trenched:")
print(trenched_data.groupby('siteid').size())
print("\nUntrenched:")
print(untrenched_data.groupby('siteid').size())

# After filtering
print("\n=== Rows Remaining After Filtering ===")
print("Trenched:")
print(plot_data_trenched.groupby('siteid').size())
print("\nUntrenched:")
print(plot_data_untrenched.groupby('siteid').size())

# Summary of what was removed
print("\n=== Rows Removed by Site ===")
print("Trenched:")
trenched_before = trenched_data.groupby('siteid').size()
trenched_after = plot_data_trenched.groupby('siteid').size()
trenched_removed = trenched_before.subtract(trenched_after, fill_value=0)
print(trenched_removed)

print("\nUntrenched:")
untrenched_before = untrenched_data.groupby('siteid').size()
untrenched_after = plot_data_untrenched.groupby('siteid').size()
untrenched_removed = untrenched_before.subtract(untrenched_after, fill_value=0)
print(untrenched_removed)


# Filter for trenched data only
trenched_data = wholedb_co2_lowpass[wholedb_co2_lowpass['Trenched'] == True]
untrenched_data = wholedb_co2_lowpass[wholedb_co2_lowpass['Trenched'] == False]

# Remove rows with NaN in the variables we need
plot_data_trenched_temp = trenched_data.dropna(subset=['soil_temp_5cm', 'merged_flux', 'treatment'])
plot_data_untrenched_temp = untrenched_data.dropna(subset=['soil_temp_5cm', 'merged_flux', 'treatment'])
#two separate objects otherwise we remove all the NAs with the moisture missing
plot_data_trenched_moist = trenched_data.dropna(subset=['tsmoisture', 'merged_flux', 'treatment'])
plot_data_untrenched_moist = untrenched_data.dropna(subset=['tsmoisture','merged_flux', 'treatment'])



# Resolution of figures, very high DPI with larger fonts 
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.labelsize': 16,      # X and Y labels
    'axes.titlesize': 18,      # Subplot titles
    'xtick.labelsize': 12,     # X tick labels
    'ytick.labelsize': 12,     # Y tick labels
    'legend.fontsize': 12,     # Legend
    'figure.titlesize': 20     # Main title
})




# Create figure with two subplots, TEMPERATURE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  

# Calculate the common Y-axis range from both datasets
all_flux_values = pd.concat([plot_data_trenched['merged_flux'], 
                            plot_data_untrenched['merged_flux']])
y_min = all_flux_values.min()
y_max = all_flux_values.max()
# Add small padding to the range
y_padding = (y_max - y_min) * 0.05
y_range = [y_min - y_padding, y_max + y_padding]

# Plot 1: Trenched data
for treatment in all_treatments:
    treatment_data_trenched = plot_data_trenched_temp[plot_data_trenched_temp['treatment'] == treatment]
    if len(treatment_data_trenched) > 0:
        color, marker = treatment_style[treatment]
        ax1.scatter(treatment_data_trenched['soil_temp_5cm'], treatment_data_trenched['merged_flux'],
                   c=[color], marker=marker, s=60, alpha=0.7,
                   label=treatment, edgecolors='black', linewidth=0.5)

# Customize plot 1
ax1.set_xlabel('Soil Temperature 5cm (°C)', fontsize=12)
ax1.set_ylabel(r'CO$_2$ flux', fontsize=12)  # LaTeX notation
ax1.set_title('Trenched Data', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(y_range)  # Set common Y-axis range

# Add correlation coefficient for trenched
correlation_trenched = plot_data_trenched_temp['soil_temp_5cm'].corr(plot_data_trenched_temp['merged_flux'])
ax1.text(0.05, 0.95, f'r = {correlation_trenched:.3f}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for trenched
ax1.text(0.05, 0.88, f'n = {len(plot_data_trenched)}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Plot 2: Untrenched data
for treatment in all_treatments:
    treatment_data_untrenched = plot_data_untrenched_temp[plot_data_untrenched_temp['treatment'] == treatment]
    if len(treatment_data_untrenched) > 0:
        color, marker = treatment_style[treatment]
        ax2.scatter(treatment_data_untrenched['soil_temp_5cm'], treatment_data_untrenched['merged_flux'],
                   c=[color], marker=marker, s=60, alpha=0.7,
                   label=treatment, edgecolors='black', linewidth=0.5)

# Customize plot 2
ax2.set_xlabel('Soil Temperature 5cm (°C)', fontsize=12)
ax2.set_ylabel(r'CO$_2$ flux', fontsize=12)  # LaTeX notation
ax2.set_title('Untrenched Data', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(y_range)  # Set common Y-axis range

# Add correlation coefficient for untrenched
correlation_untrenched = plot_data_untrenched_temp['soil_temp_5cm'].corr(plot_data_untrenched_temp['merged_flux'])
ax2.text(0.05, 0.95, f'r = {correlation_untrenched:.3f}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for untrenched
ax2.text(0.05, 0.88, f'n = {len(plot_data_untrenched_temp)}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Add legend only to panel 1 (trenched data) inside the plot area
# Get handles and labels from ax1
handles1, labels1 = ax1.get_legend_handles_labels()

# Add legend inside panel 1 (upper right corner)
ax1.legend(handles1, labels1, title='Treatment', loc='upper right', 
          fontsize=8, title_fontsize=9, frameon=True, 
          fancybox=True, shadow=True, framealpha=0.9)

# Add overall title
fig.suptitle(r'CO$_2$ Flux vs Soil Temperature',
             fontsize=16, fontweight='bold')

# Adjust layout to accommodate legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Print statistics
print(f"Trenched data points: {len(plot_data_trenched_temp)}")
print(f"Untrenched data points: {len(plot_data_untrenched_temp)}")
print(f"Total treatments: {len(all_treatments)}")
print(f"Treatments: {sorted(all_treatments)}")
print(f"Y-axis range: {y_min:.2f} to {y_max:.2f}")

# Save the figure as PNG

plt.savefig('co2_flux_temperature_comparison.png', 
            dpi=600,                    # Very high DPI for manuscripts
            bbox_inches='tight',
            facecolor='white', 
            edgecolor='none',
            format='png',
            pil_kwargs={'optimize': True})

print("Figure saved as 'co2_flux_temperature_comparison.png'")

# Show the plot
plt.show()






# Create figure with two subplots, MOISTURE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  

# Calculate the common Y-axis range from both datasets
all_flux_values = pd.concat([plot_data_trenched_moist['merged_flux'], 
                            plot_data_untrenched_moist['merged_flux']])
y_min = all_flux_values.min()
y_max = all_flux_values.max()
# Add small padding to the range
y_padding = (y_max - y_min) * 0.05
y_range = [y_min - y_padding, y_max + y_padding]

# Plot 1: Trenched data
for treatment in all_treatments:
    treatment_data_trenched = plot_data_trenched_moist[plot_data_trenched_moist['treatment'] == treatment]
    if len(treatment_data_trenched) > 0:
        color, marker = treatment_style[treatment]
        ax1.scatter(treatment_data_trenched['tsmoisture'], treatment_data_trenched['merged_flux'],
                   c=[color], marker=marker, s=60, alpha=0.7,
                   label=treatment, edgecolors='black', linewidth=0.5)

# Customize plot 1
ax1.set_xlabel('Soil Moisture 5cm', fontsize=12)
ax1.set_ylabel(r'CO$_2$ flux', fontsize=12)  # LaTeX notation
ax1.set_title('Trenched Data', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(y_range)  # Set common Y-axis range

# Add correlation coefficient for trenched
correlation_trenched = plot_data_trenched_moist['tsmoisture'].corr(plot_data_trenched_moist['merged_flux'])
ax1.text(0.05, 0.95, f'r = {correlation_trenched:.3f}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for trenched
ax1.text(0.05, 0.88, f'n = {len(plot_data_trenched_moist)}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Plot 2: Untrenched data
for treatment in all_treatments:
    treatment_data_untrenched = plot_data_untrenched_moist[plot_data_untrenched_moist['treatment'] == treatment]
    if len(treatment_data_untrenched) > 0:
        color, marker = treatment_style[treatment]
        ax2.scatter(treatment_data_untrenched['tsmoisture'], treatment_data_untrenched['merged_flux'],
                   c=[color], marker=marker, s=60, alpha=0.7,
                   label=treatment, edgecolors='black', linewidth=0.5)

# Customize plot 2
ax2.set_xlabel('Soil Moisture 5cm', fontsize=12)
ax2.set_ylabel(r'CO$_2$ flux', fontsize=12)  # LaTeX notation
ax2.set_title('Untrenched Data', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(y_range)  # Set common Y-axis range

# Add correlation coefficient for untrenched
correlation_untrenched = plot_data_untrenched_moist['tsmoisture'].corr(plot_data_untrenched_moist['merged_flux'])
ax2.text(0.05, 0.95, f'r = {correlation_untrenched:.3f}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for untrenched
ax2.text(0.05, 0.88, f'n = {len(plot_data_untrenched_moist)}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Add legend only to panel 1 (trenched data) inside the plot area
# Get handles and labels from ax1
handles1, labels1 = ax1.get_legend_handles_labels()

# Add legend inside panel 1 (upper right corner)
ax1.legend(handles1, labels1, title='Treatment', loc='upper right', 
          fontsize=8, title_fontsize=9, frameon=True, 
          fancybox=True, shadow=True, framealpha=0.9)

# Add overall title
fig.suptitle(r'CO$_2$ Flux vs Soil Moisture',
             fontsize=16, fontweight='bold')

# Adjust layout to accommodate legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Print statistics
print(f"Trenched data points: {len(plot_data_trenched_moist)}")
print(f"Untrenched data points: {len(plot_data_untrenched_moist)}")
print(f"Total treatments: {len(all_treatments)}")
print(f"Treatments: {sorted(all_treatments)}")
print(f"Y-axis range: {y_min:.2f} to {y_max:.2f}")

# Save the figure as PNG
plt.savefig('co2_flux_moisture_comparison.png', 
            dpi=600,                    # Very high DPI for manuscripts
            bbox_inches='tight',
            facecolor='white', 
            edgecolor='none',
            format='png',
            pil_kwargs={'optimize': True})
print("Figure saved as 'co2_flux_moisture_comparison.png'")

# Show the plot
plt.show()




# Create figure with two subplots, MOISTURE vs TEMPERATURE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  

# Calculate the common Y-axis range from both datasets (now for soil temperature)
all_temp_values = pd.concat([plot_data_trenched_moist['soil_temp_5cm'], 
                            plot_data_untrenched_moist['soil_temp_5cm']])
y_min = all_temp_values.min()
y_max = all_temp_values.max()
# Add small padding to the range
y_padding = (y_max - y_min) * 0.05
y_range = [y_min - y_padding, y_max + y_padding]

# Plot 1: Trenched data
for treatment in all_treatments:
    treatment_data_trenched = plot_data_trenched_moist[plot_data_trenched_moist['treatment'] == treatment]
    if len(treatment_data_trenched) > 0:
        color, marker = treatment_style[treatment]
        ax1.scatter(treatment_data_trenched['tsmoisture'], treatment_data_trenched['soil_temp_5cm'],
                   c=[color], marker=marker, s=60, alpha=0.7,
                   label=treatment, edgecolors='black', linewidth=0.5)

# Customize plot 1
ax1.set_xlabel('Soil Moisture', fontsize=12)
ax1.set_ylabel('Soil Temperature 5cm (°C)', fontsize=12)
ax1.set_title('Trenched Data', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(y_range)  # Set common Y-axis range

# Add correlation coefficient for trenched
correlation_trenched = plot_data_trenched_moist['tsmoisture'].corr(plot_data_trenched_moist['soil_temp_5cm'])
ax1.text(0.05, 0.95, f'r = {correlation_trenched:.3f}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for trenched
ax1.text(0.05, 0.88, f'n = {len(plot_data_trenched_moist)}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Plot 2: Untrenched data
for treatment in all_treatments:
    treatment_data_untrenched = plot_data_untrenched_moist[plot_data_untrenched_moist['treatment'] == treatment]
    if len(treatment_data_untrenched) > 0:
        color, marker = treatment_style[treatment]
        ax2.scatter(treatment_data_untrenched['tsmoisture'], treatment_data_untrenched['soil_temp_5cm'],
                   c=[color], marker=marker, s=60, alpha=0.7,
                   label=treatment, edgecolors='black', linewidth=0.5)

# Customize plot 2
ax2.set_xlabel('Soil Moisture', fontsize=12)
ax2.set_ylabel('Soil Temperature 5cm (°C)', fontsize=12)
ax2.set_title('Untrenched Data', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(y_range)  # Set common Y-axis range

# Add correlation coefficient for untrenched
correlation_untrenched = plot_data_untrenched_moist['tsmoisture'].corr(plot_data_untrenched_moist['soil_temp_5cm'])
ax2.text(0.05, 0.95, f'r = {correlation_untrenched:.3f}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for untrenched
ax2.text(0.05, 0.88, f'n = {len(plot_data_untrenched_moist)}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Add legend only to panel 1 (trenched data) inside the plot area
# Get handles and labels from ax1
handles1, labels1 = ax1.get_legend_handles_labels()

# Add legend inside panel 1 (upper right corner)
ax1.legend(handles1, labels1, title='Treatment', loc='upper right', 
          fontsize=8, title_fontsize=9, frameon=True, 
          fancybox=True, shadow=True, framealpha=0.9)

# Add overall title
fig.suptitle('Soil Temperature vs Soil Moisture: Trenched vs Untrenched Data',
             fontsize=16, fontweight='bold')

# Adjust layout to accommodate legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Print statistics
print(f"Trenched data points: {len(plot_data_trenched_moist)}")
print(f"Untrenched data points: {len(plot_data_untrenched_moist)}")
print(f"Total treatments: {len(all_treatments)}")
print(f"Treatments: {sorted(all_treatments)}")
print(f"Y-axis range (temperature): {y_min:.2f} to {y_max:.2f} °C")

# Save the figure as PNG
plt.savefig('soil_temperature_moisture_comparison.png',
            dpi=600,                    # Very high DPI for manuscripts
            bbox_inches='tight',
            facecolor='white', 
            edgecolor='none',
            format='png',
            pil_kwargs={'optimize': True})
print("Figure saved as 'soil_temperature_moisture_comparison.png'")

# Show the plot
plt.show()




############### Respiration time series


# Ensure date column is datetime and filter nulls
wholedb_co2['date'] = pd.to_datetime(wholedb_co2_lowpass['date'])
data_clean = wholedb_co2_lowpass.dropna(subset=['merged_flux'])


# Create FacetGrid
g = sns.FacetGrid(data_clean, col='siteid', col_wrap=4, height=3, aspect=1.2)

# Map line plot for each treatment
g.map_dataframe(sns.lineplot, x='date', y='merged_flux', hue='treatment', style='Trenched', alpha=0.8, palette=colors)


# Customize
g.set_axis_labels('Date', 'CO₂ Flux (merged)')
g.set_titles('Site: {col_name}')

# Rotate x-axis labels
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45)

# Set same y-axis range for all subplots
y_min = data_clean['merged_flux'].min()
y_max = data_clean['merged_flux'].max()
y_range = y_max - y_min
y_margin = 0.05 * y_range  # 5% margin

for ax in g.axes.flat:
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

# Set same x-axis range for all subplots
x_min = data_clean['date'].min()
x_max = data_clean['date'].max()

for ax in g.axes.flat:
    ax.set_xlim(x_min, x_max)

# Adjust subplot parameters to make room for legend
plt.subplots_adjust(right=0.78)

# Get legend handles and labels from the first axis
handles, labels = g.axes.flat[0].get_legend_handles_labels()

# Add legend to the figure
g.fig.legend(handles, labels, title='Treatment', bbox_to_anchor=(0.79, 0.5), loc='center left')

# Save as PNG
plt.savefig('co2_flux_timeseries.png',
            dpi=600,                    # Very high DPI for manuscripts
            bbox_inches='tight',
            facecolor='white', 
            edgecolor='none',
            format='png',
            pil_kwargs={'optimize': True})
plt.show()