
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


#OUtlier removal

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


##### plotting
# Filter for trenched data only
trenched_data = wholedb_co2_lowpass[wholedb_co2_lowpass['Trenched'] == True]
untrenched_data = wholedb_co2_lowpass[wholedb_co2_lowpass['Trenched'] == False]

# Remove rows with NaN in the variables we need
plot_data_trenched = trenched_data.dropna(subset=['soil_temp_5cm', 'merged_flux', 'treatment'])
plot_data_untrenched = untrenched_data.dropna(subset=['soil_temp_5cm', 'merged_flux', 'treatment'])

# Get ALL unique treatments from both datasets to ensure consistent legend
all_treatments = pd.concat([plot_data_trenched['treatment'], 
                           plot_data_untrenched['treatment']]).unique()

n_treatments = len(all_treatments)

# Create color and marker mappings for all treatments
colors = sns.color_palette("tab10", n_treatments)
base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'X']
markers = (base_markers * ((n_treatments // len(base_markers)) + 1))[:n_treatments]

# Create treatment style mapping
treatment_style = {}
for i, treatment in enumerate(all_treatments):
        treatment_style[treatment] = (colors[i], markers[i])



# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

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
    treatment_data_trenched = plot_data_trenched[plot_data_trenched['treatment'] == treatment]
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
correlation_trenched = plot_data_trenched['soil_temp_5cm'].corr(plot_data_trenched['merged_flux'])
ax1.text(0.05, 0.95, f'r = {correlation_trenched:.3f}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for trenched
ax1.text(0.05, 0.88, f'n = {len(plot_data_trenched)}', transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        fontsize=10)

# Plot 2: Untrenched data
for treatment in all_treatments:
    treatment_data_untrenched = plot_data_untrenched[plot_data_untrenched['treatment'] == treatment]
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
correlation_untrenched = plot_data_untrenched['soil_temp_5cm'].corr(plot_data_untrenched['merged_flux'])
ax2.text(0.05, 0.95, f'r = {correlation_untrenched:.3f}', transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=11)

# Add data count for untrenched
ax2.text(0.05, 0.88, f'n = {len(plot_data_untrenched)}', transform=ax2.transAxes,
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
print(f"Trenched data points: {len(plot_data_trenched)}")
print(f"Untrenched data points: {len(plot_data_untrenched)}")
print(f"Total treatments: {len(all_treatments)}")
print(f"Treatments: {sorted(all_treatments)}")
print(f"Y-axis range: {y_min:.2f} to {y_max:.2f}")

# Save the figure as PNG
plt.savefig('co2_flux_temperature_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Figure saved as 'co2_flux_temperature_comparison.png'")

# Show the plot
plt.show()