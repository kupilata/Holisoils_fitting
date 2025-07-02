
import pandas as pd
import pyarrow
import numpy as np

wholedb = pd.read_csv("../Data/Holisoils_GHG_data/All_sites.csv", dtype_backend='pyarrow')  # if you have pyarrow installed
print(wholedb['siteid'].unique())


# Fix site names
wholedb.loc[wholedb['siteid'] == "Zwolse bos", 'siteid'] = "Zwolse Bos"
wholedb.loc[wholedb['siteid'] == "Kroondomein ", 'siteid'] = "Kroondomein"

# Assigning the countries
wholedb['country'] = np.nan

wholedb.loc[wholedb['siteid'] == "Dobroc", 'country'] = "Slovakia"
wholedb.loc[wholedb['siteid'] == "Dumbravita", 'country'] = "Romania"
wholedb.loc[wholedb['siteid'] == "DumbravitaTrench", 'country'] = "Romania"
wholedb.loc[wholedb['siteid'] == "Gamiz", 'country'] = "Spain"
wholedb.loc[wholedb['siteid'] == "Karstula75", 'country'] = "Finland"
wholedb.loc[wholedb['siteid'] == "Karstula76", 'country'] = "Finland"
wholedb.loc[wholedb['siteid'] == "Kelheim-Parsberg", 'country'] = "Germany"
wholedb.loc[wholedb['siteid'] == "Kranzberg-Freising", 'country'] = "Germany"
wholedb.loc[wholedb['siteid'] == "Kroondomein", 'country'] = "Netherlands"
wholedb.loc[wholedb['siteid'] == "Zwolse Bos", 'country'] = "Netherlands"
wholedb.loc[wholedb['siteid'] == "NP Hoge Veluwe", 'country'] = "Netherlands"
wholedb.loc[wholedb['siteid'] == "Llobera", 'country'] = "Spain"
wholedb.loc[wholedb['siteid'] == "Ränskälänkorpi", 'country'] = "Finland"
wholedb.loc[wholedb['siteid'] == "Saint Mitre", 'country'] = "France"
wholedb.loc[wholedb['siteid'] == "Secanella", 'country'] = "Spain"
wholedb.loc[wholedb['siteid'] == "St Christol", 'country'] = "France"
wholedb.loc[wholedb['siteid'] == "Wasserburg-Maitenbeth", 'country'] = "Germany"

# Replace "nan" and "0" with NaN in pointtype
wholedb.loc[wholedb['pointtype'].isin(["nan", "0"]), 'pointtype'] = np.nan

# Trenched yes no or NA
print(wholedb['pointtype'].unique())
wholedb['Trenched'] = np.nan

# Making uniform the Finnish sites
trenched_types = ["Trenched", "Trenched, without fabric lid", "Trenched, with fabric lid"]
wholedb.loc[wholedb['pointtype'].isin(trenched_types), 'Trenched'] = True
print(wholedb['Trenched'].value_counts(dropna=False))

# Sites from WP4, in some cases they specified the trenching in the subsiteid
wholedb.loc[wholedb['subsiteid'].str.contains("trenching", case=False, na=False), 'Trenched'] = True
print(wholedb.loc[wholedb['subsiteid'] == "No thinning / Trenching", 'Trenched'])  # checking

wholedb.loc[wholedb['siteid'] == "DumbravitaTrench", 'Trenched'] = True

# Set to FALSE the untrenched points of sites with trenching
siteids_trenched = wholedb.loc[wholedb['Trenched'] == True, 'siteid'].unique()
print(siteids_trenched)

for site in siteids_trenched:
    mask = (wholedb['Trenched'].isna()) & (wholedb['siteid'] == site)
    wholedb.loc[mask, 'Trenched'] = False

# Dobroc has an error, all are trenched, info is in file ID
dobroc_subset = wholedb[wholedb['siteid'] == "Dobroc"]
dobroc_mask = wholedb['siteid'] == "Dobroc"

wholedb.loc[dobroc_mask, 'Trenched'] = np.where(
    dobroc_subset['point'].str.contains("C", na=False), False,
    np.where(dobroc_subset['point'].str.contains("T", na=False), True, np.nan)
)


# Convert to regular bool (not nullable boolean)
wholedb['Trenched'] = wholedb['Trenched'].astype(bool)
wholedb['Trenched'] = wholedb['Trenched'].astype('boolean')
print(wholedb['Trenched'].value_counts(dropna=False))

print(wholedb['id'].isna().sum())  # equivalent to which(is.na())




group1 = ["Dobroc", "Kelheim-Parsberg", "Kranzberg-Freising", "Wasserburg-Maitenbeth"]  # mixed stands
group2 = ["Kroondomein", "Zwolse Bos", "NP Hoge Veluwe", "Saint Mitre"]  # WP4 thinning
group3 = ["Dumbravita", "Gamiz", "St Christol"]  # WP5 thinning
group4 = ["Llobera", "Secanella"]  # forest fires
group5 = ["Karstula75", "Karstula76"]  # N treatments, Lithuania goes here
group6 = ["Ränskälänkorpi"]  # peat treatment
group7 = ["DumbravitaTrench"]  # WP5 trenching, ground vegetation

wholedb['group'] = np.nan

wholedb.loc[wholedb['siteid'].isin(group1), 'group'] = "mixture"
wholedb.loc[wholedb['siteid'].isin(group2), 'group'] = "thinning"
wholedb.loc[wholedb['siteid'].isin(group3), 'group'] = "slash"
wholedb.loc[wholedb['siteid'].isin(group4), 'group'] = "burning"
wholedb.loc[wholedb['siteid'].isin(group5), 'group'] = "N_treatment"
wholedb.loc[wholedb['siteid'].isin(group6), 'group'] = "peat"
wholedb.loc[wholedb['siteid'].isin(group7), 'group'] = "ground_vegetation"

wholedb['treatment'] = np.nan
wholedb['control'] = np.nan

# treatments for mixture group (group 1)
print(wholedb.loc[wholedb['siteid'].isin(group1), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group1), 'pointtype'].unique())

wholedb.loc[(wholedb['siteid'].isin(group1)) & (wholedb['subsiteid'] == "DP-MC"), 'control'] = True
wholedb.loc[(wholedb['siteid'].isin(group1)) & (wholedb['subsiteid'] == "Spruce"), 'control'] = True
wholedb.loc[wholedb['siteid'].isin(group1), 'treatment'] = "mixture"

print(len(wholedb.loc[wholedb['siteid'].isin(group1), 'treatment'].isna()))
print(wholedb.loc[wholedb['siteid'].isin(group1), 'treatment'])

# checking that the Trenched boolean is correctly assigned
print(wholedb.loc[(wholedb['siteid'].isin(group1)) & (wholedb['Trenched'] == True), 'pointtype'])
print(wholedb.loc[wholedb['siteid'].isin(group1), 'treatment'].unique())

# treatments for thinning group (group 2)
print(wholedb.loc[wholedb['siteid'].isin(group2), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group2), 'pointtype'].unique())

wholedb.loc[wholedb['siteid'].isin(group2), 'treatment'] = wholedb.loc[wholedb['siteid'].isin(group2), 'subsiteid']

# finish setting the trenched boolean
mask_trenching = (wholedb['siteid'].isin(group2)) & (wholedb['treatment'].str.contains("Trenching", na=False))
wholedb.loc[mask_trenching, 'Trenched'] = True

mask_control = (wholedb['siteid'].isin(group2)) & (wholedb['treatment'] == "Control") & (~wholedb['treatment'].isna())
wholedb.loc[mask_control, 'control'] = True

# String replacements for group 2
group2_mask = wholedb['siteid'].isin(group2)
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("Shelterwood", "shelterwood")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("High-thinning", "high-thinning")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("Clearcut", "clearcut")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("Medium thinning / Trenching", "medium-thinning")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("Medium thinning / Pine without understory vegetation", "medium-thinning_no_understorey")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("Medium thinning / Pine with understory vegetation", "medium-thinning")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("High thinning / Trenching", "high-thinning")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("High thinning / Pine without understory vegetation", "high-thinning_no_understorey")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("High thinning / Pine with understory vegetation", "high-thinning")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("No thinning / Trenching", "no-thinning")
wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'treatment'].str.replace("No thinning / Pine without understory vegetation", "no-thinning")

print(wholedb.loc[wholedb['siteid'].isin(group2), 'treatment'].unique())

# treatments for slash group (group 3)
print(wholedb.loc[wholedb['siteid'].isin(group3), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group3), 'pointtype'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group3), 'siteid'].unique())

wholedb.loc[wholedb['siteid'].isin(group3), 'subsiteid'] = wholedb.loc[wholedb['siteid'].isin(group3), 'subsiteid'].str.replace("-", "", regex=False)

mask_control_3 = (wholedb['siteid'].isin(group3)) & (wholedb['pointtype'] == "control") & (~wholedb['pointtype'].isna())
wholedb.loc[mask_control_3, 'control'] = True

print(wholedb.loc[wholedb['siteid'].isin(group3), 'subsiteid'].unique())

wholedb.loc[wholedb['siteid'].isin(group3), 'treatment'] = wholedb.loc[wholedb['siteid'].isin(group3), 'subsiteid']

# String replacements for group 3
group3_mask = wholedb['siteid'].isin(group3)
wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass\\+shrubs/with understorey", "grass+shrubs_no_understorey", regex=True)
wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass\\+shrubs/no understorey", "grass+shrubs_no_understorey", regex=True)
wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass/with understorey", "grass", regex=True)
wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass/no understorey", "grass_no_understorey", regex=True)
wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("clear_cut", "clearcut", regex=False)

print(wholedb.loc[wholedb['siteid'].isin(group3), 'treatment'].unique())

# Modify the treatment column by appending "_slash" when pointtype is "slash"
valid_rows = (wholedb['siteid'].isin(group3)) & (wholedb['pointtype'] == "slash") & (~wholedb['pointtype'].isna())
wholedb.loc[valid_rows, 'treatment'] = wholedb.loc[valid_rows, 'treatment'] + "_slash"

print(wholedb.loc[wholedb['siteid'].isin(group3), 'treatment'].unique())

# treatments for burning group (group 4)
print(wholedb.loc[wholedb['siteid'].isin(group4), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group4), 'pointtype'].unique())

mask_control_4 = (wholedb['siteid'].isin(group4)) & (wholedb['subsiteid'] == "Control") & (~wholedb['subsiteid'].isna())
wholedb.loc[mask_control_4, 'control'] = True

wholedb.loc[wholedb['siteid'].isin(group4), 'treatment'] = wholedb.loc[wholedb['siteid'].isin(group4), 'subsiteid']

# String replacements for group 4
group4_mask = wholedb['siteid'].isin(group4)
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("Low thinning", "low-thinning")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("High thinning", "high-thinning")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("Low thinning + prescribed burning", "low-thinning+burning")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("High thinning + prescribed burning", "high-thinning+burning")

print(wholedb.loc[wholedb['siteid'].isin(group4), 'treatment'].unique())

# treatments for N fert group (group 5)
print(wholedb.loc[wholedb['siteid'].isin(group5), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group5), 'pointtype'].unique())

wholedb.loc[wholedb['siteid'].isin(group5), 'treatment'] = wholedb.loc[wholedb['siteid'].isin(group5), 'subsiteid']

print(wholedb.loc[(wholedb['siteid'].isin(group5)) & (wholedb['pointtype'] == "Trenched"), 'Trenched'])
print(wholedb.loc[(wholedb['siteid'].isin(group5)) & (wholedb['pointtype'] == "Trenched, without fabric lid"), 'Trenched'])

mask_control_5 = (wholedb['siteid'].isin(group5)) & (wholedb['subsiteid'] == "Control") & (~wholedb['subsiteid'].isna())
wholedb.loc[mask_control_5, 'control'] = True

# String replacements for group 5
group5_mask = wholedb['siteid'].isin(group5)
wholedb.loc[group5_mask, 'treatment'] = wholedb.loc[group5_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group5_mask, 'treatment'] = wholedb.loc[group5_mask, 'treatment'].str.replace("Nitrogen fertilization", "nitrogen")

print(wholedb.loc[wholedb['siteid'].isin(group5), 'treatment'].unique())

# treatments for peat group (group 6)
print(wholedb.loc[wholedb['siteid'].isin(group6), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group6), 'pointtype'].unique())

wholedb.loc[wholedb['siteid'].isin(group6), 'treatment'] = wholedb.loc[wholedb['siteid'].isin(group6), 'subsiteid']

mask_control_6 = (wholedb['siteid'].isin(group6)) & (wholedb['subsiteid'] == "Control") & (~wholedb['subsiteid'].isna())
wholedb.loc[mask_control_6, 'control'] = True

# String replacements for group 6
group6_mask = wholedb['siteid'].isin(group6)
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Continues cover forestry", "CCF")
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Clearcut", "clearcut")
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Ditch", "ditch")

print(wholedb.loc[wholedb['siteid'].isin(group6), 'treatment'].unique())

# treatments for ground vegetation group (group 7)
print(wholedb.loc[wholedb['siteid'].isin(group7), 'subsiteid'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group7), 'pointtype'].unique())

wholedb.loc[wholedb['siteid'].isin(group7), 'treatment'] = wholedb.loc[wholedb['siteid'].isin(group7), 'subsiteid']

# String replacements for group 7
group7_mask = wholedb['siteid'].isin(group7)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass\\+shrubs/with understorey", "grass+shrubs_with_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass\\+shrubs/no understorey", "grass+shrubs_no_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass/with understorey", "grass_with_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass/no understorey", "grass_no_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("clear_cut", "clearcut", regex=False)

print(wholedb.loc[wholedb['siteid'].isin(group7), 'treatment'].unique())
print(wholedb.loc[wholedb['siteid'].isin(group7), 'Trenched'].unique())

wholedb['treatment'] = wholedb['treatment'].astype('category')

# select only CO2 values
wholedb_co2 = wholedb[wholedb['gas'] == "co2"].copy()
wholedb_co2['soil_temp_5cm'] = pd.to_numeric(wholedb_co2['soil_temp_5cm'], errors='coerce')

# select only CH4 values
wholedb_ch4 = wholedb[wholedb['gas'] == "ch4"].copy()
wholedb_ch4['soil_temp_5cm'] = pd.to_numeric(wholedb_ch4['soil_temp_5cm'], errors='coerce')

# coalesce, merging personal and autotrim fluxes
wholedb_co2['merged_flux'] = wholedb_co2['autotrim_flux'].fillna(wholedb_co2['personal_flux'])
wholedb_ch4['merged_flux'] = wholedb_ch4['autotrim_flux'].fillna(wholedb_ch4['personal_flux'])



