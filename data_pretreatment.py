# ==============
# IMPORT MODULES
# ==============

import pandas as pd
import numpy as np
try:
  import pyarrow
  pyarrow_available = True
except ModuleNotFoundError:
  pyarrow_available = False
if pyarrow_available:
  print("PyArrow is available. You can use it.")
else:
  print("PyArrow is not installed. Skipping related steps.")

# =========
# LOAD DATA
# =========

wholedb = pd.read_csv("../../Taavi/All_sites.csv", low_memory = False)  # if you have pyarrow installed
print(wholedb['siteid'].unique())

# ================
# GENERAL CLEANING
# ================

# Fix site names
wholedb.loc[wholedb['siteid'] == "Zwolse bos", 'siteid'] = "Zwolse Bos"
wholedb.loc[wholedb['siteid'] == "Kroondomein ", 'siteid'] = "Kroondomein"

# Assigning the countries
wholedb['country'] = pd.NA

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

# =======================
# TRENCHED BOOLEAN COLUMN
# =======================

# Trenched yes no or NA
print(wholedb['pointtype'].unique())
wholedb['Trenched'] = pd.Series(pd.NA, index=wholedb.index, dtype='boolean')

# Making uniform the Trenched sites
trenched_types = ["Trenched", "Trenched, without fabric lid", "Trenched, with fabric lid"]
wholedb.loc[wholedb['pointtype'].isin(trenched_types), 'Trenched'] = True
print(wholedb['Trenched'].value_counts(dropna=False))

# Sites from WP4, in some cases they specified the trenching in the subsiteid
print(wholedb['subsiteid'].unique())
wholedb.loc[wholedb['subsiteid'].str.contains("trenching", case=False, na=False), 'Trenched'] = True
print(wholedb.loc[wholedb['subsiteid'] == "No thinning / Trenching", 'Trenched'])  # checking

wholedb.loc[wholedb['siteid'] == "DumbravitaTrench", 'Trenched'] = True

# Set to FALSE the untrenched points of sites with trenching
siteids_trenched = wholedb.loc[wholedb['Trenched'] == True, 'siteid'].unique()
print(siteids_trenched)

wholedb['Trenched'].unique()
for site in siteids_trenched:
    mask = (wholedb['Trenched'].isna()) & (wholedb['siteid'] == site)
    wholedb.loc[mask, 'Trenched'] = False

# Dobroc has an error, all are trenched, info is in file ID ('point')
dobroc_subset = wholedb[wholedb['siteid'] == "Dobroc"]
dobroc_mask = wholedb['siteid'] == "Dobroc"

wholedb.loc[dobroc_mask, 'Trenched'] = np.where(
    dobroc_subset['point'].str.contains("C", na=False), False,
    np.where(dobroc_subset['point'].str.contains("T", na=False), True, pd.NA)
)

wholedb['Trenched'].unique()
# Convert to nullable boolean
wholedb['Trenched'] = wholedb['Trenched'].astype('boolean')
print(wholedb['Trenched'].value_counts(dropna=False))

print(wholedb['id'].isna().sum())  # equivalent to which(is.na())



# =========================
# GROUP SITES BY TREATMENTS
# =========================

group1 = ["Dobroc", "Kelheim-Parsberg", "Kranzberg-Freising", "Wasserburg-Maitenbeth"]  # mixed stands
group2 = ["Kroondomein", "Zwolse Bos", "NP Hoge Veluwe", "Saint Mitre"]  # WP4 thinning
group3 = ["Dumbravita", "Gamiz", "St Christol"]  # WP5 thinning
group4 = ["Llobera", "Secanella"]  # forest fires
group5 = ["Karstula75", "Karstula76"]  # N treatments, Lithuania goes here
group6 = ["Ränskälänkorpi"]  # peat treatment
group7 = ["DumbravitaTrench"]  # WP5 trenching, ground vegetation

wholedb['group'] = pd.NA

wholedb.loc[wholedb['siteid'].isin(group1), 'group'] = "mixture"
wholedb.loc[wholedb['siteid'].isin(group2), 'group'] = "thinning"
wholedb.loc[wholedb['siteid'].isin(group3), 'group'] = "slash"
wholedb.loc[wholedb['siteid'].isin(group4), 'group'] = "burning"
wholedb.loc[wholedb['siteid'].isin(group5), 'group'] = "N_treatment"
wholedb.loc[wholedb['siteid'].isin(group6), 'group'] = "peat"
wholedb.loc[wholedb['siteid'].isin(group7), 'group'] = "ground_vegetation"

# ===========================
# TREATMENT & CONTROL COLUMNS
# ===========================

# create two columns that tell what the treatment is and one that tells wether it belongs to control of that treatment or not
wholedb['treatment'] = pd.NA
wholedb['control'] = pd.Series(pd.NA, index=wholedb.index, dtype='boolean')


# =======
# GROUP 1
# =======

# treatments for mixture group (group 1)
group1_mask = wholedb['siteid'].isin(group1)

print(wholedb.loc[group1_mask, 'subsiteid'].unique())
print(wholedb.loc[group1_mask, 'pointtype'].unique())

# assign controls and treatments
wholedb.loc[(group1_mask) & (wholedb['subsiteid'] == "DP-MC"), 'control'] = True
wholedb.loc[(group1_mask) & (wholedb['subsiteid'] == "Spruce"), 'control'] = True
wholedb.loc[group1_mask, 'treatment'] = "mixture"
wholedb.loc[(group1_mask) & (wholedb['control'] == True), 'treatment'] = "control"

# check
print(wholedb.loc[group1_mask, 'treatment'].isna().sum())
print(len(wholedb.loc[group1_mask, 'treatment']))

# checking that the Trenched boolean is correctly assigned
print(wholedb.loc[(group1_mask) & (wholedb['Trenched'] == True), 'pointtype'].unique())
print(wholedb.loc[(wholedb['siteid'].isin(["Kelheim-Parsberg", "Kranzberg-Freising", "Wasserburg-Maitenbeth"])) & (wholedb['Trenched'] == False), 'pointtype'].unique())
print(wholedb.loc[(wholedb['siteid'] == 'Dobroc'), 'pointtype'].unique())
print(wholedb.loc[group1_mask, 'treatment'].unique())

# =======
# GROUP 2
# =======

## treatments for thinning group (group 2)
group2_mask = wholedb['siteid'].isin(group2)

print(wholedb.loc[group2_mask, 'subsiteid'].unique())
print(wholedb.loc[group2_mask, 'pointtype'].unique())

wholedb.loc[group2_mask, 'treatment'] = wholedb.loc[group2_mask, 'subsiteid']

# finish setting the trenched boolean
mask_trenching = (group2_mask) & (wholedb['treatment'].str.contains("Trenching", na=False))
wholedb.loc[mask_trenching, 'Trenched'] = True

mask_control = (group2_mask) & (wholedb['treatment'] == "Control") & (~wholedb['treatment'].isna())
wholedb.loc[mask_control, 'control'] = True

# String replacements for group 2
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

print(wholedb.loc[group2_mask, 'treatment'].unique())

# Check if for St Mitre's treatment "no thinning" is set as control
print(wholedb.loc[(wholedb['siteid'] == 'Saint Mitre') & (wholedb['treatment'] == 'no-thinning'), 'control'].unique())
print(wholedb.loc[(wholedb['siteid'] == 'Saint Mitre') & (wholedb['treatment'] != 'no-thinning'), 'control'].unique())
# Double check the Netherlands sites for same
print(wholedb.loc[(wholedb['siteid'].isin(["Kroondomein", "Zwolse Bos", "NP Hoge Veluwe"])) & (wholedb['treatment'] == 'control'), 'control'].unique())
print(wholedb.loc[(wholedb['siteid'].isin(["Kroondomein", "Zwolse Bos", "NP Hoge Veluwe"])) & (wholedb['treatment'] != 'control'), 'control'].unique())
# let's include no-thining treatment as control
wholedb.loc[(group2_mask) & (wholedb['treatment'] == 'no-thinning'), 'control'] = True

# =======
# GROUP 3
# =======

# treatments for slash group (group 3)
group3_mask = wholedb['siteid'].isin(group3)

print(wholedb.loc[group3_mask, 'subsiteid'].unique())
print(wholedb.loc[group3_mask, 'pointtype'].unique())
print(wholedb.loc[group3_mask, 'siteid'].unique())

wholedb.loc[group3_mask, 'subsiteid'] = wholedb.loc[group3_mask, 'subsiteid'].str.replace("-", "", regex=False)

mask_control_3 = (group3_mask) & (wholedb['pointtype'] == "control") & (~wholedb['pointtype'].isna())
wholedb.loc[mask_control_3, 'control'] = True

print(wholedb.loc[group3_mask, 'subsiteid'].unique())

wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'subsiteid']

# This is relevant for DumbravitaTrench

# String replacements for group 3
#group3_mask = wholedb['siteid'].isin(group3)
#wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass\\+shrubs/with understorey", "grass+shrubs_no_understorey", regex=True)
#wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass\\+shrubs/no understorey", "grass+shrubs_no_understorey", regex=True)
#wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass/with understorey", "grass", regex=True)
#wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("grass/no understorey", "grass_no_understorey", regex=True)
#wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("clear_cut", "clearcut", regex=False)

print(wholedb.loc[group3_mask, 'treatment'].unique())
wholedb.loc[group3_mask, 'treatment'] = wholedb.loc[group3_mask, 'treatment'].str.replace("clear_cut", "clearcut", regex=False)

# Modify the treatment column by appending "_slash" when pointtype is "slash"
valid_rows = (group3_mask) & (wholedb['pointtype'] == "slash") & (~wholedb['pointtype'].isna())
wholedb.loc[valid_rows, 'treatment'] = wholedb.loc[valid_rows, 'treatment'] + "_slash"

print(wholedb.loc[group3_mask, 'treatment'].unique())
print(wholedb.loc[(group3_mask) & (wholedb['treatment'] == 'control_slash'), 'control'])
print(wholedb.loc[(group3_mask) & (wholedb['treatment'] == 'control_slash'), 'siteid'])
# There is two rows in site Gamiz with control in subsite but also slash treatment in pointtype. Let's remove these contradictory rows.
wholedb = wholedb[~((group3_mask) & (wholedb['treatment'] == 'control_slash'))]

# =======
# GROUP 4
# =======

# treatments for burning group (group 4)
group4_mask = wholedb['siteid'].isin(group4)
print(wholedb.loc[group4_mask, 'subsiteid'].unique())
print(wholedb.loc[group4_mask, 'pointtype'].unique())

mask_control_4 = (group4_mask) & (wholedb['subsiteid'] == "Control") & (~wholedb['subsiteid'].isna())
wholedb.loc[mask_control_4, 'control'] = True

wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'subsiteid']

# String replacements for group 4
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("Low thinning", "low-thinning")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("High-thinning", "high-thinning")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("low-thinning + prescribed burning", "low-thinning+burning")
wholedb.loc[group4_mask, 'treatment'] = wholedb.loc[group4_mask, 'treatment'].str.replace("High thinning + prescribed burning", "high-thinning+burning")

print(wholedb.loc[group4_mask, 'treatment'].unique())

# =======
# GROUP 5
# =======

# treatments for N fert group (group 5)
group5_mask = wholedb['siteid'].isin(group5)

print(wholedb.loc[group5_mask, 'subsiteid'].unique())
print(wholedb.loc[group5_mask, 'pointtype'].unique())

wholedb.loc[group5_mask, 'treatment'] = wholedb.loc[group5_mask, 'subsiteid']

print(wholedb.loc[(group5_mask) & (wholedb['pointtype'] == "Trenched"), 'Trenched'].value_counts())
print(wholedb.loc[(group5_mask) & (wholedb['pointtype'] == "Trenched, without fabric lid"), 'Trenched'].value_counts())
print(wholedb.loc[(group5_mask) & (wholedb['pointtype'] == "Untrenched"), 'Trenched'].value_counts())

mask_control_5 = (group5_mask) & (wholedb['subsiteid'] == "Control") & (~wholedb['subsiteid'].isna())
wholedb.loc[mask_control_5, 'control'] = True

# String replacements for group 5

wholedb.loc[group5_mask, 'treatment'] = wholedb.loc[group5_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group5_mask, 'treatment'] = wholedb.loc[group5_mask, 'treatment'].str.replace("Nitrogen fertilization", "nitrogen")

print(wholedb.loc[group5_mask, 'treatment'].unique())

# =======
# GROUP 6
# =======

# treatments for peat group (group 6)
group6_mask = wholedb['siteid'].isin(group6)
print(wholedb.loc[group6_mask, 'subsiteid'].unique())
print(wholedb.loc[group6_mask, 'pointtype'].unique())

wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'subsiteid']

mask_control_6 = (group6_mask) & (wholedb['subsiteid'] == "Control") & (~wholedb['subsiteid'].isna())
wholedb.loc[mask_control_6, 'control'] = True

# String replacements for group 6
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Control", "control")
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Continues cover forestry", "CCF")
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Clearcut", "clearcut")
wholedb.loc[group6_mask, 'treatment'] = wholedb.loc[group6_mask, 'treatment'].str.replace("Ditch", "ditch")

print(wholedb.loc[group6_mask, 'treatment'].unique())

# =======
# GROUP 7
# =======

# treatments for ground vegetation group (group 7)
group7_mask = wholedb['siteid'].isin(group7)

print(wholedb.loc[group7_mask, 'subsiteid'].unique())
print(wholedb.loc[group7_mask, 'pointtype'].unique())

wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'subsiteid']

# String replacements for group 7
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass\\+shrubs/with understorey", "grass+shrubs_with_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass\\+shrubs/no understorey", "grass+shrubs_no_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass/with understorey", "grass_with_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("grass/no understorey", "grass_no_understorey", regex=True)
wholedb.loc[group7_mask, 'treatment'] = wholedb.loc[group7_mask, 'treatment'].str.replace("clear_cut", "clearcut", regex=False)

print(wholedb.loc[group7_mask, 'treatment'].unique())
print(wholedb.loc[group7_mask, 'Trenched'].unique())

# =====================
# Check Trenched column
# =====================
print(wholedb.loc[wholedb['Trenched'].isna(), 'siteid'].unique())
print(wholedb.loc[wholedb['Trenched'].isna(), 'siteid'].value_counts())
print(wholedb.loc[wholedb['Trenched'] == True, 'siteid'].value_counts())
print(wholedb.loc[wholedb['Trenched'] == False, 'siteid'].value_counts())
# NA values are in Dumbravita(nonTrenched), Gamiz and St Christol
# These locations didnt have trenching at all so we want to set them as False
wholedb.loc[:, 'Trenched'] = wholedb['Trenched'].fillna(False)


# =============================
# MERGE Dumbravita and Karstula
# =============================

# Replace DumbravitaTrench with Dumbravita
wholedb.loc[:, 'siteid'] = wholedb['siteid'].str.replace('DumbravitaTrench', 'Dumbravita')
wholedb.loc[:, 'siteid'] = wholedb['siteid'].str.replace(r'Karstula76|Karstula75', 'Karstula', regex=True)

# ======================
# TREATMENT CONFIRMATION
# ======================
wholedb.loc[:, 'treatment'] = wholedb['treatment'].astype('category')
print(wholedb['treatment'].value_counts())

treatment_site_map = {}
unique_treatments = wholedb['treatment'].unique()
for treatment in unique_treatments:
  sites = wholedb.loc[wholedb['treatment'] == treatment, 'siteid'].unique()
  treatment_site_map[treatment] = sorted(sites)
  
for treatment, sites in treatment_site_map.items():
  print(f"Treatment: {treatment}")
  print(f"Sites: {sites}")
  print("-" * 40)

# ================
# FILTERING BY GAS
# ================

# select only CO2 values
wholedb_co2 = wholedb[wholedb['gas'] == "co2"].copy()
wholedb_co2['soil_temp_5cm'] = pd.to_numeric(wholedb_co2['soil_temp_5cm'], errors='coerce')

# select only CH4 values
wholedb_ch4 = wholedb[wholedb['gas'] == "ch4"].copy()
wholedb_ch4['soil_temp_5cm'] = pd.to_numeric(wholedb_ch4['soil_temp_5cm'], errors='coerce')

# coalesce, merging personal and autotrim fluxes
wholedb_co2['merged_flux'] = wholedb_co2['autotrim_flux'].fillna(wholedb_co2['personal_flux'])
wholedb_ch4['merged_flux'] = wholedb_ch4['autotrim_flux'].fillna(wholedb_ch4['personal_flux'])
