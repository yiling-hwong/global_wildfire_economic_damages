# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append('../')
import utils

import warnings
warnings.filterwarnings(action='ignore')

"""
PARAMETERS
"""
base_path = f"../../data"
input_file_ssp = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"
input_file_hist = f"{base_path}/historical/TRAINING_DATA.csv"
beta_coef_file = f"{base_path}/historical/beta_coef.csv"
y_hist_mean_std_file = f"{base_path}/historical/mean_SD_X_y.csv"
model_file = f"{base_path}/historical/model.pickle"

y_var = "damage_gdp_weighted"
limit_forest_pct = True
forest_threshold_pct = 0.05
years = [2030,2040,2050,2060,2070]
ssps = ["ssp1","ssp2","ssp3"]
regions = ["AFR","APC","DEV","EEA","LAM","MEA"]

"""
LOAD DATA
"""
#HIST
df_hist = pd.read_csv(input_file_hist)
df_hist = df_hist[df_hist['region_ar6_6'].isin(regions)]
df_hist = df_hist.dropna(subset=[y_var])
cntry_hist = df_hist['iso'].unique()

#SSP
df_ssp = pd.read_csv(input_file_ssp)
df_ssp = df_ssp[df_ssp['esm'] != 'MM']

if limit_forest_pct == True:
    df_ssp = df_ssp[df_ssp['forest_pct'] >= forest_threshold_pct]

df_ssp = df_ssp[df_ssp['year'].isin(years)]
df_ssp = df_ssp[df_ssp['region_ar6_6'].isin(regions)]
df_ssp = df_ssp[df_ssp['ssp'].isin(ssps)]
df_ssp = df_ssp.dropna(subset=[y_var])
df_ssp = df_ssp[df_ssp['iso'].isin(cntry_hist)]
cntry_ssp = df_ssp['iso'].unique()

#GET COMMON COUNTRIES
df_hist = df_hist[df_hist['iso'].isin(cntry_ssp)]
num_cntry_hist = df_hist['iso'].nunique()
num_cntry_ssp = df_ssp['iso'].nunique()
print("NUM countries HIST and SSP:", num_cntry_hist, num_cntry_ssp)

# CREATE ADDITIONAL COLUMNS IN DF_SSP WITH HISTORICAL VALUES FOR LOG_HAZARD_STD, LOG_EXP_STD, LOG_VULN_STD
df_ssp = df_ssp.merge(df_hist[['iso', 'log_vpd_std']], on='iso', how='left', suffixes=('', '_new'))
df_ssp.rename(columns={'log_vpd_std_new': 'log_vpd_std_hist'}, inplace=True)

df_ssp = df_ssp.merge(df_hist[['iso', 'log_pdforest_std']], on='iso', how='left', suffixes=('', '_new'))
df_ssp.rename(columns={'log_pdforest_std_new': 'log_pdforest_std_hist'}, inplace=True)

df_ssp = df_ssp.merge(df_hist[['iso', 'log_hvi_std']], on='iso', how='left', suffixes=('', '_new'))
df_ssp.rename(columns={'log_hvi_std_new': 'log_hvi_std_hist'}, inplace=True)

"""
GET FITTED MODEL PARAMETERS AND LOAD MODEL
"""
df_beta_coef = pd.read_csv(beta_coef_file)
beta_haz = df_beta_coef['log_vpd'].values
beta_exp = df_beta_coef['log_pdforest'].values
beta_vuln = df_beta_coef['log_hvi'].values

df_y_hist_mean_std = pd.read_csv(y_hist_mean_std_file)
mean_log_haz = df_y_hist_mean_std['mean_log_vpd'].values
mean_log_exp = df_y_hist_mean_std['mean_log_pdforest'].values
mean_log_vuln = df_y_hist_mean_std['mean_log_hvi'].values
std_log_haz = df_y_hist_mean_std['std_log_vpd'].values
std_log_exp = df_y_hist_mean_std['std_log_pdforest'].values
std_log_vuln = df_y_hist_mean_std['std_log_hvi'].values

mean_y_hist = df_y_hist_mean_std['mean_y_hist'].values
std_y_hist = df_y_hist_mean_std['std_y_hist'].values

"""
PLOT
"""
import seaborn as sns
sns.set_theme(style="whitegrid")

def calculate_95_ci(data):
    """Calculate 95% confidence interval using t-distribution across ESMs"""
    # First average across countries for each ESM
    esm_means = data.groupby('esm')['value'].mean()
    
    # Calculate mean across ESMs
    mean = esm_means.mean()
    
    if len(esm_means) > 1:
        # Calculate standard deviation across ESMs
        std = esm_means.std(ddof=1)
        n = len(esm_means)  # number of ESMs
        sem = std / np.sqrt(n)
        
        # Calculate 95% CI width
        ci = stats.t.interval(confidence=0.95, df=n-1, loc=0, scale=sem)[1]
    else:
        ci = 0
    return mean, ci

fig = plt.figure(figsize=(13, 3))

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 12
legend_fontsize = 12

num_row = 1
num_col = 3
num_plots = num_row * num_col

colors = ["dimgrey", "mediumseagreen", "royalblue", "tomato"]
labels_alphabets = utils.generate_alphabet_list(4, option="lower")
ssp_labels = utils.get_ssp_labels(ssps)

# First collect all y values and error bars to find global min and max
all_y_values = []
all_yerr_values = []

for ssp in ssps:
    y_all = []
    y_haz = []
    y_exp = []
    y_vuln = []
    yerr_all = []
    yerr_haz = []
    yerr_exp = []
    yerr_vuln = []
    
    for year in years:
        # Get data for this SSP and year
        df_year = df_ssp[(df_ssp['ssp'] == ssp) & (df_ssp['year'] == year)]
        
        # Calculate predictions for each case
        # ALL - using all future values
        X_all = np.column_stack([
            df_year['log_vpd_std'] * std_y_hist,
            df_year['log_pdforest_std'] * std_y_hist,
            df_year['log_hvi_std'] * std_y_hist
        ])
        log_y_all = (X_all @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_all_unlog = pd.DataFrame({
            'value': np.exp(log_y_all).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        # Calculate mean and 95% CI
        mean_all, ci_all = calculate_95_ci(y_all_unlog)
        y_all.append(mean_all)
        yerr_all.append(ci_all)
        
        # HAZARD - only future VPD
        X_haz = np.column_stack([
            df_year['log_vpd_std'] * std_y_hist,
            df_year['log_pdforest_std_hist'] * std_y_hist,
            df_year['log_hvi_std_hist'] * std_y_hist
        ])
        log_y_haz = (X_haz @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_haz_unlog = pd.DataFrame({
            'value': np.exp(log_y_haz).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        mean_haz, ci_haz = calculate_95_ci(y_haz_unlog)
        y_haz.append(mean_haz)
        yerr_haz.append(ci_haz)
        
        # EXPOSURE - only future population density
        X_exp = np.column_stack([
            df_year['log_vpd_std_hist'] * std_y_hist,
            df_year['log_pdforest_std'] * std_y_hist,
            df_year['log_hvi_std_hist'] * std_y_hist
        ])
        log_y_exp = (X_exp @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_exp_unlog = pd.DataFrame({
            'value': np.exp(log_y_exp).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        mean_exp, ci_exp = calculate_95_ci(y_exp_unlog)
        y_exp.append(mean_exp)
        yerr_exp.append(ci_exp)
        
        # VULNERABILITY - only future HVI
        X_vuln = np.column_stack([
            df_year['log_vpd_std_hist'] * std_y_hist,
            df_year['log_pdforest_std_hist'] * std_y_hist,
            df_year['log_hvi_std'] * std_y_hist
        ])
        log_y_vuln = (X_vuln @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_vuln_unlog = pd.DataFrame({
            'value': np.exp(log_y_vuln).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        mean_vuln, ci_vuln = calculate_95_ci(y_vuln_unlog)
        y_vuln.append(mean_vuln)
        yerr_vuln.append(ci_vuln)
    
    all_y_values.extend([y_all, y_haz, y_exp, y_vuln])
    all_yerr_values.extend([yerr_all, yerr_haz, yerr_exp, yerr_vuln])

# Calculate global min and max
y_values_array = np.array(all_y_values)
yerr_values_array = np.array(all_yerr_values)

# Fixed limits for better symmetry
global_min = 0.2  # Set a fixed minimum that makes the plot look symmetrical
global_max = 1.8   # Fixed maximum as requested

for index, ssp in enumerate(ssps):
    ax1 = fig.add_subplot(num_row, num_col, index + 1)
    title = f"({labels_alphabets[index]}) {ssp_labels[index]}"
    
    y_all = []
    y_haz = []
    y_exp = []
    y_vuln = []
    yerr_all = []
    yerr_haz = []
    yerr_exp = []
    yerr_vuln = []
    
    for year in years:
        # Get data for this SSP and year
        df_year = df_ssp[(df_ssp['ssp'] == ssp) & (df_ssp['year'] == year)]
        
        # Calculate predictions for each case
        # ALL - using all future values
        X_all = np.column_stack([
            df_year['log_vpd_std'] * std_y_hist,
            df_year['log_pdforest_std'] * std_y_hist,
            df_year['log_hvi_std'] * std_y_hist
        ])
        log_y_all = (X_all @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_all_unlog = pd.DataFrame({
            'value': np.exp(log_y_all).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        # Calculate mean and 95% CI
        mean_all, ci_all = calculate_95_ci(y_all_unlog)
        y_all.append(mean_all)
        yerr_all.append(ci_all)
        
        # HAZARD - only future VPD
        X_haz = np.column_stack([
            df_year['log_vpd_std'] * std_y_hist,
            df_year['log_pdforest_std_hist'] * std_y_hist,
            df_year['log_hvi_std_hist'] * std_y_hist
        ])
        log_y_haz = (X_haz @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_haz_unlog = pd.DataFrame({
            'value': np.exp(log_y_haz).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        mean_haz, ci_haz = calculate_95_ci(y_haz_unlog)
        y_haz.append(mean_haz)
        yerr_haz.append(ci_haz)
        
        # EXPOSURE - only future population density
        X_exp = np.column_stack([
            df_year['log_vpd_std_hist'] * std_y_hist,
            df_year['log_pdforest_std'] * std_y_hist,
            df_year['log_hvi_std_hist'] * std_y_hist
        ])
        log_y_exp = (X_exp @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_exp_unlog = pd.DataFrame({
            'value': np.exp(log_y_exp).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        mean_exp, ci_exp = calculate_95_ci(y_exp_unlog)
        y_exp.append(mean_exp)
        yerr_exp.append(ci_exp)
        
        # VULNERABILITY - only future HVI
        X_vuln = np.column_stack([
            df_year['log_vpd_std_hist'] * std_y_hist,
            df_year['log_pdforest_std_hist'] * std_y_hist,
            df_year['log_hvi_std'] * std_y_hist
        ])
        log_y_vuln = (X_vuln @ np.array([beta_haz, beta_exp, beta_vuln])) + mean_y_hist
        y_vuln_unlog = pd.DataFrame({
            'value': np.exp(log_y_vuln).flatten(),
            'esm': df_year['esm'].tolist(),
            'iso': df_year['iso'].tolist()
        })
        
        mean_vuln, ci_vuln = calculate_95_ci(y_vuln_unlog)
        y_vuln.append(mean_vuln)
        yerr_vuln.append(ci_vuln)
    
    # Plot lines with error bars
    offset = 1.0  # Offset for separating points around each year
    x_offsets = [-offset/2, -offset/6, offset/6, offset/2]  # Distribute points around the year
    
    # Plot with error bars
    ax1.errorbar(np.array(years) + x_offsets[0], y_all, yerr=np.array(yerr_all), 
                color=colors[0], linestyle="dotted", marker='o', markersize=4, 
                capsize=2, capthick=0.8, elinewidth=0.8, label='_nolegend_',alpha=0.8)
    
    ax1.errorbar(np.array(years) + x_offsets[1], y_haz, yerr=np.array(yerr_haz),
                color=colors[1], marker='o', markersize=4,
                capsize=2, capthick=0.8, elinewidth=0.8, label='_nolegend_',alpha=0.8)
    
    ax1.errorbar(np.array(years) + x_offsets[2], y_exp, yerr=np.array(yerr_exp),
                color=colors[2], marker='o', markersize=4,
                capsize=2, capthick=0.8, elinewidth=0.8, label='_nolegend_',alpha=0.8)
    
    ax1.errorbar(np.array(years) + x_offsets[3], y_vuln, yerr=np.array(yerr_vuln),
                color=colors[3], marker='o', markersize=4,
                capsize=2, capthick=0.8, elinewidth=0.8, label='_nolegend_',alpha=0.8)
    
    # Add connecting lines
    ax1.plot(np.array(years) + x_offsets[0], y_all, color=colors[0], linestyle="dotted", label='_nolegend_')
    ax1.plot(np.array(years) + x_offsets[1], y_haz, color=colors[1], label='_nolegend_')
    ax1.plot(np.array(years) + x_offsets[2], y_exp, color=colors[2], label='_nolegend_')
    ax1.plot(np.array(years) + x_offsets[3], y_vuln, color=colors[3], label='_nolegend_')
    
    # Add legend entries
    ax1.plot([], [], color=colors[0], linestyle="dotted", marker='o', markersize=4, label="All")
    ax1.plot([], [], color=colors[1], marker='o', markersize=4, label=r"VPD$_{fs}$")
    ax1.plot([], [], color=colors[2], marker='o', markersize=4, label=r"PD$_{forest}$")
    ax1.plot([], [], color=colors[3], marker='o', markersize=4, label=r"HVI")
    
    ax1.set_title(title, fontsize=title_fontsize, weight="bold")
    ax1.set_xlabel("Year", fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.grid(True, color="gainsboro")
    
    # Set x-ticks to show only the actual years
    ax1.set_xticks(years)
    ax1.set_xticklabels(years)
    
    # Add horizontal line at y=0
    #ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set the same y-limits for all subplots
    ax1.set_ylim(global_min, global_max)
    
    if index == 0:
        ax1.set_ylabel("%GDP", fontsize=label_fontsize)
    if index == 2:
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)

fig.subplots_adjust(left=0.06,
                   bottom=0.2,
                   right=0.89,
                   top=0.88,
                   wspace=0.18)

plt.show()
