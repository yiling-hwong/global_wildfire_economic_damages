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
import scipy.stats as stats

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

y_var = "damage_gdp_weighted"
years = [2030,2040,2050,2060,2070]
ssps = ["ssp1","ssp2","ssp3"]
regions = ["ldc","developing","developed"]

"""
LOAD DATA
"""
#HIST
df_hist = pd.read_csv(input_file_hist)
df_hist = df_hist[df_hist['region_dev'].isin(regions)]
df_hist = df_hist.dropna(subset=[y_var])
cntry_hist = df_hist['iso'].unique()

#SSP
df_ssp = pd.read_csv(input_file_ssp)
df_ssp = df_ssp[df_ssp['year'].isin(years)]
df_ssp = df_ssp[df_ssp['region_dev'].isin(regions)]
df_ssp = df_ssp[df_ssp['ssp'].isin(ssps)]
df_ssp = df_ssp.dropna(subset=[y_var])
df_ssp = df_ssp[df_ssp['iso'].isin(cntry_hist)]
cntry_ssp = df_ssp['iso'].unique()

#GET COMMON COUNTRIES
df_hist = df_hist[df_hist['iso'].isin(cntry_ssp)]
num_cntry_hist = df_hist['iso'].nunique()
num_cntry_ssp = df_ssp['iso'].nunique()
print ("NUM countries HIST and SSP:",num_cntry_hist,num_cntry_ssp)

# CREATE ADDITIONAL COLUMNS IN DF_SSP WITH HISTORICAL VALUES FOR LOG_HAZARD_STD, LOG_EXP_STD, LOG_VULN_STD
df_ssp = df_ssp.merge(df_hist[['iso', 'log_vpd_std']], on='iso', how='left', suffixes=('', '_new'))
df_ssp.rename(columns={'log_vpd_std_new': 'log_vpd_std_hist'}, inplace=True)

df_ssp = df_ssp.merge(df_hist[['iso', 'log_pdforest_std']], on='iso', how='left', suffixes=('', '_new'))
df_ssp.rename(columns={'log_pdforest_std_new': 'log_pdforest_std_hist'}, inplace=True)

df_ssp = df_ssp.merge(df_hist[['iso', 'log_hvi_std']], on='iso', how='left', suffixes=('', '_new'))
df_ssp.rename(columns={'log_hvi_std_new': 'log_hvi_std_hist'}, inplace=True)

"""
GET FITTED MODEL PARAMETERS (BETA COEF, MEAN, STD, etc)
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
GET SSP3 - SSP1 DIFFERENCE
"""

# Initialize arrays for storing results
y_all = []
y_haz = []
y_exp = []
y_vuln = []

y_all_dev = []
y_haz_dev = []
y_exp_dev = []
y_vuln_dev = []

y_all_ldc = []
y_haz_ldc = []
y_exp_ldc = []
y_vuln_ldc = []

error_all = []
error_haz = []
error_exp = []
error_vuln = []

error_all_dev = []
error_haz_dev = []
error_exp_dev = []
error_vuln_dev = []

error_all_ldc = []
error_haz_ldc = []
error_exp_ldc = []
error_vuln_ldc = []

def calculate_95ci(data_ssp3, data_ssp1):
    """Calculate 95% confidence interval across ESMs after averaging across countries"""
    # First average across countries for each ESM
    ssp3_by_esm = data_ssp3.groupby('esm')['value'].mean()
    ssp1_by_esm = data_ssp1.groupby('esm')['value'].mean()
    
    # Calculate differences for each ESM
    differences = ssp3_by_esm - ssp1_by_esm
    
    if len(differences) > 1:
        # Calculate standard error across ESMs
        std = differences.std(ddof=1)
        n = len(differences)  # number of ESMs
        sem = std / np.sqrt(n)
        
        # Calculate 95% CI using t-distribution
        t_value = stats.t.ppf(0.975, df=n-1)  # Two-tailed 95% CI
        ci = t_value * sem
    else:
        ci = 0
    
    return np.mean(differences), ci

for year in years:

    df_to_plot = df_ssp[df_ssp['year'] == year]

    df_ssp1 = df_to_plot[df_to_plot['ssp'] == "ssp1"]
    df_ssp2 = df_to_plot[df_to_plot['ssp'] == "ssp2"]
    df_ssp3 = df_to_plot[df_to_plot['ssp'] == "ssp3"]

    df_ssp1_dev = df_ssp1[df_ssp1['region_dev'] == "developed"]
    df_ssp2_dev = df_ssp2[df_ssp2['region_dev'] == "developed"]
    df_ssp3_dev = df_ssp3[df_ssp3['region_dev'] == "developed"]

    df_ssp1_ldc = df_ssp1[df_ssp1['region_dev'] == "ldc"]
    df_ssp2_ldc = df_ssp2[df_ssp2['region_dev'] == "ldc"]
    df_ssp3_ldc = df_ssp3[df_ssp3['region_dev'] == "ldc"]

    #########################--------------- ALL
    # Create DataFrames with values and ESM info
    ssp1_data = pd.DataFrame({
        'value': df_ssp1['damage_gdp_weighted'].values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp1['esm'].values
    })
    ssp3_data = pd.DataFrame({
        'value': df_ssp3['damage_gdp_weighted'].values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp3['esm'].values
    })
    mean_diff, ci = calculate_95ci(ssp3_data, ssp1_data)
    y_all.append(mean_diff)
    error_all.append(ci)

    #DEV
    ssp1_data_dev = pd.DataFrame({
        'value': df_ssp1_dev['damage_gdp_weighted'].values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_dev['esm'].values
    })
    ssp3_data_dev = pd.DataFrame({
        'value': df_ssp3_dev['damage_gdp_weighted'].values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_dev['esm'].values
    })
    mean_diff_dev, ci_dev = calculate_95ci(ssp3_data_dev, ssp1_data_dev)
    y_all_dev.append(mean_diff_dev)
    error_all_dev.append(ci_dev)

    #LDC
    ssp1_data_ldc = pd.DataFrame({
        'value': df_ssp1_ldc['damage_gdp_weighted'].values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_ldc['esm'].values
    })
    ssp3_data_ldc = pd.DataFrame({
        'value': df_ssp3_ldc['damage_gdp_weighted'].values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_ldc['esm'].values
    })
    mean_diff_ldc, ci_ldc = calculate_95ci(ssp3_data_ldc, ssp1_data_ldc)
    y_all_ldc.append(mean_diff_ldc)
    error_all_ldc.append(ci_ldc)

    #########################--------------- HAZ
    # Calculate log values first - vary VPD only
    log_y_haz_ssp1 = ((beta_haz * df_ssp1['log_vpd_std'] * std_y_hist)
                      + (beta_exp * df_ssp1['log_pdforest_std_hist'] * std_y_hist)
                      + (beta_vuln * df_ssp1['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    log_y_haz_ssp3 = ((beta_haz * df_ssp3['log_vpd_std'] * std_y_hist)
                      + (beta_exp * df_ssp3['log_pdforest_std_hist'] * std_y_hist)
                      + (beta_vuln * df_ssp3['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    # Calculate damages for each ESM
    ssp1_data_haz = pd.DataFrame({
        'value': np.exp(log_y_haz_ssp1).values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp1['esm'].values
    })
    ssp3_data_haz = pd.DataFrame({
        'value': np.exp(log_y_haz_ssp3).values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp3['esm'].values
    })
    mean_diff_haz, ci_haz = calculate_95ci(ssp3_data_haz, ssp1_data_haz)
    y_haz.append(mean_diff_haz)
    error_haz.append(ci_haz)

    #DEV
    log_y_haz_ssp1_dev = ((beta_haz * df_ssp1_dev['log_vpd_std'] * std_y_hist)
                          + (beta_exp * df_ssp1_dev['log_pdforest_std_hist'] * std_y_hist)
                          + (beta_vuln * df_ssp1_dev['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    log_y_haz_ssp3_dev = ((beta_haz * df_ssp3_dev['log_vpd_std'] * std_y_hist)
                          + (beta_exp * df_ssp3_dev['log_pdforest_std_hist'] * std_y_hist)
                          + (beta_vuln * df_ssp3_dev['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    ssp1_data_haz_dev = pd.DataFrame({
        'value': np.exp(log_y_haz_ssp1_dev).values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_dev['esm'].values
    })
    ssp3_data_haz_dev = pd.DataFrame({
        'value': np.exp(log_y_haz_ssp3_dev).values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_dev['esm'].values
    })
    mean_diff_haz_dev, ci_haz_dev = calculate_95ci(ssp3_data_haz_dev, ssp1_data_haz_dev)
    y_haz_dev.append(mean_diff_haz_dev)
    error_haz_dev.append(ci_haz_dev)

    #LDC
    log_y_haz_ssp1_ldc = ((beta_haz * df_ssp1_ldc['log_vpd_std'] * std_y_hist)
                          + (beta_exp * df_ssp1_ldc['log_pdforest_std_hist'] * std_y_hist)
                          + (beta_vuln * df_ssp1_ldc['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    log_y_haz_ssp3_ldc = ((beta_haz * df_ssp3_ldc['log_vpd_std'] * std_y_hist)
                          + (beta_exp * df_ssp3_ldc['log_pdforest_std_hist'] * std_y_hist)
                          + (beta_vuln * df_ssp3_ldc['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    ssp1_data_haz_ldc = pd.DataFrame({
        'value': np.exp(log_y_haz_ssp1_ldc).values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_ldc['esm'].values
    })
    ssp3_data_haz_ldc = pd.DataFrame({
        'value': np.exp(log_y_haz_ssp3_ldc).values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_ldc['esm'].values
    })
    mean_diff_haz_ldc, ci_haz_ldc = calculate_95ci(ssp3_data_haz_ldc, ssp1_data_haz_ldc)
    y_haz_ldc.append(mean_diff_haz_ldc)
    error_haz_ldc.append(ci_haz_ldc)

    #########################--------------- EXP
    # Calculate log values - vary population density only
    log_y_exp_ssp1 = ((beta_haz * df_ssp1['log_vpd_std_hist'] * std_y_hist)
                      + (beta_exp * df_ssp1['log_pdforest_std'] * std_y_hist)
                      + (beta_vuln * df_ssp1['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    log_y_exp_ssp3 = ((beta_haz * df_ssp3['log_vpd_std_hist'] * std_y_hist)
                      + (beta_exp * df_ssp3['log_pdforest_std'] * std_y_hist)
                      + (beta_vuln * df_ssp3['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    # Calculate damages for each ESM
    ssp1_data_exp = pd.DataFrame({
        'value': np.exp(log_y_exp_ssp1).values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp1['esm'].values
    })
    ssp3_data_exp = pd.DataFrame({
        'value': np.exp(log_y_exp_ssp3).values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp3['esm'].values
    })
    mean_diff_exp, ci_exp = calculate_95ci(ssp3_data_exp, ssp1_data_exp)
    y_exp.append(mean_diff_exp)
    error_exp.append(ci_exp)

    #DEV
    log_y_exp_ssp1_dev = ((beta_haz * df_ssp1_dev['log_vpd_std_hist'] * std_y_hist)
                          + (beta_exp * df_ssp1_dev['log_pdforest_std'] * std_y_hist)
                          + (beta_vuln * df_ssp1_dev['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    log_y_exp_ssp3_dev = ((beta_haz * df_ssp3_dev['log_vpd_std_hist'] * std_y_hist)
                          + (beta_exp * df_ssp3_dev['log_pdforest_std'] * std_y_hist)
                          + (beta_vuln * df_ssp3_dev['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    ssp1_data_exp_dev = pd.DataFrame({
        'value': np.exp(log_y_exp_ssp1_dev).values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_dev['esm'].values
    })
    ssp3_data_exp_dev = pd.DataFrame({
        'value': np.exp(log_y_exp_ssp3_dev).values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_dev['esm'].values
    })
    mean_diff_exp_dev, ci_exp_dev = calculate_95ci(ssp3_data_exp_dev, ssp1_data_exp_dev)
    y_exp_dev.append(mean_diff_exp_dev)
    error_exp_dev.append(ci_exp_dev)

    #LDC
    log_y_exp_ssp1_ldc = ((beta_haz * df_ssp1_ldc['log_vpd_std_hist'] * std_y_hist)
                          + (beta_exp * df_ssp1_ldc['log_pdforest_std'] * std_y_hist)
                          + (beta_vuln * df_ssp1_ldc['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    log_y_exp_ssp3_ldc = ((beta_haz * df_ssp3_ldc['log_vpd_std_hist'] * std_y_hist)
                          + (beta_exp * df_ssp3_ldc['log_pdforest_std'] * std_y_hist)
                          + (beta_vuln * df_ssp3_ldc['log_hvi_std_hist'] * std_y_hist) + mean_y_hist)

    ssp1_data_exp_ldc = pd.DataFrame({
        'value': np.exp(log_y_exp_ssp1_ldc).values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_ldc['esm'].values
    })
    ssp3_data_exp_ldc = pd.DataFrame({
        'value': np.exp(log_y_exp_ssp3_ldc).values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_ldc['esm'].values
    })
    mean_diff_exp_ldc, ci_exp_ldc = calculate_95ci(ssp3_data_exp_ldc, ssp1_data_exp_ldc)
    y_exp_ldc.append(mean_diff_exp_ldc)
    error_exp_ldc.append(ci_exp_ldc)

    #########################--------------- VULN
    # Calculate log values - vary HVI only
    log_y_vuln_ssp1 = ((beta_haz * df_ssp1['log_vpd_std_hist'] * std_y_hist)
                       + (beta_exp * df_ssp1['log_pdforest_std_hist'] * std_y_hist)
                       + (beta_vuln * df_ssp1['log_hvi_std'] * std_y_hist) + mean_y_hist)

    log_y_vuln_ssp3 = ((beta_haz * df_ssp3['log_vpd_std_hist'] * std_y_hist)
                       + (beta_exp * df_ssp3['log_pdforest_std_hist'] * std_y_hist)
                       + (beta_vuln * df_ssp3['log_hvi_std'] * std_y_hist) + mean_y_hist)

    # Calculate damages for each ESM
    ssp1_data_vuln = pd.DataFrame({
        'value': np.exp(log_y_vuln_ssp1).values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp1['esm'].values
    })
    ssp3_data_vuln = pd.DataFrame({
        'value': np.exp(log_y_vuln_ssp3).values * df_ssp2['gdp'].values / 100 / 1e9,
        'esm': df_ssp3['esm'].values
    })
    mean_diff_vuln, ci_vuln = calculate_95ci(ssp3_data_vuln, ssp1_data_vuln)
    y_vuln.append(mean_diff_vuln)
    error_vuln.append(ci_vuln)

    #DEV
    log_y_vuln_ssp1_dev = ((beta_haz * df_ssp1_dev['log_vpd_std_hist'] * std_y_hist)
                           + (beta_exp * df_ssp1_dev['log_pdforest_std_hist'] * std_y_hist)
                           + (beta_vuln * df_ssp1_dev['log_hvi_std'] * std_y_hist) + mean_y_hist)

    log_y_vuln_ssp3_dev = ((beta_haz * df_ssp3_dev['log_vpd_std_hist'] * std_y_hist)
                           + (beta_exp * df_ssp3_dev['log_pdforest_std_hist'] * std_y_hist)
                           + (beta_vuln * df_ssp3_dev['log_hvi_std'] * std_y_hist) + mean_y_hist)

    ssp1_data_vuln_dev = pd.DataFrame({
        'value': np.exp(log_y_vuln_ssp1_dev).values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_dev['esm'].values
    })
    ssp3_data_vuln_dev = pd.DataFrame({
        'value': np.exp(log_y_vuln_ssp3_dev).values * df_ssp2_dev['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_dev['esm'].values
    })
    mean_diff_vuln_dev, ci_vuln_dev = calculate_95ci(ssp3_data_vuln_dev, ssp1_data_vuln_dev)
    y_vuln_dev.append(mean_diff_vuln_dev)
    error_vuln_dev.append(ci_vuln_dev)

    #LDC
    log_y_vuln_ssp1_ldc = ((beta_haz * df_ssp1_ldc['log_vpd_std_hist'] * std_y_hist)
                           + (beta_exp * df_ssp1_ldc['log_pdforest_std_hist'] * std_y_hist)
                           + (beta_vuln * df_ssp1_ldc['log_hvi_std'] * std_y_hist) + mean_y_hist)

    log_y_vuln_ssp3_ldc = ((beta_haz * df_ssp3_ldc['log_vpd_std_hist'] * std_y_hist)
                           + (beta_exp * df_ssp3_ldc['log_pdforest_std_hist'] * std_y_hist)
                           + (beta_vuln * df_ssp3_ldc['log_hvi_std'] * std_y_hist) + mean_y_hist)

    ssp1_data_vuln_ldc = pd.DataFrame({
        'value': np.exp(log_y_vuln_ssp1_ldc).values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp1_ldc['esm'].values
    })
    ssp3_data_vuln_ldc = pd.DataFrame({
        'value': np.exp(log_y_vuln_ssp3_ldc).values * df_ssp2_ldc['gdp'].values / 100 / 1e9,
        'esm': df_ssp3_ldc['esm'].values
    })
    mean_diff_vuln_ldc, ci_vuln_ldc = calculate_95ci(ssp3_data_vuln_ldc, ssp1_data_vuln_ldc)
    y_vuln_ldc.append(mean_diff_vuln_ldc)
    error_vuln_ldc.append(ci_vuln_ldc)

# print (len(y_all),len(y_haz),len(y_exp),len(y_vuln))
# print (len(y_all_dev),len(y_haz_dev),len(y_exp_dev),len(y_vuln_dev))
# print (len(y_all_ldc),len(y_haz_ldc),len(y_exp_ldc),len(y_vuln_ldc))

"""
PLOT
"""

import seaborn as sns
sns.set_theme(style="whitegrid")

fig = plt.figure(figsize=(14, 3))

title_fontsize = 16
label_fontsize = 14
tick_fontsize = 14
legend_fontsize = 14

num_row = 1
num_col = 3
num_plots = num_row * num_col

colors = ["dimgrey", "mediumseagreen","royalblue","tomato"]
ssp_labels = utils.get_ssp_labels(ssps)

ymin_list = [-10,-2.5,-20]
ymax_list = [40,9,85]

for n in range(num_plots):

    ax1 = fig.add_subplot(num_row, num_col,n+1)

    y_zero = [0 for x in range(len(years))]
    ymin = ymin_list[n]
    ymax = ymax_list[n]

    if n == 0:
        all_to_plot = y_all
        haz_to_plot = y_haz
        exp_to_plot = y_exp
        vuln_to_plot = y_vuln
        error_all_to_plot = error_all
        error_haz_to_plot = error_haz
        error_exp_to_plot = error_exp
        error_vuln_to_plot = error_vuln
        title = "(a) All countries"

    elif n == 1:
        all_to_plot = y_all_dev
        haz_to_plot = y_haz_dev
        exp_to_plot = y_exp_dev
        vuln_to_plot = y_vuln_dev
        error_all_to_plot = error_all_dev
        error_haz_to_plot = error_haz_dev
        error_exp_to_plot = error_exp_dev
        error_vuln_to_plot = error_vuln_dev
        title = "(b) Developed countries"

    elif n == 2:
        all_to_plot = y_all_ldc
        haz_to_plot = y_haz_ldc
        exp_to_plot = y_exp_ldc
        vuln_to_plot = y_vuln_ldc
        error_all_to_plot = error_all_ldc
        error_haz_to_plot = error_haz_ldc
        error_exp_to_plot = error_exp_ldc
        error_vuln_to_plot = error_vuln_ldc
        title = "(c) Least-developed countries"

    offset = 0.8  # Offset for clear separation
    #x_offsets = [-offset*1.5, -offset/2, offset/2, offset*1.5]  # Distribute points around the year
    x_offsets = [0,0,0,0]

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=1)
    
    ax1.errorbar(np.array(years) + x_offsets[0], all_to_plot, yerr=error_all_to_plot, 
                label=None, marker='o', color=colors[0], linestyle="dotted",
                capsize=2, capthick=0.8, elinewidth=0.8, markersize=4,
                ecolor=colors[0], alpha=0.8)
    
    ax1.errorbar(np.array(years) + x_offsets[1], haz_to_plot, yerr=error_haz_to_plot, 
                label=None, marker='o', color=colors[1],
                capsize=2, capthick=0.8, elinewidth=0.8, markersize=4,
                ecolor=colors[1], alpha=0.8)
    
    ax1.errorbar(np.array(years) + x_offsets[2], exp_to_plot, yerr=error_exp_to_plot, 
                label=None, marker='o', color=colors[2],
                capsize=2, capthick=0.8, elinewidth=0.8, markersize=4,
                ecolor=colors[2], alpha=0.8)
    
    ax1.errorbar(np.array(years) + x_offsets[3], vuln_to_plot, yerr=error_vuln_to_plot, 
                label=None, marker='o', color=colors[3],
                capsize=2, capthick=0.8, elinewidth=0.8, markersize=4,
                ecolor=colors[3], alpha=0.8)
    
    ax1.plot(np.array(years) + x_offsets[0], all_to_plot, color=colors[0], linestyle="dotted", marker='o', markersize=4, label="All")
    ax1.plot(np.array(years) + x_offsets[1], haz_to_plot, color=colors[1], marker='o', markersize=4, label=r"VPD$_{fs}$")
    ax1.plot(np.array(years) + x_offsets[2], exp_to_plot, color=colors[2], marker='o', markersize=4, label=r"PD$_{forest}$")
    ax1.plot(np.array(years) + x_offsets[3], vuln_to_plot, color=colors[3], marker='o', markersize=4, label=r"HVI")

    ax1.set_title(title, fontsize=title_fontsize, weight="bold")
    ax1.set_xlabel("Year",fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.grid(True,color="gainsboro")

    ax1.set_yscale('linear')
    ax1.set_ylim(ymin, ymax)

    ax1.set_title(title, fontsize=title_fontsize, weight="bold")
    ax1.set_xlabel("Year",fontsize=label_fontsize)
    ax1.grid(True,color="gainsboro")

    if n == 0:
        ax1.set_ylabel("Avoided damage [Billion US$]", fontsize=label_fontsize)

    if n == 2:
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=label_fontsize)

# -----------------------------------------
fig.subplots_adjust(left=0.065,
                    bottom=0.2,
                    right=0.885,
                    top=0.87,
                    wspace=0.2)  # 0.02
                    #hspace=0.65)  # 0.65

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()
