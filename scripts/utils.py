# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import sys
sys.path.append('../')
import pandas as pd
import numpy as np

regions_path = '../../data/source/region_classification.xlsx'

def get_plot_titles_and_labels(plot_option,average_flag):

    """
    Return a dictionary with the labels for y and X
    """

    if plot_option == "label" and average_flag == True:

        var_dict = {
            'damage_gdppc_weighted':r'% GDP per 1M capita'
            ,'log_damage_gdppc_weighted':r'% GDP per 1M capita'
            , 'log_damage_gdppc_weighted_std': r'% GDP per 1M capita'
            ,'damage_gdp_weighted':r'%GDP'
            , 'log_damage_gdp_weighted': r'%GDP'
            , 'log_damage_gdp_weighted_std': r'%GDP'
            ,'gdp': r'$\sum$GDP [trillion USD]'
            , 'pop': r'$\sum$Population [billion]'
            , 'gdppc': r'$\overline{gdppc}$ [kUSD]'
            , 'pop_forest': r'$\overline{PD_{forest}}$ [per km$^2$]'
            , 'norm_pop_forest': r'$\overline{p_{forest}}$ [-]'
            , 'vpd': r'$\overline{VPD}$ [hPa]'
            , 'summer_vpd': r'$\overline{VPD_{fs}}$ [hPa]'
            , 'fwixx': r'$\overline{fwixx}$ [-]'
            , 'fwixd': r'$\overline{fwixd}$ [day]'
            , 'fwils': r'$\overline{fwils}$ [day]'
            , 'fwisa': r'$\overline{fwisa}$ [-]'
            , 'totprecipyear': r'$\overline{pr}$ [m]'
            , 'hdi': r'$\overline{HDI}$ [-]'
            , 'hvi':r'$\overline{HVI}$ [-]'
            , 'gee': r'$\overline{gee}$ [-]'
            , 'cce': r'$\overline{cqe}$ [-]'
            , 'rqe': r'$\overline{rqe}$ [-]'
            , 'gii': r'$\overline{gii}$ [-]'
            , 'gini': r'$\overline{gini}$ [-]'
            , 'pop_density': r'$\overline{pop\_density}$ [per km$^2$]'
            , 'forest_pct': r'$\overline{forest\_area}$ [%]'
            , 'log_hazard': r'$\it{log}$(hazard)'
            , 'log_exp': r'$\it{log}$(exp)'
            , 'log_vuln': r'$\it{log}$(vuln)'
            , 'log_hazard_std': r'$\it{log}$(hazard)'
            , 'log_exp_std': r'$\it{log}$(exp)'
            , 'log_vuln_std': r'$\it{log}$(vuln)'
            , 'log_norm_hazard': r'$\it{log}$(hazard)'
            , 'log_norm_exp': r'$\it{log}$(exp)'
            , 'log_norm_vuln': r'$\it{log}$(vuln)'
                    }

    if plot_option == "label" and average_flag == False:

        var_dict = {
            'damage_gdppc_weighted':r'% GDP per 1M capita'
            ,'log_damage_gdppc_weighted':r'% GDP per 1M capita'
            , 'log_damage_gdppc_weighted_std': r'% GDP per 1M capita'
            ,'damage_gdp_weighted':r'%GDP'
            , 'log_damage_gdp_weighted': r'%GDP'
            , 'log_damage_gdp_weighted_std': r'%GDP'
            ,'gdp': r'$\sum$GDP [trillion USD]'
            , 'pop': r'$\sum$Population [billion]'
            , 'gdppc': r'$gdppc$ [kUSD]'
            , 'pop_forest': r'PD$_{forest}$ [per km$^2$]'
            , 'norm_pop_forest': r'$p_{forest}$ [-]'
            , 'vpd': r'$VPD$ [hPa]'
            , 'summer_vpd': r'VPD$_{fs}$ [hPa]'
            , 'log_vpd': r'$log(VPD_{fs})$'
            , 'log_pdforest': r'$log(PD_{forest})$'
            , 'log_hvi':  r'$log(HVI)$'
            , 'fwixx': r'$fwixx$ [-]'
            , 'fwixd': r'$fwixd$ [day]'
            , 'fwils': r'$fwils$ [day]'
            , 'fwisa': r'$fwisa$ [-]'
            , 'totprecipyear': r'$pr$ [m]'
            , 'hdi': r'HDI [-]'
            , 'hvi':r'HVI [-]'
            , 'gee': r'$gee$ [-]'
            , 'cce': r'$cqe$ [-]'
            , 'rqe': r'$rqe$ [-]'
            , 'gii': r'$gii$ [-]'
            , 'gini': r'$gini$ [-]'
            , 'pop_density': r'$pop\_density$ [per km$^2$]'
            , 'forest_pct': r'$forest\_area$ [%]'
            , 'log_hazard': r'$\it{log}$(hazard)'
            , 'log_exp': r'$\it{log}$(exp)'
            , 'log_vuln': r'$\it{log}$(vuln)'
            , 'log_hazard_std': r'$\it{log}$(hazard)'
            , 'log_exp_std': r'$\it{log}$(exp)'
            , 'log_vuln_std': r'$\it{log}$(vuln)'
            , 'log_norm_hazard': r'$\it{log}$(hazard)'
            , 'log_norm_exp': r'$\it{log}$(exp)'
            , 'log_norm_vuln': r'$\it{log}$(vuln)'
                    }

    if plot_option == "title":
        var_dict = {'damage_gdppc_weighted':r'Wildfire damage'
            ,'log_damage_gdppc_weighted':r'Wildfire damage'
            , 'log_damage_gdppc_weighted_std': r'Wildfire damage'
            ,'damage_gdp_weighted':r'Wildfire damage'
            , 'log_damage_gdp_weighted': r'Wildfire damage'
            , 'log_damage_gdp_weighted_std': r'Wildfire damage'
            ,'gdp': 'GDP'
            , 'pop': 'Population'
            , 'gdppc': 'GDP per capita'
            , 'pop_forest': 'Population density (forest)'
            , 'norm_pop_forest': 'Population close to forest'
            , 'vpd': 'Vapor pressure deficit'
            , 'summer_vpd': 'Fire season VPD'
            , 'log_vpd': 'Fire season VPD'
            , 'log_pdforest': 'Population close to forest'
            , 'log_hvi': 'Human vulnerability index'
            , 'fwixx': 'Extreme value of FWI'
            , 'fwixd': 'Extreme fire weather'
            , 'fwils': 'Length of fire season'
            , 'fwisa': 'Seasonal average of FWI'
            , 'totprecipyear': 'Annual precipitation'
            , 'hdi': 'Human development index'
            , 'hvi': 'Human vulnerability index'
            , 'gee': 'Governance efficiency'
            , 'cce': 'Corruption control'
            , 'rqe': 'Regulatory quality'
            , 'gii': 'Gender inequality'
            , 'gini': 'Income inequality'
            , 'pop_density': f'Population density'
            , 'forest_pct': f'Forest pct.'
            , 'log_hazard': r'$\it{log}$(hazard)'
            , 'log_exp': r'$\it{log}$(exposure)'
            , 'log_vuln': r'$\it{log}$(vulnerability)'
            , 'log_hazard_std': r'$\it{log}$(hazard)'
            , 'log_exp_std': r'$\it{log}$(exposure)'
            , 'log_vuln_std': r'$\it{log}$(vulnerability)'
            , 'log_norm_hazard': r'$\it{log}$(hazard)'
            , 'log_norm_exp': r'$\it{log}$(exposure)'
            , 'log_norm_vuln': r'$\it{log}$(vulnerability)'}

    return var_dict

def generate_alphabet_list(n,option):

    import string

    if option == "lower":
        alphabets = list(string.ascii_lowercase)

    if option == "upper":
        alphabets = list(string.ascii_uppercase)

    alphabet_list = alphabets[:n]

    return alphabet_list

def get_ssp_years():
    
    """
    Get the list of years for SSP projection
    """
    years = [2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070]

    return years

def get_ssp_labels(ssps):

    """
    Return a list of complete SSP labels
    """
    ssp_labels = []

    for ssp in ssps:
        if ssp == "ssp1":
            ssp_labels.append("SSP126")
        if ssp == "ssp2":
            ssp_labels.append("SSP245")
        if ssp == "ssp3":
            ssp_labels.append("SSP370")
        if ssp == "ssp4":
            ssp_labels.append("SSP460")
        if ssp == "ssp5":
            ssp_labels.append("SSP585")

    return ssp_labels

def get_esm_labels(esms):

    """
    Return list of shortened ESM labels
    """

    esm_labels = {
        "GFDL-ESM4": "gfdl",
        "IPSL-CM6A-LR": "ipsl",
        "MPI-ESM1-2-HR": "mpi",
        "MRI-ESM2-0": "mri",
        "UKESM1-0-LL": "ukesm"
    }

    # Generate the output dictionary
    return {esm: esm_labels.get(esm, "unknown") for esm in esms}

def get_ar6_region(iso,ar6_label):

    """
    Get AR6 region for a country
    """
    df_dev_level = pd.read_excel(regions_path)

    if iso in df_dev_level['ISO'].values:
        region = df_dev_level.loc[df_dev_level['ISO'] == iso, ar6_label].values.tolist()[0]
    else:
        print ("ISO not found for:",iso)
        region = np.nan

    return region

def get_all_country_names():

    """
    Get list of country names
    """

    df = pd.read_excel(regions_path)
    countries = df.name.tolist()
    countries_unique = list(set(countries))

    return countries_unique

def get_all_iso():

    """
    Get list of ISO
    """
    df = pd.read_excel(regions_path)
    iso_list = df.ISO.tolist()
    iso_unique = list(set(iso_list))

    return iso_unique

def get_iso(country):

    """
    Get the ISO code of a country
    """
    df_iso = pd.read_excel(regions_path)
    iso = df_iso.loc[df_iso['name'] == country, 'ISO'].values.tolist()[0]

    return iso

