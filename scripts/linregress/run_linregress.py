# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import sys
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

"""
PARAMETERS
"""
pop_forest_option = "pop_forest" #"pop_forest" for pop_forest = pop/forest_area; "pop_wui" for pop_forest = population in forested grid

if pop_forest_option == "pop_forest":
    pdforest_str = ""
if pop_forest_option == "pop_wui":
    pdforest_str = "pop_wui/"

base_path = f'../../data'
input_file_hist = f'{base_path}/historical/{pdforest_str}INPUT_HISTORICAL.csv'
training_data_save_file = f'{base_path}/historical/{pdforest_str}TRAINING_DATA.csv'
model_save_file = f'{base_path}/historical/{pdforest_str}model.pickle'
mean_std_X_y_file = f'{base_path}/historical/{pdforest_str}mean_SD_X_y.csv'
beta_coef_file = f'{base_path}/historical/{pdforest_str}beta_coef.csv'

input_file_ssp = f'{base_path}/ssp/{pdforest_str}INPUT_SSP.csv'
ssp_prediction_output_file = f'{base_path}/ssp/{pdforest_str}OUTPUT_SSP_PREDICTION.csv'

y_var = 'damage_gdp_weighted'
predictors = ['log_vpd', 'log_pdforest', 'log_hvi']
predictors_non_log = ['summer_vpd','pop_forest','hvi']

do_kfold_cv = True
do_bootstrap = False
plot_coefficients = True
do_ssp_projection = True

"""
LOAD DATA
"""
df = pd.read_csv(input_file_hist)

df = df.dropna(subset=[y_var])
df = df.dropna(subset=predictors)
print (f"SHAPE OF INPUT DATA: {df.shape}")

df_header = df[['iso','region_dev','region_ar6_6','pop','pop_density','forest_pct','gdp','damage']]

"""
GET X AND Y
"""
y_non_log = df[y_var]
y_log = df[[y_var]].rename(columns={y_var: f'log_{y_var}'})
y_log = np.log(y_log)
y = y_log

X_non_log = df[predictors_non_log]
X_log = df[predictors]
X_log = X_log.dropna(how='any')
X = X_log
X = sm.add_constant(X)

"""
10-FOLD CROSS VALIDATION
"""
if do_kfold_cv == True:
    print()
    print("################################")
    print("Do 5-fold cross validation...")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize an empty list to store scores
    r2_scores = []
    mse_scores = []

    weights = df['weights']

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kfold.split(X)):

        # print("Fold number:", i)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

        # Standardize X and y
        X_train_standardized = x_scaler.fit_transform(X_train)
        X_test_standardized = x_scaler.transform(X_test)  # ONLY transform X_test, DO NOT FIT again

        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        y_train_standardized = y_scaler.fit_transform(y_train)
        y_test_standardized = y_scaler.transform(y_test)

        model_cv = sm.GLM(y_train_standardized, X_train_standardized, missing='drop', family=sm.families.Gaussian(),var_weights=weights_train).fit()

        y_pred_standardized = model_cv.predict(X_test_standardized)
        mse_score = mean_squared_error(y_test_standardized, y_pred_standardized)
        r2_cv = model_cv.pseudo_rsquared()
        r2_scores.append(r2_cv)
        mse_scores.append(mse_score)

    # MEAN AND AVG
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)

    print()
    print("MEAN and STD R2 score for cross-validation:", round(r2_mean, 4),round(r2_std, 4))
    print("MEAN and STD MSE score for cross-validation:", round(mse_mean, 4),round(mse_std, 4))
    print("################################")


"""
STANDARDIZE TRAINING DATA
"""
X_col_names = X.columns.tolist()
scaler_std_X = StandardScaler()
scaler_std_y = StandardScaler()
X = scaler_std_X.fit_transform(X)

y = y.values.reshape(-1, 1)
y = scaler_std_y.fit_transform(y)

# -------
# Save mean and STD to csv file
x_hist_mean = scaler_std_X.mean_
y_hist_mean = scaler_std_y.mean_
x_hist_std = scaler_std_X.scale_
y_hist_std = scaler_std_y.scale_

mean_sd_X_y = np.hstack((x_hist_mean, x_hist_std, y_hist_mean, y_hist_std))
col_names_1 = ["mean_" + x for x in X_col_names]
col_names_2 = ["std_" + x for x in X_col_names]
col_names_all = col_names_1 + col_names_2 + ["mean_y_hist", "std_y_hist"]
col_names_all = np.array(col_names_all)

mean_sd_array = np.vstack((col_names_all, mean_sd_X_y))
np.savetxt(mean_std_X_y_file, mean_sd_array, delimiter=',', fmt='%s')

"""
SAVE TRAINING DATA
"""

print ()
print ("Saving training data...")
columns_X_std = [f'{col}_std' for col in predictors]
X_std = X[:,1:]
X_std = pd.DataFrame(X_std, columns=columns_X_std)

columns_y_std = ['log_damage_gdp_weighted_std']
y_std = pd.DataFrame(y,columns=columns_y_std)

df_for_model_training = pd.concat([df_header,y_non_log,y_log,y_std,X_non_log,X_log,X_std], axis=1)
df_for_model_training = df_for_model_training.dropna()
df_for_model_training.to_csv(training_data_save_file,index=False)
num_country = df_for_model_training['iso'].nunique()
print ("Number of countries:",num_country)

"""
BOOTSTRAP
"""
if do_bootstrap == True:

    print()
    print("################################")
    print("Do bootstrap...")

    from sklearn.utils import resample
    n_bootstrap = 1000

    bootstrap_coefs = []
    weights = df['weights']

    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        X_resampled, y_resampled, weights_resampled = resample(X, y, weights, replace=True,random_state=i)
        model = sm.GLM(y_resampled, X_resampled,family=sm.families.Gaussian(),var_weights=weights_resampled).fit()
        bootstrap_coefs.append(model.params)

    bootstrap_coefs_df = pd.DataFrame(bootstrap_coefs)

    bs_coef_means = bootstrap_coefs_df.mean()
    bs_coef_stds = bootstrap_coefs_df.std()

    # Compute 95% confidence intervals
    lower_bounds = bootstrap_coefs_df.quantile(0.025)
    upper_bounds = bootstrap_coefs_df.quantile(0.975)

    # Print results
    print("Coefficient Estimates (Mean):")
    print(bs_coef_means)

    print("\nCoefficient Estimates (Standard Deviation):")
    print(bs_coef_stds)

    print("\n95% Confidence Intervals:")
    print(pd.DataFrame({'Lower Bound': lower_bounds, 'Upper Bound': upper_bounds}))

    print("################################")

    """
    Plot bootstrap coeffs.
    """

    bs_coef_means = bs_coef_means[1:]
    X_col_names = X_col_names[1:]
    lower_bounds = lower_bounds[1:]
    upper_bounds = upper_bounds[1:]

    plt.figure(figsize=(6, 4))
    plt.errorbar(X_col_names, bs_coef_means, yerr=[bs_coef_means - lower_bounds, upper_bounds - bs_coef_means], fmt='o',capsize=5)
    plt.axhline(y=0, color="black", linestyle="dashed")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel('Coefficient value',fontsize=14)
    plt.title(f'Coefficient estimates, n_bootstrap = {n_bootstrap}', fontsize=16)
    plt.tight_layout()
    plt.show()


"""
FIT MODEL
"""
print ()
print ("################################")
print ("FITTING GLM MODEL...")
model_glm = sm.GLM(y, X, missing='drop', family=sm.families.Gaussian(), var_weights=df["weights"]).fit()

print ()
print(model_glm.summary(xname=X_col_names))

r_squared = model_glm.pseudo_rsquared()
coefficients = model_glm.params
se = model_glm.bse
num_obs = model_glm.nobs

# SAVE MODEL AND COEFFICIENTS
model_glm.save(model_save_file)
coeff_arr = np.vstack((np.array(X_col_names),coefficients))
np.savetxt(beta_coef_file,coeff_arr, delimiter=',', fmt='%s')

# GET PARTIAL R-SQUARE
ssr_full = np.sum(model_glm.resid_response ** 2)
partial_r_squared = {}
for i in range(X.shape[1]):
    if i == 0:
        continue  # Skip the constant term
    X_reduced = np.delete(X, i, axis=1)  # Exclude ith predictor
    model_reduced = sm.GLM(y, X_reduced, family=sm.families.Gaussian())
    results_reduced = model_reduced.fit()
    # Compute SSR for the reduced model
    ssr_reduced = np.sum(results_reduced.resid_response ** 2)

    # Calculate partial R-squared
    partial_r_squared[X_col_names[i]] = 1 - (ssr_full / ssr_reduced)

print ()
print("Partial R2 for each predictor:")
print(partial_r_squared)
print ("-------------------")
print(f"Number of observations used: {int(num_obs)}")
print ("Predictors used:",predictors)
print ("Coefficients:",coefficients)
print ("-------------------")
print(f"R2 of fitted GLM model: {round(r_squared,4)}")

"""
DO SSP PREDICTION
"""
if do_ssp_projection == True:

    print()
    print("################################")
    print("Doing prediction with new SSP data...")

    df_ssp = pd.read_csv(input_file_ssp)

    X_ssp_ori = df_ssp[predictors]
    X_ssp = df_ssp[predictors]
    X_ssp_predictors_non_log = df_ssp[predictors_non_log] #X_ssp_predictors_all

    X_ssp = sm.add_constant(X_ssp)

    # GET MEAN AND STD for fitted StandardScalar()
    x_hist_mean = scaler_std_X.mean_
    y_hist_mean = scaler_std_y.mean_
    x_hist_std = scaler_std_X.scale_
    y_hist_std = scaler_std_y.scale_

    X_ssp = scaler_std_X.transform(X_ssp)

    scaler_std_X_ssp_new = StandardScaler()
    X_ssp_predictors_non_log_std = scaler_std_X_ssp_new.fit_transform(X_ssp_predictors_non_log) #X_ssp_predictors_all_std

    # Load model
    model_loaded = sm.load(model_save_file)

    y_ssp = model_loaded.predict(X_ssp)
    y_ssp_predicted_std = y_ssp  # for saving to csv later

    y_hist_std_non_log = np.nanstd(y_non_log)
    y_hist_mean_non_log = np.nanmean(y_non_log)
    y_hist_std_log = np.nanstd(y_log)
    y_hist_mean_log = np.nanmean(y_log)

    # transform Y back to original scale
    y_std = y_hist_std_log
    y_mean = y_hist_mean_log
    y_ssp_log = (y_ssp * y_std) + y_mean  # unstandardized
    y_ssp = np.exp(y_ssp_log)
    y_ssp_std_log = np.nanstd(y_ssp_log)
    y_ssp_mean_log = np.nanmean(y_ssp_log)
    y_ssp[y_ssp < 0] = 0

    # Get mean and std of predicted y
    y_ssp_mean = np.nanmean(y_ssp)
    y_ssp_std = np.nanstd(y_ssp)

    print()
    print("Shape of X_SSP:", X_ssp.shape)
    print("Shape of y_SSP:", y_ssp.shape)
    print(f"MEAN and SD of y_historical (log-transformed):  {y_mean}, {y_std}")
    print(f"MEAN and SD of y_ssp (log-transformed):         {y_ssp_mean_log}, {y_ssp_std_log}")
    print(f"MEAN and SD of y_historical (un-log):           {y_hist_mean_non_log}, {y_hist_std_non_log}")
    print(f"MEAN and SD of y_SSP (un-log):                  {y_ssp_mean}, {y_ssp_std}")

    # SAVE SSP PREDICTION OUTPUT
    header_cols = ['iso','year','ssp','region_dev','region_ar6_6','esm','pop','pop_density','forest_pct','gdp', 'gdppc']
    df_header_ssp = df_ssp[header_cols] #df_iso_year

    df_y_ssp = pd.DataFrame(y_ssp)  # un-log and unstandardized
    df_y_ssp_std = pd.DataFrame(y_ssp_predicted_std)  # log and standardized
    df_y_ssp_log = pd.DataFrame(y_ssp_log) #log-transformed y_ssp

    df_X_ssp = pd.DataFrame(X_ssp_ori) #log_X
    df_X_ssp_std = pd.DataFrame(X_ssp)
    df_X_ssp_std = df_X_ssp_std.drop(df_X_ssp_std.columns[0], axis=1) # drop "const" column

    df_ssp_prediction = pd.concat([df_header_ssp, df_y_ssp, df_y_ssp_log, df_y_ssp_std, df_X_ssp, df_X_ssp_std], axis=1)
    df_ssp_prediction.columns = (header_cols + [y_var] + [f"log_{y_var}"] + [f"log_{y_var}_std"]
                                 + predictors + [f'{pred}_std' for pred in predictors])

    # Calculate damage in USD
    dmg = (df_ssp_prediction[y_var] / 100) * df_ssp_prediction['gdp']  # damage (in USD)
    df_ssp_prediction.insert(11, 'damage', dmg)

    # SAVE SSP prediction file
    print()
    print("Saving SSP projection csv ...")
    df_ssp_prediction.to_csv(ssp_prediction_output_file, index=False)

    print("################################")
    print()

"""
PLOT COEFFICIENTS
"""

if plot_coefficients == True:
    # PREPARE DATA
    coefficients = coefficients[1:]
    se = se[1:]
    conf_int = model_glm.conf_int(alpha=0.05, cols=None)[1:]
    p_vals = model_glm.pvalues[1:]

    coefficients_df = pd.DataFrame({'Coefficient': coefficients, 'Standard_Error': se})
    confidence_intervals = conf_int
    confidence_intervals_df = pd.DataFrame(confidence_intervals, columns=['CI_Lower', 'CI_Upper'])
    coefficients_df = pd.concat([coefficients_df, confidence_intervals_df], axis=1)
    coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

    y_labels = {
        'log_vpd': r'$log(VPD_{fs})$*'
        , 'log_pdforest': r'$log(PD_{forest})$*'
        , 'log_hvi': r'$log(HVI)$*'
    }

    # Plot the coefficients and their confidence intervals
    fig = plt.figure(figsize=(8, 3))
    title_fontsize = 15
    label_fontsize = 14
    legend_fontsize = 15
    markersize = 6
    capsize = 5

    ax1 = fig.add_subplot(111)  # 3 rows, 1 column, subplot 1
    ax1 = plt.errorbar(y=range(len(coefficients_df)),
                       x=coefficients_df['Coefficient'],
                       xerr=[coefficients_df['Coefficient'] - coefficients_df['CI_Lower'],
                             coefficients_df['CI_Upper'] - coefficients_df['Coefficient']],
                       fmt='o', markersize=markersize, capsize=capsize, color='skyblue', ecolor='black')

    # Plot asterisks for significant predictors (p-value < 0.05)
    for i, p_value in enumerate(p_vals):  # Exclude the intercept
        # for i, p_value in enumerate(model.pvalues[1:]):  # Exclude the intercept
        if p_value < 0.001:
            idx = np.where(coefficients_df.index.values == i)[0]
            plt.text(coefficients_df['Coefficient'][i], idx+0.3, '***', fontsize=legend_fontsize, ha='center',va='top', color='red')

        elif p_value < 0.01:
            idx = np.where(coefficients_df.index.values == i)[0]
            plt.text(coefficients_df['Coefficient'][i], idx+0.3, '**', fontsize=legend_fontsize, ha='center', va='top',color='red')

        elif p_value < 0.05:
            idx = np.where(coefficients_df.index.values == i)[0]
            plt.text(coefficients_df['Coefficient'][i], idx+0.3, '*', fontsize=legend_fontsize, ha='center', va='top',color='red')

    plt.yticks(range(len(coefficients_df)), [y_labels[predictors[i]] for i in coefficients_df.index],ha='right', fontsize=label_fontsize)
    plt.ylim(bottom=-0.5, top=2.5)
    plt.xticks(fontsize=label_fontsize)
    plt.xlabel(r'$\beta$-coefficient', fontsize=label_fontsize)
    plt.axvline(0, linestyle='--', color='gray', linewidth=1)
    plt.text(0.8,2.2,f'$R^{2}$ = ' + f'{r_squared:.2f}', fontsize=title_fontsize, ha='right', va='top')

    fig.subplots_adjust(left=0.18,
                        bottom=0.23,
                        right=0.93,
                        top=0.85,
                        wspace=0.27,  # 0.02
                        hspace=0.65)  # 0.65

    plt.show()

