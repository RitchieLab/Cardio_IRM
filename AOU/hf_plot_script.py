# load packages
import pandas as pd
import argparse as ap
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, precision_recall_curve 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
import seaborn as sns

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--sig', required = True, help = 'significant filename')
    
    parser.add_argument('--important', required = True, help = 'important filename')
    
    parser.add_argument('--beta', required = True, help = 'beta filename')
    
    parser.add_argument('--mean_metrics', required = True, help = 'mean metrics filename')
    
    parser.add_argument('--output_dir', required = True, help = 'output filename')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
sig_filename = args.sig
important_filename = args.important
beta_filename = args.beta
mean_metrics_filename = args.mean_metrics
output_dir = args.output_dir

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# read in input files
hf_balanced = pd.read_csv(input_filename)
significant_95 = pd.read_csv(sig_filename, index_col = 0)
important_95 = pd.read_csv(important_filename, index_col = 0)
beta = pd.read_csv(beta_filename, index_col = 0, usecols = ['COLUMN', colname])
mean_metrics = pd.read_csv(mean_metrics_filename, index_col = 0)

# set significance threshold
corrected_sig = 0.05

# split dataset
hf_train = hf_balanced.sample(frac = 0.7, random_state = iter)
hf_reg = hf_train.sample(frac = 0.5, random_state = iter)
hf_lasso = hf_train.drop(hf_reg.index)
hf_no_train = hf_balanced.drop(hf_train.index)

# make CRS column list
crs_cols = ['T2D',
            'TRIG_INV_NORMAL_SCALE',
            'LDL_INV_NORMAL_SCALE',
            'HDL_INV_NORMAL_SCALE',
            'GLUCOSE_INV_NORMAL_SCALE',
            'HbA1c_INV_NORMAL_SCALE',
            'SBP_INV_NORMAL_SCALE',
            'DBP_INV_NORMAL_SCALE']

# filter CRS columns to those that are important and significant
crs_cols = [item for item in crs_cols if item in significant_95.index]
crs_cols = [item for item in crs_cols if item in important_95.index]

# create weighted column list
crs_weighted_cols = [col.replace(col, (col + '_WEIGHTED')) for col in crs_cols]

# make PXS column list
pxs_cols = ['BMI_INV_NORMAL_SCALE',
            'SMOKING',
            'PA_EVERYDAY_SCALE',
            'NEIGHBORHOOD_DRUG_USE_SCALE',
            'NEIGHBORHOOD_SAFE_CRIME_SCALE',
            'NEIGHBORHOOD_TRUST_SCALE',
            'NEIGHBORHOOD_BUILDING_SCALE',
            'NEIGHBORHOOD_ALCOHOL_SCALE',
            'NEIGHBORHOOD_VANDALISM_SCALE',
            'NEIGHBORHOOD_SIDEWALK_SCALE',
            'NEIGHBORHOOD_BIKE_SCALE',
            'NEIGHBORHOOD_CLEAN_SCALE',
            'NEIGHBORHOOD_WATCH_SCALE',
            'NEIGHBORHOOD_HOUSING_SCALE',
            'NEIGHBORHOOD_GET_ALONG_SCALE',
            'NEIGHBORHOOD_UNSAFE_WALK_SCALE',
            'NEIGHBORHOOD_CARE_SCALE',
            'NEIGHBORHOOD_ALOT_CRIME_SCALE',
            'NEIGHBORHOOD_CRIME_WALK_SCALE',
            'NEIGHBORHOOD_SAME_VALUES_SCALE',
            'NEIGHBORHOOD_NOISE_SCALE',
            'NEIGHBORHOOD_GRAFFITI_SCALE',
            'NEIGHBORHOOD_FREE_AMENITIES_SCALE',
            'NEIGHBORHOOD_PPL_HANGING_AROUND_SCALE',
            'NEIGHBORHOOD_TROUBLE_SCALE',
            'NEIGHBORHOOD_STORES_SCALE',
            'NEIGHBORHOOD_TRANSIT_SCALE',
            'INCOME_SCALE',
            'EDUCATION_HIGHEST_SCALE',
            'CENSUS_MEDIAN_INCOME_INV_NORMAL_SCALE',
            'SOCIAL_DEPRIVATION_INDEX_INV_NORMAL_SCALE']

# filter PXS columns to those that are important and significant
pxs_cols = [item for item in pxs_cols if item in significant_95.index]
pxs_cols = [item for item in pxs_cols if item in important_95.index]

# create weighted column list
pxs_weighted_cols = [col.replace(col, (col + '_WEIGHTED')) for col in pxs_cols]

# create eval column list
eval_col_list = ['PGS',
                 'CRS_SUM',
                 'CRS_WEIGHTED_SUM',
                 'PXS_SUM',
                 'PXS_WEIGHTED_SUM',
                 ['PGS', 'CRS_SUM'],
                 ['PGS', 'CRS_WEIGHTED_SUM'],
                 ['PGS', 'PXS_SUM'],
                 ['PGS', 'PXS_WEIGHTED_SUM'],
                 ['CRS_SUM', 'PXS_SUM'],
                 ['CRS_WEIGHTED_SUM', 'PXS_WEIGHTED_SUM'],
                 ['PGS', 'CRS_SUM', 'PXS_SUM'],
                 ['PGS', 'CRS_WEIGHTED_SUM', 'PXS_WEIGHTED_SUM'],
                 crs_cols,
                 pxs_cols,
                 (crs_cols + pxs_cols),
                 (['PGS'] + crs_cols + pxs_cols)]

# compute weighted columns
for col in (crs_cols + pxs_cols):
    weighted_colname = col + '_WEIGHTED'
    beta_val = beta.loc[col, colname]
    hf_no_train[weighted_colname] = hf_no_train[col] * beta_val

# compute integrated scores
hf_no_train['CRS_SUM'] = hf_no_train[crs_cols].sum(axis = 1, min_count = 1)
hf_no_train['CRS_WEIGHTED_SUM'] = hf_no_train[crs_weighted_cols].sum(axis = 1, min_count = 1)

hf_no_train['PXS_SUM'] = hf_no_train[pxs_cols].sum(axis = 1, min_count = 1)
hf_no_train['PXS_WEIGHTED_SUM'] = hf_no_train[pxs_weighted_cols].sum(axis = 1, min_count = 1)

# split testing dataset
hf_val = hf_no_train.sample(frac = 0.5, random_state = iter)
hf_test = hf_no_train.drop(hf_val.index)

# apply SMOTE
x = hf_val.drop(columns = ['HF'])
y = hf_val[['HF']]
x_resampled, y_resampled = SMOTE(random_state = iter).fit_resample(x, y)
hf_val = pd.concat([x_resampled, y_resampled], axis = 1)

# create empty dictionaries
roc_data = {}
prc_data = {}
auroc_auprc = {}

# evaluate models
for index, col in enumerate(eval_col_list, start = 1):
    if isinstance(col, str):
        model_df = hf_val[['HF', 'AGE', 'SEX'] + [col]].dropna()
        predictors = ['AGE', 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = hf_val[['HF', 'AGE', 'SEX'] + col].dropna()
        predictors = ['AGE', 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = ['HF'] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        
    else:
        model = sm.Logit(model_df['HF'], model_df[predictors]).fit()
        test_df = hf_test[all_cols].dropna()
        y_prob_cont = model.predict(test_df[predictors])
        y_prob_bin = (y_prob_cont >= 0.5).astype(int)
        
        fpr, tpr, _ = roc_curve(test_df['HF'], y_prob_cont)
        precision, recall, _ = precision_recall_curve(test_df['HF'], y_prob_cont)
        
        #print(col)
        col = str(col).replace("'", "")
        col = col.replace("[", "")
        col = col.replace("]", "")
        col = col.replace(",", " +")
        col = col.replace("PGS + T2D + LDL_INV_NORMAL_SCALE + HDL_INV_NORMAL_SCALE + BMI_INV_NORMAL_SCALE + SMOKING + PA_EVERYDAY_SCALE + NEIGHBORHOOD_DRUG_USE_SCALE + NEIGHBORHOOD_SAFE_CRIME_SCALE + NEIGHBORHOOD_TRUST_SCALE + NEIGHBORHOOD_BUILDING_SCALE + NEIGHBORHOOD_VANDALISM_SCALE + NEIGHBORHOOD_SIDEWALK_SCALE + NEIGHBORHOOD_BIKE_SCALE + NEIGHBORHOOD_CLEAN_SCALE + NEIGHBORHOOD_UNSAFE_WALK_SCALE + NEIGHBORHOOD_CARE_SCALE + NEIGHBORHOOD_ALOT_CRIME_SCALE + NEIGHBORHOOD_CRIME_WALK_SCALE + NEIGHBORHOOD_GRAFFITI_SCALE + NEIGHBORHOOD_FREE_AMENITIES_SCALE + NEIGHBORHOOD_PPL_HANGING_AROUND_SCALE + NEIGHBORHOOD_TROUBLE_SCALE + NEIGHBORHOOD_STORES_SCALE + NEIGHBORHOOD_TRANSIT_SCALE + INCOME_SCALE + EDUCATION_HIGHEST_SCALE + CENSUS_MEDIAN_INCOME_INV_NORMAL_SCALE",
                              "PGS + CRS Risk Factors + PXS Risk Factors")
        col = col.replace("T2D + LDL_INV_NORMAL_SCALE + HDL_INV_NORMAL_SCALE + BMI_INV_NORMAL_SCALE + SMOKING + PA_EVERYDAY_SCALE + NEIGHBORHOOD_DRUG_USE_SCALE + NEIGHBORHOOD_SAFE_CRIME_SCALE + NEIGHBORHOOD_TRUST_SCALE + NEIGHBORHOOD_BUILDING_SCALE + NEIGHBORHOOD_VANDALISM_SCALE + NEIGHBORHOOD_SIDEWALK_SCALE + NEIGHBORHOOD_BIKE_SCALE + NEIGHBORHOOD_CLEAN_SCALE + NEIGHBORHOOD_UNSAFE_WALK_SCALE + NEIGHBORHOOD_CARE_SCALE + NEIGHBORHOOD_ALOT_CRIME_SCALE + NEIGHBORHOOD_CRIME_WALK_SCALE + NEIGHBORHOOD_GRAFFITI_SCALE + NEIGHBORHOOD_FREE_AMENITIES_SCALE + NEIGHBORHOOD_PPL_HANGING_AROUND_SCALE + NEIGHBORHOOD_TROUBLE_SCALE + NEIGHBORHOOD_STORES_SCALE + NEIGHBORHOOD_TRANSIT_SCALE + INCOME_SCALE + EDUCATION_HIGHEST_SCALE + CENSUS_MEDIAN_INCOME_INV_NORMAL_SCALE",
                              "CRS Risk Factors + PXS Risk Factors")
        col = col.replace("T2D + LDL_INV_NORMAL_SCALE + HDL_INV_NORMAL_SCALE",
                              "CRS Risk Factors")
        col = col.replace("BMI_INV_NORMAL_SCALE + SMOKING + PA_EVERYDAY_SCALE + NEIGHBORHOOD_DRUG_USE_SCALE + NEIGHBORHOOD_SAFE_CRIME_SCALE + NEIGHBORHOOD_TRUST_SCALE + NEIGHBORHOOD_BUILDING_SCALE + NEIGHBORHOOD_VANDALISM_SCALE + NEIGHBORHOOD_SIDEWALK_SCALE + NEIGHBORHOOD_BIKE_SCALE + NEIGHBORHOOD_CLEAN_SCALE + NEIGHBORHOOD_UNSAFE_WALK_SCALE + NEIGHBORHOOD_CARE_SCALE + NEIGHBORHOOD_ALOT_CRIME_SCALE + NEIGHBORHOOD_CRIME_WALK_SCALE + NEIGHBORHOOD_GRAFFITI_SCALE + NEIGHBORHOOD_FREE_AMENITIES_SCALE + NEIGHBORHOOD_PPL_HANGING_AROUND_SCALE + NEIGHBORHOOD_TROUBLE_SCALE + NEIGHBORHOOD_STORES_SCALE + NEIGHBORHOOD_TRANSIT_SCALE + INCOME_SCALE + EDUCATION_HIGHEST_SCALE + CENSUS_MEDIAN_INCOME_INV_NORMAL_SCALE",
                              "PXS Risk Factors")
        #print(col)
        col = 'Model ' + str(index) + ': ' + col
        
        auroc = mean_metrics.loc[col, 'AUROC']
        roc_data[col] = (fpr, tpr, auroc)
        auprc = mean_metrics.loc[col, 'AUPRC']
        prc_data[col] = (precision, recall, auprc)
        auroc_auprc[col] = (auroc, auprc)

# set colorblind palette
sns.set_palette("colorblind")

# make ROC and PRC curves on 2 panels of the same plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax1 = axes[0]
ax2 = axes[1]

# make ROC curve
#plt.figure(figsize = (10, 8))
for col, (fpr, tpr, auroc) in roc_data.items():
    ax1.plot(fpr, tpr, lw = 7)
ax1.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', lw = 7)
ax1.set_xlabel('False Positive Rate', fontsize = 35)
ax1.set_ylabel('True Positive Rate', fontsize = 35)
ax1.set_title('AOU Heart Failure Prediction Receiver-Operating Curves', fontsize = 40)
ax1.tick_params(axis = 'both', labelsize = 30)
#ax1.legend(loc='lower right', fontsize = 'small')
ax1.grid(True)
#ax1.text(0.02, 1.02, "A", transform = ax1.transAxes, fontsize = 35, fontweight = "bold", va = "bottom", ha = "right")
#plt.tight_layout()
#plt.savefig((output_dir + "HF_ROC_curve.png"), dpi = 300)

# make PRC curve
#plt.figure(figsize = (10, 8))
for col, (precision, recall, auprc) in prc_data.items():
    ax2.plot(recall, precision, lw = 7)
ax2.set_xlabel('Recall', fontsize = 35)
ax2.set_ylabel('Precision', fontsize = 35)
ax2.set_title('AOU Heart Failure Prediction Precision-Recall Curves', fontsize = 40)
ax2.tick_params(axis = 'both', labelsize = 30)
#ax2.legend(loc = 'upper right', fontsize = 'small')
ax2.grid(True)
#ax2.text(0.02, 1.02, "B", transform = ax2.transAxes, fontsize = 35, fontweight = "bold", va = "bottom", ha = "right")
#plt.tight_layout()
#plt.savefig((output_dir + "HF_PRC_curve.png"), dpi = 300)

# create legend
handles = ax1.get_lines()  # handles for the legend
#combined_labels = [
#    f"{model} (AUROC = {auroc:.3f}, AUPRC = {auprc:.3f})"
#    for model, (auroc, auprc) in auroc_auprc.items()]
combined_labels = [f"{model}" for model, (auroc, auprc) in auroc_auprc.items()]
fig.legend(handles, combined_labels, loc = 'lower center', ncol = 2, fontsize = 35)#, bbox_to_anchor = (0.5, -0.08))
fig.set_size_inches(36, 24)

# export plot
plt.tight_layout(rect = [0, 0.3, 1, 1])
plt.savefig(output_dir + "HF_ROC_PRC_curve_combined.png", dpi = 300)
