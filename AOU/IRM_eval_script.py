# load packages
import pandas as pd
import argparse as ap
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--sig', required = True, help = 'significant filename')
    
    parser.add_argument('--important', required = True, help = 'important filename')
    
    parser.add_argument('--beta', required = True, help = 'beta filename')
    
    parser.add_argument('--pheno', required = True, help = 'phenotype')
    
    parser.add_argument('--output_dir', required = True, help = 'output filename')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
sig_filename = args.sig
important_filename = args.important
beta_filename = args.beta
pheno = args.pheno
output_dir = args.output_dir
print(output_dir)

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# read in input files
input = pd.read_csv(input_filename)
significant_95 = pd.read_csv(sig_filename, index_col = 0)
important_95 = pd.read_csv(important_filename, index_col = 0)
beta = pd.read_csv(beta_filename, index_col = 0, usecols = ['COLUMN', colname])

# set significance threshold
corrected_sig = 0.05

# split dataset
train = input.sample(frac = 0.7, random_state = iter)
reg = train.sample(frac = 0.5, random_state = iter)
lasso = train.drop(reg.index)
no_train = input.drop(train.index)

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

# rename PGS

# create eval column list
eval_col_list = ['PGS',
                 'CRS_SUM',
                 'CRS_WEIGHTED_SUM',
                 'PXS_SUM',
                 'PXS_WEIGHTED_SUM',
                 ['PGS', 'CRS_SUM', 'PXS_SUM'],
                 ['CRS_SUM', 'PXS_SUM'],
                 ['PGS', 'CRS_SUM'],
                 ['PGS', 'PXS_SUM'],
                 ['PGS', 'CRS_WEIGHTED_SUM', 'PXS_WEIGHTED_SUM'],
                 ['CRS_WEIGHTED_SUM', 'PXS_WEIGHTED_SUM'],
                 ['PGS', 'CRS_WEIGHTED_SUM'],
                 ['PGS', 'PXS_WEIGHTED_SUM'],
                 (['PGS'] + crs_cols + pxs_cols),
                 crs_cols,
                 pxs_cols,
                 (crs_cols + pxs_cols)]

# compute weighted columns
for col in (crs_cols + pxs_cols):
    weighted_colname = col + '_WEIGHTED'
    beta_val = beta.loc[col, colname]
    no_train[weighted_colname] = no_train[col] * beta_val

# compute integrated scores
no_train['CRS_SUM'] = no_train[crs_cols].sum(axis = 1, min_count = 1)
no_train['CRS_WEIGHTED_SUM'] = no_train[crs_weighted_cols].sum(axis = 1, min_count = 1)

no_train['PXS_SUM'] = no_train[pxs_cols].sum(axis = 1, min_count = 1)
no_train['PXS_WEIGHTED_SUM'] = no_train[pxs_weighted_cols].sum(axis = 1, min_count = 1)

# split testing dataset
val = no_train.sample(frac = 0.5, random_state = iter)
test = no_train.drop(val.index)

# apply SMOTE
x = val.drop(columns = [pheno])
y = val[[pheno]]
x_resampled, y_resampled = SMOTE(random_state = iter).fit_resample(x, y)
val = pd.concat([x_resampled, y_resampled], axis = 1)

# make age col
age_col = 'AGE_' + pheno

# create empty dictionaries
auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}

# evaluate models
for col in eval_col_list:
    if isinstance(col, str):
        model_df = val[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = val[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            auroc_dict[col].append(np.nan)
            auprc_dict[col].append(np.nan)
            f1_dict[col].append(np.nan)
            balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            auroc_dict[tuple(col)].append(np.nan)
            auprc_dict[tuple(col)].append(np.nan)
            f1_dict[tuple(col)].append(np.nan)
            balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        model = sm.Logit(model_df[pheno], model_df[predictors]).fit()
        test_df = test[all_cols].dropna()
        y_prob_cont = model.predict(test_df[predictors])
        y_prob_bin = (y_prob_cont >= 0.5).astype(int)
        auroc = roc_auc_score(test_df[pheno], y_prob_cont)
        auprc = average_precision_score(test_df[pheno], y_prob_cont)
        f1 = f1_score(test_df[pheno], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
        if isinstance(col, str):
            auroc_dict[col].append(auroc)
            auprc_dict[col].append(auprc)
            f1_dict[col].append(f1)
            balanced_acc_dict[col].append(balanced_acc)
        elif isinstance(col, list):
            auroc_dict[tuple(col)].append(auroc)
            auprc_dict[tuple(col)].append(auprc)
            f1_dict[tuple(col)].append(f1)
            balanced_acc_dict[tuple(col)].append(balanced_acc)
        else:
            raise ValueError("conditions not met")

# make output dfs
auroc_df = pd.DataFrame.from_dict(auroc_dict, orient = 'index', columns = [colname])
auprc_df = pd.DataFrame.from_dict(auprc_dict, orient = 'index', columns = [colname])
f1_df = pd.DataFrame.from_dict(f1_dict, orient = 'index', columns = [colname])
balanced_acc_df = pd.DataFrame.from_dict(balanced_acc_dict, orient = 'index', columns = [colname])

# export dfs
auroc_df.to_csv((output_dir + pheno + '.AUROC.' + colname + '.txt'), sep = '\t')
auprc_df.to_csv((output_dir + pheno + '.AUPRC.' + colname + '.txt'), sep = '\t')
f1_df.to_csv((output_dir + pheno + '.F1_SCORE.' + colname + '.txt'), sep = '\t')
balanced_acc_df.to_csv((output_dir + pheno + '.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
