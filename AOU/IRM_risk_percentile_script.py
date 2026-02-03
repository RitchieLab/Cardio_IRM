# load packages
import pandas as pd
import argparse as ap
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--sig', required = True, help = 'significant filename')
    
    parser.add_argument('--important', required = True, help = 'important filename')
    
    parser.add_argument('--beta', required = True, help = 'beta filename')
    
    parser.add_argument('--pheno', required = True, help = 'phenotype')
    
    parser.add_argument('--cutoff', required = True, help = 'percentile cutoff for the phenotype, should be an integer')
    
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
cutoff = int(args.cutoff)
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
                 ['PGS', 'PXS_WEIGHTED_SUM']]

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

# apply SMOTE
x = no_train.drop(columns = [pheno])
y = no_train[[pheno]]
x_resampled, y_resampled = SMOTE(random_state = iter).fit_resample(x, y)
no_train = pd.concat([x_resampled, y_resampled], axis = 1)

# compute percentile case classifications
new_cols = {}

def inverse_normal_transform(x):
    """Applies rank-based inverse normal transformation."""
    ranks = x.rank(method='average', na_option='keep')
    n = ranks.notna().sum()
    transformed = norm.ppf((ranks - 0.5) / n)
    return transformed

for col in eval_col_list:
    if isinstance(col, str):
        inv_norm_col = col + '_INV_NORMAL'
        new_cols[inv_norm_col] = inverse_normal_transform(no_train[col])
        ntile_col = col + '_ntile'
        ntile_case_col = ntile_col + '_case'
        if col == 'PGS':
            new_cols[ntile_col] = 100 * norm.cdf(no_train[col])
        else:
            new_cols[ntile_col] = 100 * norm.cdf(new_cols[inv_norm_col])
        new_cols[ntile_case_col] = np.where(new_cols[ntile_col] >= cutoff, 1, 0)
    elif isinstance(col, list) and len(col) == 2:
        c1, c2 = col
        ntile_col1 = c1 + '_ntile'
        ntile_col2 = c2 + '_ntile'
        #ntile_case_col = '_'.join([c1, c2])
        ntile_case_col = str(col) + '_ntile_case'
        #print(ntile_case_col)
        new_cols[ntile_case_col] = np.where(((new_cols[ntile_col1] >= cutoff) & (new_cols[ntile_col2] >= cutoff)), 1, 0)
    elif isinstance(col, list) and len(col) == 3:
        c1, c2, c3 = col
        ntile_col1 = c1 + '_ntile'
        ntile_col2 = c2 + '_ntile'
        ntile_col3 = c3 + '_ntile'
        #ntile_case_col = '_'.join([c1, c2, c3])
        ntile_case_col = str(col) + '_ntile_case'
        #print(ntile_case_col)
        new_cols[ntile_case_col] = np.where(((new_cols[ntile_col1] >= cutoff) & (new_cols[ntile_col2] >= cutoff) & (new_cols[ntile_col3] >= cutoff)), 1, 0)
    else:
        raise ValueError("conditions not met")
no_train = pd.concat([no_train, pd.DataFrame(new_cols, index = no_train.index)], axis = 1)
no_train.columns = no_train.columns.str.replace("'", "")
no_train.columns = no_train.columns.str.replace("[", "")
no_train.columns = no_train.columns.str.replace("]", "")
no_train.columns = no_train.columns.str.replace(",", " +")

ntile_columns = no_train.columns[no_train.columns.str.contains('ntile_case')]
eval_col_list.extend(ntile_columns.to_list())

for col in ['PGS', 'CRS_SUM', 'CRS_WEIGHTED_SUM', 'PXS_SUM', 'PXS_WEIGHTED_SUM']:
    sns.kdeplot(no_train[col], fill = True)
    plt.title(col)
    plt.tight_layout()
    plt.savefig(output_dir + pheno + '.' + col + ".distribution_plot.png", dpi = 300)
    plt.clf()
    
    inv_norm_col = col + '_INV_NORMAL'
    if col != 'PGS':
        sns.kdeplot(no_train[inv_norm_col], fill = True)
        plt.title(inv_norm_col)
        plt.tight_layout()
        plt.savefig(output_dir + pheno + '.' + inv_norm_col + ".distribution_plot.png", dpi = 300)
        plt.clf()
    
    ntile_col = col + '_ntile'
    sns.kdeplot(no_train[ntile_col], fill = True)
    plt.title(ntile_col)
    plt.tight_layout()
    plt.savefig(output_dir + pheno + '.' + ntile_col + ".distribution_plot.png", dpi = 300)
    
    case_col = ntile_col + '_case'
    sns.countplot(data = no_train, x = case_col, hue = case_col)
    plt.title(ntile_col)
    plt.tight_layout()
    plt.savefig(output_dir + pheno + '.' + case_col + ".distribution_plot.png", dpi = 300)
    plt.clf()
    
sys.exit()

# make age col
age_col = 'AGE_' + pheno

# create empty dictionaries
or_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
ci_lower_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}
ci_upper_dict = {tuple(item) if isinstance(item, list) else item: [] for item in eval_col_list}

# evaluate models
for col in eval_col_list:
    print(col)
    if isinstance(col, str):
        model_df = no_train[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = no_train[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            or_dict[col].append(np.nan)
            ci_lower_dict[col].append(np.nan)
            ci_upper[col].append(np.nan)
        elif isinstance(col, list):
            or_dict[col].append(np.nan)
            ci_lower_dict[col].append(np.nan)
            ci_upper[col].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        model = sm.Logit(model_df[pheno], model_df[predictors]).fit()
        results = pd.concat([np.exp(model.params), np.exp(model.conf_int())], axis = 1)
        results = results.iloc[[2], :]
        results.columns = ['OR', 'CI_lower', 'CI_upper']
        odds_ratio = results.reset_index(drop = True).loc[0, 'OR']
        ci_lower = results.reset_index(drop = True).loc[0, 'CI_lower']
        ci_upper = results.reset_index(drop = True).loc[0, 'CI_upper']
        
        if isinstance(col, str):
            or_dict[col].append(odds_ratio)
            ci_lower_dict[col].append(ci_lower)
            ci_upper_dict[col].append(ci_upper)
        elif isinstance(col, list):
            or_dict[tuple(col)].append(odds_ratio)
            ci_lower_dict[tuple(col)].append(ci_lower)
            ci_upper_dict[tuple(col)].append(ci_upper)
        else:
            raise ValueError("conditions not met")


# make output dfs
or_df = pd.DataFrame.from_dict(or_dict, orient = 'index', columns = [colname])
ci_lower_df = pd.DataFrame.from_dict(ci_lower_dict, orient = 'index', columns = [colname])
ci_upper_df = pd.DataFrame.from_dict(ci_upper_dict, orient = 'index', columns = [colname])

# export dfs
or_df.to_csv((output_dir + pheno + '.OR.' + colname + '.txt'), sep = '\t')
ci_lower_df.to_csv((output_dir + pheno + '.CI_lower.' + colname + '.txt'), sep = '\t')
ci_upper_df.to_csv((output_dir + pheno + '.CI_upper.' + colname + '.txt'), sep = '\t')

# make plot input
plot_input = pd.concat([or_df, ci_lower_df, ci_upper_df], axis = 1)
plot_input.columns = ['OR', 'CI_lower', 'CI_upper']
plot_input.index = plot_input.index.map(lambda x: str(x).replace("'", ""))
plot_input.index = plot_input.index.map(lambda x: x.replace("(", ""))
plot_input.index = plot_input.index.map(lambda x: x.replace(")", ""))
plot_input.index = plot_input.index.map(lambda x: x.replace(",", " +"))
plot_input['TERM'] = plot_input.index

print(plot_input)
sys.exit()

# make plot
plt.errorbar(
    x = plot_input['OR'],
    y = plot_input['TERM'],
    xerr = [plot_input['OR'] - plot_input['CI_lower'], plot_input['CI_upper'] - plot_input['OR']],
    fmt = 'o',
    color = 'blue',
    capsize = 4)
plt.axvline(x = 1, color = 'red', linestyle = '--', linewidth = 1)
plt.hlines(y = range(len(plot_input)), xmin = 0, xmax = 5, colors = 'lightgray', linestyles = '--', linewidth = 0.8)
plt.tight_layout()
plt.savefig(output_dir + pheno + ".forest_plot.png", dpi = 300)
