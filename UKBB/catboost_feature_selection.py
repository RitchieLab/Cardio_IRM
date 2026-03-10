# load subprocess and sys
import subprocess
import sys

# install packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

# load packages
import pandas as pd
import argparse as ap
from catboost import CatBoostClassifier, Pool
import numpy as np

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--pheno', required = True, help = 'phenotype')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
pheno = args.pheno

# read in input file
input = pd.read_csv(input_filename)

# downsample CAD to get more distributed portions of case/control status
if pheno == 'CAD':
    case = input[input['CAD'] == 1]
    control = input[input['CAD'] == 0]
    sample_size = len(case.index) - 14216
    case_sample = case.sample(n = sample_size, random_state = iter)
    input = pd.concat([case_sample, control], axis = 0)

# remove pheno and person id
input_sub = input.drop(columns = ['eid', pheno])
print(input_sub)

# create cat cols list
cat_cols = input_sub.columns[~input_sub.columns.str.contains('INV_NORMAL|AGE')].tolist()
input_sub[cat_cols] = input_sub[cat_cols].fillna('nan').astype(str)
print(cat_cols)

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# create train pool
train_pool = Pool(input_sub, input[[pheno]], cat_features = cat_cols)

# create model
model = CatBoostClassifier(
    loss_function = "Logloss",
    eval_metric = "AUC",
    random_seed = iter,
    verbose = 200
)

# fit model
model.fit(train_pool)

# get gain feature importance
importance = model.get_feature_importance(train_pool)
fi_df = pd.DataFrame({"feature" : input_sub.columns, colname : importance})
print(fi_df.sort_values(by = colname, ascending = False))  

# get shap feature importance
shap_values = model.get_feature_importance(train_pool, type = "ShapValues")
shap_importance = shap_values[:, :-1]

# mean absolute SHAP per feature
shap_mean = np.abs(shap_importance).mean(axis = 0)
shap_df = (pd.DataFrame({"feature": input_sub.columns, colname: shap_mean}))
print(shap_df.sort_values(by = colname, ascending = False))

# export dfs
fi_df.to_csv((pheno + '.catboost.gain.feature_importance.' + colname + '.txt'), sep = '\t')
shap_df.to_csv((pheno + '.catboost.shap.feature_importance.' + colname + '.txt'), sep = '\t')
