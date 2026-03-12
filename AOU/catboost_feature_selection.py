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
    
    parser.add_argument('--output_dir', required = True, help = 'output filename')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
pheno = args.pheno
output_dir = args.output_dir

# read in input file
input = pd.read_csv(input_filename)

# remove pheno and person id
input_sub = input.drop(columns = ['person_id', pheno])
print(input_sub)

# create cat cols list
cat_cols = input_sub.columns[~input_sub.columns.str.contains('INV_NORMAL|AGE')].tolist()
input_sub[cat_cols] = input_sub[cat_cols].fillna('nan').astype(str)
print(cat_cols)
print(input_sub['NEIGHBORHOOD_DRUG_USE_SCALE'].unique())

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
fi_df.to_csv((output_dir + pheno + '.catboost.gain.feature_importance.' + colname + '.txt'), sep = '\t')
shap_df.to_csv((output_dir + pheno + '.catboost.shap.feature_importance.' + colname + '.txt'), sep = '\t')
