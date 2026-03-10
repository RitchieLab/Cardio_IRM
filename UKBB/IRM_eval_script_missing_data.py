# load subprocess and sys
import subprocess
import sys

# install packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])

# load packages
import pandas as pd
import argparse as ap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.datasets import make_classification
import statsmodels.api as sm

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--weight', required = True, help = 'weight filename')
    
    parser.add_argument('--pheno', required = True, help = 'phenotype')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
weight_filename = args.weight
pheno = args.pheno

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# read in input files
input = pd.read_csv(input_filename)
weight = pd.read_csv(weight_filename, sep = '\t', index_col = 0)

# downsample females for sex proportions are similar
if pheno == 'AFIB':
    female = input[input['SEX'] == 2]
    male = input[input['SEX'] == 1]
    sample_size = len(female.index) - 45764
    female_sample = female.sample(n = sample_size, random_state = iter)
    input = pd.concat([female_sample, male], axis = 0)

# remove missing from AFIB DF
if pheno == 'AFIB':
    input_c2hest = input.dropna(subset = ['AFIB_C2HEST_score'])
    prevent_cols = input.columns[input.columns.str.contains('PREVENT')].tolist()
    input = input.dropna(subset = prevent_cols)

# split dataset
train = input.sample(frac = 0.7, random_state = iter)
print(train[pheno].value_counts(dropna = False))
test = input.drop(train.index)
print(test[pheno].value_counts(dropna = False))   

if pheno == 'AFIB':
    train_c2hest = input_c2hest.sample(frac = 0.7, random_state = iter)
    test_c2hest = input_c2hest.drop(train_c2hest.index)
    print(train_c2hest[pheno].value_counts(dropna = False))
    print(test_c2hest[pheno].value_counts(dropna = False))

# create PXS column list
pxs_cols = weight.index.tolist()
pxs_cols.remove(('AGE_' + pheno))
pxs_cols.remove('SEX')
#print(len(pxs_cols))

# create eval column list
total_cvd_eval_col_list = [('PGS_' + pheno),
                                   (pheno + "_PREVENT_CRS_total_cvd"),
                                   'PXS_AVG',
                                   'PXS_WEIGHTED_AVG',
                                   [('PGS_' + pheno), (pheno + "_PREVENT_CRS_total_cvd"), 'PXS_AVG'],
                                   [(pheno + "_PREVENT_CRS_total_cvd"), 'PXS_AVG'],
                                   [('PGS_' + pheno), (pheno + "_PREVENT_CRS_total_cvd")],
                                   [('PGS_' + pheno), 'PXS_AVG'],
                                   [('PGS_' + pheno), (pheno + "_PREVENT_CRS_total_cvd"), 'PXS_WEIGHTED_AVG'],
                                   [(pheno + "_PREVENT_CRS_total_cvd"), 'PXS_WEIGHTED_AVG'],
                                   [('PGS_' + pheno), 'PXS_WEIGHTED_AVG']]

ascvd_eval_col_list = [('PGS_' + pheno),
                               (pheno + "_PREVENT_CRS_ascvd"),
                               'PXS_AVG',
                               'PXS_WEIGHTED_AVG',
                               [('PGS_' + pheno), (pheno + "_PREVENT_CRS_ascvd"), 'PXS_AVG'],
                               [(pheno + "_PREVENT_CRS_ascvd"), 'PXS_AVG'],
                               [('PGS_' + pheno), (pheno + "_PREVENT_CRS_ascvd")],
                               [('PGS_' + pheno), 'PXS_AVG'],
                               [('PGS_' + pheno), (pheno + "_PREVENT_CRS_ascvd"), 'PXS_WEIGHTED_AVG'],
                               [(pheno + "_PREVENT_CRS_ascvd"), 'PXS_WEIGHTED_AVG'],
                               [('PGS_' + pheno), 'PXS_WEIGHTED_AVG']]

heart_failure_eval_col_list = [('PGS_' + pheno),
                                       (pheno + "_PREVENT_CRS_heart_failure"),
                                       'PXS_AVG',
                                       'PXS_WEIGHTED_AVG',
                                       [('PGS_' + pheno), (pheno + "_PREVENT_CRS_heart_failure"), 'PXS_AVG'],
                                       [(pheno + "_PREVENT_CRS_heart_failure"), 'PXS_AVG'],
                                       [('PGS_' + pheno), (pheno + "_PREVENT_CRS_heart_failure")],
                                       [('PGS_' + pheno), 'PXS_AVG'],
                                       [('PGS_' + pheno), (pheno + "_PREVENT_CRS_heart_failure"), 'PXS_WEIGHTED_AVG'],
                                       [(pheno + "_PREVENT_CRS_heart_failure"), 'PXS_WEIGHTED_AVG'],
                                       [('PGS_' + pheno), 'PXS_WEIGHTED_AVG']]

stroke_eval_col_list = [('PGS_' + pheno),
                                (pheno + "_PREVENT_CRS_stroke"),
                                'PXS_AVG',
                                'PXS_WEIGHTED_AVG',
                                [('PGS_' + pheno), (pheno + "_PREVENT_CRS_stroke"), 'PXS_AVG'],
                                [(pheno + "_PREVENT_CRS_stroke"), 'PXS_AVG'],
                                [('PGS_' + pheno), (pheno + "_PREVENT_CRS_stroke")],
                                [('PGS_' + pheno), 'PXS_AVG'],
                                [('PGS_' + pheno), (pheno + "_PREVENT_CRS_stroke"), 'PXS_WEIGHTED_AVG'],
                                [(pheno + "_PREVENT_CRS_stroke"), 'PXS_WEIGHTED_AVG'],
                                [('PGS_' + pheno), 'PXS_WEIGHTED_AVG']]

chd_eval_col_list = [('PGS_' + pheno),
                             (pheno + "_PREVENT_CRS_chd"),
                             'PXS_AVG',
                             'PXS_WEIGHTED_AVG',
                             [('PGS_' + pheno), (pheno + "_PREVENT_CRS_chd"), 'PXS_AVG'],
                             [(pheno + "_PREVENT_CRS_chd"), 'PXS_AVG'],
                             [('PGS_' + pheno), (pheno + "_PREVENT_CRS_chd")],
                             [('PGS_' + pheno), 'PXS_AVG'],
                             [('PGS_' + pheno), (pheno + "_PREVENT_CRS_chd"), 'PXS_WEIGHTED_AVG'],
                             [(pheno + "_PREVENT_CRS_chd"), 'PXS_WEIGHTED_AVG'],
                             [('PGS_' + pheno), 'PXS_WEIGHTED_AVG']]

c2hest_eval_col_list = [('PGS_' + pheno),
                        (pheno + "_C2HEST_score"),
                        'PXS_AVG',
                        'PXS_WEIGHTED_AVG',
                        [('PGS_' + pheno), (pheno + "_C2HEST_score"), 'PXS_AVG'],
                        [(pheno + "_C2HEST_score"), 'PXS_AVG'],
                        [('PGS_' + pheno), (pheno + "_C2HEST_score")],
                        [('PGS_' + pheno), 'PXS_AVG'],
                        [('PGS_' + pheno), (pheno + "_C2HEST_score"), 'PXS_WEIGHTED_AVG'],
                        [(pheno + "_C2HEST_score"), 'PXS_WEIGHTED_AVG'],
                        [('PGS_' + pheno), 'PXS_WEIGHTED_AVG']]

# compute integrated scores
train['PXS_AVG'] = train[pxs_cols].mean(axis = 1, skipna = True)
test['PXS_AVG'] = test[pxs_cols].mean(axis = 1, skipna = True)

weights = weight['WEIGHT']

w = pd.Series(0, index = train.columns, dtype = float)
w.loc[pxs_cols] = weights.loc[pxs_cols]
weighted_sum = (train * w).sum(axis = 1)
effective_weight_sum = (train.notna() * w).sum(axis = 1)
weighted_avg = weighted_sum / effective_weight_sum
train['PXS_WEIGHTED_AVG'] = weighted_avg

w = pd.Series(0, index = test.columns, dtype = float)
w.loc[pxs_cols] = weights.loc[pxs_cols]
weighted_sum = (test * w).sum(axis = 1)
effective_weight_sum = (test.notna() * w).sum(axis = 1)
weighted_avg = weighted_sum / effective_weight_sum
test['PXS_WEIGHTED_AVG'] = weighted_avg

if pheno == 'AFIB':
    train_c2hest['PXS_AVG'] = train_c2hest[pxs_cols].mean(axis = 1, skipna = True)
    test_c2hest['PXS_AVG'] = test_c2hest[pxs_cols].mean(axis = 1, skipna = True)

    weights = weight['WEIGHT']

    w = pd.Series(0, index = train_c2hest.columns, dtype = float)
    w.loc[pxs_cols] = weights.loc[pxs_cols]
    weighted_sum = (train_c2hest * w).sum(axis = 1)
    effective_weight_sum = (train_c2hest.notna() * w).sum(axis = 1)
    weighted_avg = weighted_sum / effective_weight_sum
    train_c2hest['PXS_WEIGHTED_AVG'] = weighted_avg

    w = pd.Series(0, index = test_c2hest.columns, dtype = float)
    w.loc[pxs_cols] = weights.loc[pxs_cols]
    weighted_sum = (test_c2hest * w).sum(axis = 1)
    effective_weight_sum = (test_c2hest.notna() * w).sum(axis = 1)
    weighted_avg = weighted_sum / effective_weight_sum
    test_c2hest['PXS_WEIGHTED_AVG'] = weighted_avg

# make age col
age_col = 'AGE_' + pheno

# create empty dictionaries
total_cvd_auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in total_cvd_eval_col_list}
total_cvd_auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in total_cvd_eval_col_list}
total_cvd_f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in total_cvd_eval_col_list}
total_cvd_balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in total_cvd_eval_col_list}
total_cvd_beta_list = []
total_cvd_pval_list = []

ascvd_auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in ascvd_eval_col_list}
ascvd_auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in ascvd_eval_col_list}
ascvd_f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in ascvd_eval_col_list}
ascvd_balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in ascvd_eval_col_list}
ascvd_beta_list = []
ascvd_pval_list = []

heart_failure_auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in heart_failure_eval_col_list}
heart_failure_auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in heart_failure_eval_col_list}
heart_failure_f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in heart_failure_eval_col_list}
heart_failure_balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in heart_failure_eval_col_list}
heart_failure_beta_list = []
heart_failure_pval_list = []

stroke_auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in stroke_eval_col_list}
stroke_auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in stroke_eval_col_list}
stroke_f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in stroke_eval_col_list}
stroke_balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in stroke_eval_col_list}
stroke_beta_list = []
stroke_pval_list = []

chd_auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in chd_eval_col_list}
chd_auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in chd_eval_col_list}
chd_f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in chd_eval_col_list}
chd_balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in chd_eval_col_list}
chd_beta_list = []
chd_pval_list = []

c2hest_auroc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in c2hest_eval_col_list}
c2hest_auprc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in c2hest_eval_col_list}
c2hest_f1_dict = {tuple(item) if isinstance(item, list) else item: [] for item in c2hest_eval_col_list}
c2hest_balanced_acc_dict = {tuple(item) if isinstance(item, list) else item: [] for item in c2hest_eval_col_list}
c2hest_beta_list = []
c2hest_pval_list = []



# evaluate models
for col in total_cvd_eval_col_list:
    if isinstance(col, str):
        model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            total_cvd_auroc_dict[col].append(np.nan)
            total_cvd_auprc_dict[col].append(np.nan)
            total_cvd_f1_dict[col].append(np.nan)
            total_cvd_balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            total_cvd_auroc_dict[tuple(col)].append(np.nan)
            total_cvd_auprc_dict[tuple(col)].append(np.nan)
            total_cvd_f1_dict[tuple(col)].append(np.nan)
            total_cvd_balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        # get effect size and sig
        X = model_df[predictors]
        X = sm.add_constant(X)
        y = model_df[pheno]
        logit = sm.Logit(y, X).fit()
        betas = logit.params
        pvals = logit.pvalues
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        auroc = roc_auc_score(test_df[pheno], y_prob_cont)
        auprc = average_precision_score(test_df[pheno], y_prob_cont)
        f1 = f1_score(test_df[pheno], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
        if isinstance(col, str):
            total_cvd_auroc_dict[col].append(auroc)
            total_cvd_auprc_dict[col].append(auprc)
            total_cvd_f1_dict[col].append(f1)
            total_cvd_balanced_acc_dict[col].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, col : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, col : pvals.values}).set_index('feature', drop = True)
        elif isinstance(col, list):
            total_cvd_auroc_dict[tuple(col)].append(auroc)
            total_cvd_auprc_dict[tuple(col)].append(auprc)
            total_cvd_f1_dict[tuple(col)].append(f1)
            total_cvd_balanced_acc_dict[tuple(col)].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, tuple(col) : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, tuple(col) : pvals.values}).set_index('feature', drop = True)
        else:
            raise ValueError("conditions not met")
        total_cvd_beta_list.append(beta)
        total_cvd_pval_list.append(pval)
            
for col in ascvd_eval_col_list:
    if isinstance(col, str):
        model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            ascvd_auroc_dict[col].append(np.nan)
            ascvd_auprc_dict[col].append(np.nan)
            ascvd_f1_dict[col].append(np.nan)
            ascvd_balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            ascvd_auroc_dict[tuple(col)].append(np.nan)
            ascvd_auprc_dict[tuple(col)].append(np.nan)
            ascvd_f1_dict[tuple(col)].append(np.nan)
            ascvd_balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        # get effect size and sig
        X = model_df[predictors]
        X = sm.add_constant(X)
        y = model_df[pheno]
        logit = sm.Logit(y, X).fit()
        betas = logit.params
        pvals = logit.pvalues
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        auroc = roc_auc_score(test_df[pheno], y_prob_cont)
        auprc = average_precision_score(test_df[pheno], y_prob_cont)
        f1 = f1_score(test_df[pheno], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
        if isinstance(col, str):
            ascvd_auroc_dict[col].append(auroc)
            ascvd_auprc_dict[col].append(auprc)
            ascvd_f1_dict[col].append(f1)
            ascvd_balanced_acc_dict[col].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, col : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, col : pvals.values}).set_index('feature', drop = True)
        elif isinstance(col, list):
            ascvd_auroc_dict[tuple(col)].append(auroc)
            ascvd_auprc_dict[tuple(col)].append(auprc)
            ascvd_f1_dict[tuple(col)].append(f1)
            ascvd_balanced_acc_dict[tuple(col)].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, tuple(col) : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, tuple(col) : pvals.values}).set_index('feature', drop = True)
        else:
            raise ValueError("conditions not met")
        
        ascvd_beta_list.append(beta)
        ascvd_pval_list.append(pval)

for col in heart_failure_eval_col_list:
    if isinstance(col, str):
        model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            heart_failure_auroc_dict[col].append(np.nan)
            heart_failure_auprc_dict[col].append(np.nan)
            heart_failure_f1_dict[col].append(np.nan)
            heart_failure_balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            heart_failure_auroc_dict[tuple(col)].append(np.nan)
            heart_failure_auprc_dict[tuple(col)].append(np.nan)
            heart_failure_f1_dict[tuple(col)].append(np.nan)
            heart_failure_balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        # get effect size and sig
        X = model_df[predictors]
        X = sm.add_constant(X)
        y = model_df[pheno]
        logit = sm.Logit(y, X).fit()
        betas = logit.params
        pvals = logit.pvalues
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        auroc = roc_auc_score(test_df[pheno], y_prob_cont)
        auprc = average_precision_score(test_df[pheno], y_prob_cont)
        f1 = f1_score(test_df[pheno], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
        if isinstance(col, str):
            heart_failure_auroc_dict[col].append(auroc)
            heart_failure_auprc_dict[col].append(auprc)
            heart_failure_f1_dict[col].append(f1)
            heart_failure_balanced_acc_dict[col].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, col : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, col : pvals.values}).set_index('feature', drop = True)
        elif isinstance(col, list):
            heart_failure_auroc_dict[tuple(col)].append(auroc)
            heart_failure_auprc_dict[tuple(col)].append(auprc)
            heart_failure_f1_dict[tuple(col)].append(f1)
            heart_failure_balanced_acc_dict[tuple(col)].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, tuple(col) : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, tuple(col) : pvals.values}).set_index('feature', drop = True)
        else:
            raise ValueError("conditions not met")    
        heart_failure_beta_list.append(beta)
        heart_failure_pval_list.append(pval)
            
for col in stroke_eval_col_list:
    if isinstance(col, str):
        model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            stroke_auroc_dict[col].append(np.nan)
            stroke_auprc_dict[col].append(np.nan)
            stroke_f1_dict[col].append(np.nan)
            stroke_balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            stroke_auroc_dict[tuple(col)].append(np.nan)
            stroke_auprc_dict[tuple(col)].append(np.nan)
            stroke_f1_dict[tuple(col)].append(np.nan)
            stroke_balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        # get effect size and sig
        X = model_df[predictors]
        X = sm.add_constant(X)
        y = model_df[pheno]
        logit = sm.Logit(y, X).fit()
        betas = logit.params
        pvals = logit.pvalues
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        auroc = roc_auc_score(test_df[pheno], y_prob_cont)
        auprc = average_precision_score(test_df[pheno], y_prob_cont)
        f1 = f1_score(test_df[pheno], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
        if isinstance(col, str):
            stroke_auroc_dict[col].append(auroc)
            stroke_auprc_dict[col].append(auprc)
            stroke_f1_dict[col].append(f1)
            stroke_balanced_acc_dict[col].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, col : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, col : pvals.values}).set_index('feature', drop = True)
        elif isinstance(col, list):
            stroke_auroc_dict[tuple(col)].append(auroc)
            stroke_auprc_dict[tuple(col)].append(auprc)
            stroke_f1_dict[tuple(col)].append(f1)
            stroke_balanced_acc_dict[tuple(col)].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, tuple(col) : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, tuple(col) : pvals.values}).set_index('feature', drop = True)
        else:
            raise ValueError("conditions not met") 
        
        stroke_beta_list.append(beta)
        stroke_pval_list.append(pval)
            
for col in chd_eval_col_list:
    if isinstance(col, str):
        model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
        predictors = [age_col, 'SEX'] + [col]
    elif isinstance(col, list):
        model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
        predictors = [age_col, 'SEX'] + col
    else:
        raise ValueError("conditions not met")
        
    all_cols = [pheno] + predictors
        
    if len(model_df.index) == 0:
        print('skipping, all values are zero')
        if isinstance(col, str):  
            chd_auroc_dict[col].append(np.nan)
            chd_auprc_dict[col].append(np.nan)
            chd_f1_dict[col].append(np.nan)
            chd_balanced_acc_dict[col].append(np.nan)
        elif isinstance(col, list):
            chd_auroc_dict[tuple(col)].append(np.nan)
            chd_auprc_dict[tuple(col)].append(np.nan)
            chd_f1_dict[tuple(col)].append(np.nan)
            chd_balanced_acc_dict[tuple(col)].append(np.nan)
        else:
            raise ValueError("conditions not met")
        
    else:
        # get effect size and sig
        X = model_df[predictors]
        X = sm.add_constant(X)
        y = model_df[pheno]
        logit = sm.Logit(y, X).fit()
        betas = logit.params
        pvals = logit.pvalues
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        #print(y_prob_cont.min(), y_prob_cont.max())
        auroc = roc_auc_score(test_df[pheno], y_prob_cont)
        auprc = average_precision_score(test_df[pheno], y_prob_cont)
        f1 = f1_score(test_df[pheno], y_prob_bin)
        balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
        if isinstance(col, str):
            chd_auroc_dict[col].append(auroc)
            chd_auprc_dict[col].append(auprc)
            chd_f1_dict[col].append(f1)
            chd_balanced_acc_dict[col].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, col : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, col : pvals.values}).set_index('feature', drop = True)
        elif isinstance(col, list):
            chd_auroc_dict[tuple(col)].append(auroc)
            chd_auprc_dict[tuple(col)].append(auprc)
            chd_f1_dict[tuple(col)].append(f1)
            chd_balanced_acc_dict[tuple(col)].append(balanced_acc)
            beta = pd.DataFrame({"feature" : betas.index, tuple(col) : betas.values}).set_index('feature', drop = True)
            pval = pd.DataFrame({"feature" : betas.index, tuple(col) : pvals.values}).set_index('feature', drop = True)
        else:
            raise ValueError("conditions not met")
        chd_beta_list.append(beta)
        chd_pval_list.append(pval)

if pheno == 'AFIB':
    for col in c2hest_eval_col_list:
        if isinstance(col, str):
            model_df = train_c2hest[[pheno, age_col, 'SEX'] + [col]].dropna()
            predictors = [age_col, 'SEX'] + [col]
        elif isinstance(col, list):
            model_df = train_c2hest[[pheno, age_col, 'SEX'] + col].dropna()
            predictors = [age_col, 'SEX'] + col
        else:
            raise ValueError("conditions not met")
        
        all_cols = [pheno] + predictors
        
        if len(model_df.index) == 0:
            print('skipping, all values are zero')
            if isinstance(col, str):  
                c2hest_auroc_dict[col].append(np.nan)
                c2hest_auprc_dict[col].append(np.nan)
                c2hest_f1_dict[col].append(np.nan)
                c2hest_balanced_acc_dict[col].append(np.nan)
            elif isinstance(col, list):
                c2hest_auroc_dict[tuple(col)].append(np.nan)
                c2hest_auprc_dict[tuple(col)].append(np.nan)
                c2hest_f1_dict[tuple(col)].append(np.nan)
                c2hest_balanced_acc_dict[tuple(col)].append(np.nan)
            else:
                raise ValueError("conditions not met")
        
        else:
            # get effect size and sig
            X = model_df[predictors]
            X = sm.add_constant(X)
            y = model_df[pheno]
            logit = sm.Logit(y, X).fit()
            betas = logit.params
            pvals = logit.pvalues
            
            ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
            model = ridge.fit(model_df[predictors], model_df[[pheno]])
            test_df = test_c2hest[all_cols].dropna()
            y_prob_bin = model.predict(test_df[predictors])
            y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
            auroc = roc_auc_score(test_df[pheno], y_prob_cont)
            auprc = average_precision_score(test_df[pheno], y_prob_cont)
            f1 = f1_score(test_df[pheno], y_prob_bin)
            balanced_acc = balanced_accuracy_score(test_df[pheno], y_prob_bin)
            if isinstance(col, str):
                c2hest_auroc_dict[col].append(auroc)
                c2hest_auprc_dict[col].append(auprc)
                c2hest_f1_dict[col].append(f1)
                c2hest_balanced_acc_dict[col].append(balanced_acc)
                beta = pd.DataFrame({"feature" : betas.index, col : betas.values}).set_index('feature', drop = True)
                pval = pd.DataFrame({"feature" : betas.index, col : pvals.values}).set_index('feature', drop = True)
            elif isinstance(col, list):
                c2hest_auroc_dict[tuple(col)].append(auroc)
                c2hest_auprc_dict[tuple(col)].append(auprc)
                c2hest_f1_dict[tuple(col)].append(f1)
                c2hest_balanced_acc_dict[tuple(col)].append(balanced_acc)
                beta = pd.DataFrame({"feature" : betas.index, tuple(col) : betas.values}).set_index('feature', drop = True)
                pval = pd.DataFrame({"feature" : betas.index, tuple(col) : pvals.values}).set_index('feature', drop = True)
            else:
                raise ValueError("conditions not met")
            c2hest_beta_list.append(beta)
            c2hest_pval_list.append(pval)
            
# make output dfs
total_cvd_auroc_df = pd.DataFrame.from_dict(total_cvd_auroc_dict, orient = 'index', columns = [colname])
total_cvd_auprc_df = pd.DataFrame.from_dict(total_cvd_auprc_dict, orient = 'index', columns = [colname])
total_cvd_f1_df = pd.DataFrame.from_dict(total_cvd_f1_dict, orient = 'index', columns = [colname])
total_cvd_balanced_acc_df = pd.DataFrame.from_dict(total_cvd_balanced_acc_dict, orient = 'index', columns = [colname])
total_cvd_beta_df = pd.concat(total_cvd_beta_list, axis = 1)
total_cvd_pval_df = pd.concat(total_cvd_pval_list, axis = 1)

ascvd_auroc_df = pd.DataFrame.from_dict(ascvd_auroc_dict, orient = 'index', columns = [colname])
ascvd_auprc_df = pd.DataFrame.from_dict(ascvd_auprc_dict, orient = 'index', columns = [colname])
ascvd_f1_df = pd.DataFrame.from_dict(ascvd_f1_dict, orient = 'index', columns = [colname])
ascvd_balanced_acc_df = pd.DataFrame.from_dict(ascvd_balanced_acc_dict, orient = 'index', columns = [colname])
ascvd_beta_df = pd.concat(ascvd_beta_list, axis = 1)
ascvd_pval_df = pd.concat(ascvd_pval_list, axis = 1)

heart_failure_auroc_df = pd.DataFrame.from_dict(heart_failure_auroc_dict, orient = 'index', columns = [colname])
heart_failure_auprc_df = pd.DataFrame.from_dict(heart_failure_auprc_dict, orient = 'index', columns = [colname])
heart_failure_f1_df = pd.DataFrame.from_dict(heart_failure_f1_dict, orient = 'index', columns = [colname])
heart_failure_balanced_acc_df = pd.DataFrame.from_dict(heart_failure_balanced_acc_dict, orient = 'index', columns = [colname])
heart_failure_beta_df = pd.concat(heart_failure_beta_list, axis = 1)
heart_failure_pval_df = pd.concat(heart_failure_pval_list, axis = 1)

stroke_auroc_df = pd.DataFrame.from_dict(stroke_auroc_dict, orient = 'index', columns = [colname])
stroke_auprc_df = pd.DataFrame.from_dict(stroke_auprc_dict, orient = 'index', columns = [colname])
stroke_f1_df = pd.DataFrame.from_dict(stroke_f1_dict, orient = 'index', columns = [colname])
stroke_balanced_acc_df = pd.DataFrame.from_dict(stroke_balanced_acc_dict, orient = 'index', columns = [colname])
stroke_beta_df = pd.concat(stroke_beta_list, axis = 1)
stroke_pval_df = pd.concat(stroke_pval_list, axis = 1)

chd_auroc_df = pd.DataFrame.from_dict(chd_auroc_dict, orient = 'index', columns = [colname])
chd_auprc_df = pd.DataFrame.from_dict(chd_auprc_dict, orient = 'index', columns = [colname])
chd_f1_df = pd.DataFrame.from_dict(chd_f1_dict, orient = 'index', columns = [colname])
chd_balanced_acc_df = pd.DataFrame.from_dict(chd_balanced_acc_dict, orient = 'index', columns = [colname])
chd_beta_df = pd.concat(chd_beta_list, axis = 1)
chd_pval_df = pd.concat(chd_pval_list, axis = 1)

if pheno == 'AFIB':
    c2hest_auroc_df = pd.DataFrame.from_dict(c2hest_auroc_dict, orient = 'index', columns = [colname])
    c2hest_auprc_df = pd.DataFrame.from_dict(c2hest_auprc_dict, orient = 'index', columns = [colname])
    c2hest_f1_df = pd.DataFrame.from_dict(c2hest_f1_dict, orient = 'index', columns = [colname])
    c2hest_balanced_acc_df = pd.DataFrame.from_dict(c2hest_balanced_acc_dict, orient = 'index', columns = [colname])
    c2hest_beta_df = pd.concat(c2hest_beta_list, axis = 1)
    c2hest_pval_df = pd.concat(c2hest_pval_list, axis = 1)

# export dfs
total_cvd_auroc_df.to_csv((pheno + '.PREVENT_CRS_total_cvd.AUROC.' + colname + '.txt'), sep = '\t')
total_cvd_auprc_df.to_csv((pheno + '.PREVENT_CRS_total_cvd.AUPRC.' + colname + '.txt'), sep = '\t')
total_cvd_f1_df.to_csv((pheno + '.PREVENT_CRS_total_cvd.F1_SCORE.' + colname + '.txt'), sep = '\t')
total_cvd_balanced_acc_df.to_csv((pheno + '.PREVENT_CRS_total_cvd.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
total_cvd_beta_df.to_csv((pheno + '.PREVENT_CRS_total_cvd.BETA.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')
total_cvd_pval_df.to_csv((pheno + '.PREVENT_CRS_total_cvd.PVAL.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')

ascvd_auroc_df.to_csv((pheno + '.PREVENT_CRS_ascvd.AUROC.' + colname + '.txt'), sep = '\t')
ascvd_auprc_df.to_csv((pheno + '.PREVENT_CRS_ascvd.AUPRC.' + colname + '.txt'), sep = '\t')
ascvd_f1_df.to_csv((pheno + '.PREVENT_CRS_ascvd.F1_SCORE.' + colname + '.txt'), sep = '\t')
ascvd_balanced_acc_df.to_csv((pheno + '.PREVENT_CRS_ascvd.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
ascvd_beta_df.to_csv((pheno + '.PREVENT_CRS_ascvd.BETA.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')
ascvd_pval_df.to_csv((pheno + '.PREVENT_CRS_ascvd.PVAL.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')

heart_failure_auroc_df.to_csv((pheno + '.PREVENT_CRS_heart_failure.AUROC.' + colname + '.txt'), sep = '\t')
heart_failure_auprc_df.to_csv((pheno + '.PREVENT_CRS_heart_failure.AUPRC.' + colname + '.txt'), sep = '\t')
heart_failure_f1_df.to_csv((pheno + '.PREVENT_CRS_heart_failure.F1_SCORE.' + colname + '.txt'), sep = '\t')
heart_failure_balanced_acc_df.to_csv((pheno + '.PREVENT_CRS_heart_failure.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
heart_failure_beta_df.to_csv((pheno + '.PREVENT_CRS_heart_failure.BETA.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')
heart_failure_pval_df.to_csv((pheno + '.PREVENT_CRS_heart_failure.PVAL.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')

stroke_auroc_df.to_csv((pheno + '.PREVENT_CRS_stroke.AUROC.' + colname + '.txt'), sep = '\t')
stroke_auprc_df.to_csv((pheno + '.PREVENT_CRS_stroke.AUPRC.' + colname + '.txt'), sep = '\t')
stroke_f1_df.to_csv((pheno + '.PREVENT_CRS_stroke.F1_SCORE.' + colname + '.txt'), sep = '\t')
stroke_balanced_acc_df.to_csv((pheno + '.PREVENT_CRS_stroke.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
stroke_beta_df.to_csv((pheno + '.PREVENT_CRS_stroke.BETA.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')
stroke_pval_df.to_csv((pheno + '.PREVENT_CRS_stroke.PVAL.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')

chd_auroc_df.to_csv((pheno + '.PREVENT_CRS_chd.AUROC.' + colname + '.txt'), sep = '\t')
chd_auprc_df.to_csv((pheno + '.PREVENT_CRS_chd.AUPRC.' + colname + '.txt'), sep = '\t')
chd_f1_df.to_csv((pheno + '.PREVENT_CRS_chd.F1_SCORE.' + colname + '.txt'), sep = '\t')
chd_balanced_acc_df.to_csv((pheno + '.PREVENT_CRS_chd.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
chd_beta_df.to_csv((pheno + '.PREVENT_CRS_chd.BETA.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')
chd_pval_df.to_csv((pheno + '.PREVENT_CRS_chd.PVAL.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')

if pheno == 'AFIB':
    c2hest_auroc_df.to_csv((pheno + '.C2HEST_CRS.AUROC.' + colname + '.txt'), sep = '\t')
    c2hest_auprc_df.to_csv((pheno + '.C2HEST_CRS.AUPRC.' + colname + '.txt'), sep = '\t')
    c2hest_f1_df.to_csv((pheno + '.C2HEST_CRS.F1_SCORE.' + colname + '.txt'), sep = '\t')
    c2hest_balanced_acc_df.to_csv((pheno + '.C2HEST_CRS.BALANCED_ACCURACY.' + colname + '.txt'), sep = '\t')
    c2hest_beta_df.to_csv((pheno + '.C2HEST_CRS.BETA.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')
    c2hest_pval_df.to_csv((pheno + '.C2HEST_CRS.PVAL.' + colname + '.txt'), sep = '\t', na_rep = 'NaN')