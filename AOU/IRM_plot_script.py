# load packages
import pandas as pd
import argparse as ap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve 
import matplotlib.pyplot as plt
import seaborn as sns

# parse arguments
def make_arg_parser():
    parser = ap.ArgumentParser(description = ".")

    parser.add_argument('--iter', required = True, help = 'random seed for iteration')
    
    parser.add_argument('--input', required = True, help = 'input filename')
    
    parser.add_argument('--weight', required = True, help = 'weight filename')
    
    parser.add_argument('--pheno', required = True, help = 'phenotype')
    
    parser.add_argument('--mean_metrics', required = True, help = 'mean metrics filename')
    
    parser.add_argument('--output_dir', required = True, help = 'output filename')
    
    return parser

args = make_arg_parser().parse_args()

# parse arguments
iter = int(args.iter)
input_filename = args.input
weight_filename = args.weight
pheno = args.pheno
mean_metrics_filename = args.mean_metrics
output_dir = args.output_dir

# create iteration column name
print(iter)
colname = 'ITER_' + str(iter)

# read in input files
input = pd.read_csv(input_filename)
weight = pd.read_csv(weight_filename, sep = '\t', index_col = 0)
mean_metrics = pd.read_csv(mean_metrics_filename, index_col = 0)

# remove missing from AFIB DF
if pheno == 'AFIB':
    input = input.dropna(subset = ['AFIB_C2HEST_score'])

# split dataset
train = input.sample(frac = 0.7, random_state = iter)
print(train[pheno].value_counts(dropna = False))
test = input.drop(train.index)
print(test[pheno].value_counts(dropna = False))   

# create PXS column list
pxs_cols = weight.index.tolist()
pxs_cols.remove(('AGE_' + pheno))
pxs_cols.remove('SEX')
#print(len(pxs_cols))

# create eval column list
heart_failure_eval_col_list = [('PGS_' + pheno),
                               (pheno + "_PREVENT_CRS_heart_failure"),
                               'PXS_AVG',
                               'PXS_WEIGHTED_AVG',
                               [('PGS_' + pheno), (pheno + "_PREVENT_CRS_heart_failure")],
                               [('PGS_' + pheno), 'PXS_AVG'],
                               [('PGS_' + pheno), 'PXS_WEIGHTED_AVG'],
                               [(pheno + "_PREVENT_CRS_heart_failure"), 'PXS_AVG'],
                               [(pheno + "_PREVENT_CRS_heart_failure"), 'PXS_WEIGHTED_AVG'],
                               [('PGS_' + pheno), (pheno + "_PREVENT_CRS_heart_failure"), 'PXS_AVG'],
                               [('PGS_' + pheno), (pheno + "_PREVENT_CRS_heart_failure"), 'PXS_WEIGHTED_AVG']]

chd_eval_col_list = [('PGS_' + pheno),
                     (pheno + "_PREVENT_CRS_chd"),
                     'PXS_AVG',
                     'PXS_WEIGHTED_AVG',
                     [('PGS_' + pheno), (pheno + "_PREVENT_CRS_chd")],
                     [('PGS_' + pheno), 'PXS_AVG'],
                     [('PGS_' + pheno), 'PXS_WEIGHTED_AVG'],
                     [(pheno + "_PREVENT_CRS_chd"), 'PXS_AVG'],
                     [(pheno + "_PREVENT_CRS_chd"), 'PXS_WEIGHTED_AVG'],
                     [('PGS_' + pheno), (pheno + "_PREVENT_CRS_chd"), 'PXS_AVG'],
                     [('PGS_' + pheno), (pheno + "_PREVENT_CRS_chd"), 'PXS_WEIGHTED_AVG']]

c2hest_eval_col_list = [('PGS_' + pheno),
                        (pheno + "_C2HEST_score"),
                        'PXS_AVG',
                        'PXS_WEIGHTED_AVG',
                        [('PGS_' + pheno), (pheno + "_C2HEST_score")],
                        [('PGS_' + pheno), 'PXS_AVG'],
                        [('PGS_' + pheno), 'PXS_WEIGHTED_AVG'],
                        [(pheno + "_C2HEST_score"), 'PXS_AVG'],
                        [(pheno + "_C2HEST_score"), 'PXS_WEIGHTED_AVG'],
                        [('PGS_' + pheno), (pheno + "_C2HEST_score"), 'PXS_AVG'],
                        [('PGS_' + pheno), (pheno + "_C2HEST_score"), 'PXS_WEIGHTED_AVG']]

# compute integrated scores
print(pxs_cols)
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

# make age col
age_col = 'AGE_' + pheno

# create empty dictionaries
roc_data = {}
prc_data = {}
auroc_auprc = {}

# evaluate models
if pheno == 'HF':
    for index, col in enumerate(heart_failure_eval_col_list, start = 1):
        if isinstance(col, str):
            model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
            predictors = [age_col, 'SEX'] + [col]
        elif isinstance(col, list):
            model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
            predictors = [age_col, 'SEX'] + col
        else:
            raise ValueError("conditions not met")
        
        all_cols = [pheno] + predictors
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        fpr, tpr, _ = roc_curve(test_df[[pheno]], y_prob_cont)
        precision, recall, _ = precision_recall_curve(test_df[[pheno]], y_prob_cont)
        
        col = str(col)
        col = col.replace("[", "")
        col = col.replace("]", "")
        col = col.replace("'", "")
        col = col.replace("(", "")
        col = col.replace(")", "")
        col = col.replace(",", " +")
        col = col.replace('HF_PREVENT_CRS_heart_failure', 'CRS')
        col = col.replace('AFIB_C2HEST_score', 'CRS')
        col = col.replace('_AFIB', '')
        col = col.replace('CAD_PREVENT_CRS_chd', 'CRS')
        col = col.replace('_CAD', '')
        col = col.replace('_HF', '')
        col = 'Model ' + str(index) + ': ' + col
        
        auroc = mean_metrics.loc[col, 'AUROC']
        roc_data[col] = (fpr, tpr, auroc)
        auprc = mean_metrics.loc[col, 'AUPRC']
        prc_data[col] = (precision, recall, auprc)
        auroc_auprc[col] = (auroc, auprc)
            
if pheno == 'CAD':
    for index, col in enumerate(chd_eval_col_list, start = 1):
        if isinstance(col, str):
            model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
            predictors = [age_col, 'SEX'] + [col]
        elif isinstance(col, list):
            model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
            predictors = [age_col, 'SEX'] + col
        else:
            raise ValueError("conditions not met")
            
        all_cols = [pheno] + predictors
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        fpr, tpr, _ = roc_curve(test_df[[pheno]], y_prob_cont)
        precision, recall, _ = precision_recall_curve(test_df[[pheno]], y_prob_cont)
        
        col = str(col)
        col = col.replace("[", "")
        col = col.replace("]", "")
        col = col.replace("'", "")
        col = col.replace("(", "")
        col = col.replace(")", "")
        col = col.replace(",", " +")
        col = col.replace('HF_PREVENT_CRS_heart_failure', 'CRS')
        col = col.replace('AFIB_C2HEST_score', 'CRS')
        col = col.replace('_AFIB', '')
        col = col.replace('CAD_PREVENT_CRS_chd', 'CRS')
        col = col.replace('_CAD', '')
        col = col.replace('_HF', '')
        col = 'Model ' + str(index) + ': ' + col
        
        auroc = mean_metrics.loc[col, 'AUROC']
        roc_data[col] = (fpr, tpr, auroc)
        auprc = mean_metrics.loc[col, 'AUPRC']
        prc_data[col] = (precision, recall, auprc)
        auroc_auprc[col] = (auroc, auprc)
        
if pheno == 'AFIB':
    for index, col in enumerate(c2hest_eval_col_list, start = 1):
        if isinstance(col, str):
            model_df = train[[pheno, age_col, 'SEX'] + [col]].dropna()
            predictors = [age_col, 'SEX'] + [col]
        elif isinstance(col, list):
            model_df = train[[pheno, age_col, 'SEX'] + col].dropna()
            predictors = [age_col, 'SEX'] + col
        else:
            raise ValueError("conditions not met")
            
        all_cols = [pheno] + predictors
        
        ridge = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs  = -1, class_weight = 'balanced', random_state = iter)
        model = ridge.fit(model_df[predictors], model_df[[pheno]])
        test_df = test[all_cols].dropna()
        y_prob_bin = model.predict(test_df[predictors])
        y_prob_cont = model.predict_proba(test_df[predictors])[:, 1]
        fpr, tpr, _ = roc_curve(test_df[[pheno]], y_prob_cont)
        precision, recall, _ = precision_recall_curve(test_df[[pheno]], y_prob_cont)
        
        col = str(col)
        col = col.replace("[", "")
        col = col.replace("]", "")
        col = col.replace("'", "")
        col = col.replace("(", "")
        col = col.replace(")", "")
        col = col.replace(",", " +")
        col = col.replace('HF_PREVENT_CRS_heart_failure', 'CRS')
        col = col.replace('AFIB_C2HEST_score', 'CRS')
        col = col.replace('_AFIB', '')
        col = col.replace('CAD_PREVENT_CRS_chd', 'CRS')
        col = col.replace('_CAD', '')
        col = col.replace('_HF', '')
        col = 'Model ' + str(index) + ': ' + col
        
        auroc = mean_metrics.loc[col, 'AUROC']
        roc_data[col] = (fpr, tpr, auroc)
        auprc = mean_metrics.loc[col, 'AUPRC']
        prc_data[col] = (precision, recall, auprc)
        auroc_auprc[col] = (auroc, auprc)

# set colorblind palette
sns.set_palette("colorblind")

# make ROC and PRC curves on 2 panels of the same plot
fig, axes = plt.subplots(1, 2, figsize = (16, 8))
ax1 = axes[0]
ax2 = axes[1]

# make ROC curve
for col, (fpr, tpr, auroc) in roc_data.items():
    ax1.plot(fpr, tpr, lw = 7)
ax1.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', lw = 7)
ax1.set_xlabel('False Positive Rate', fontsize = 35)
ax1.set_ylabel('True Positive Rate', fontsize = 35)
title = 'AOU ' + pheno + ' Prediction Receiver-Operating Curves'
ax1.set_title(title, fontsize = 40)
ax1.tick_params(axis = 'both', labelsize = 30)
ax1.grid(True)
ax1.text(0.02, 1.02, "A", transform = ax1.transAxes, fontsize = 35, fontweight = "bold", va = "bottom", ha = "right")

# make PRC curve
for col, (precision, recall, auprc) in prc_data.items():
    ax2.plot(recall, precision, lw = 7)
ax2.set_xlabel('Recall', fontsize = 35)
ax2.set_ylabel('Precision', fontsize = 35)
title = 'AOU ' + pheno + ' Prediction Precision-Recall Curves'
ax2.set_title(title, fontsize = 40)
ax2.tick_params(axis = 'both', labelsize = 30)
ax2.grid(True)
ax2.text(0.02, 1.02, "B", transform = ax2.transAxes, fontsize = 35, fontweight = "bold", va = "bottom", ha = "right")

# create legend
handles = ax1.get_lines()  # handles for the legend
combined_labels = [
    f"{model} (AUROC = {auroc:.3f}, AUPRC = {auprc:.3f})"
    for model, (auroc, auprc) in auroc_auprc.items()]
combined_labels = [f"{model}" for model, (auroc, auprc) in auroc_auprc.items()]
fig.legend(handles, combined_labels, loc = 'lower center', ncol = 2, fontsize = 35)
fig.set_size_inches(36, 18)

# export plot
plt.tight_layout(rect = [0, 0.3, 1, 1])
plt.savefig(output_dir + 'AOU.' + pheno + ".ROC_PRC_curve_combined.png", dpi = 300)
