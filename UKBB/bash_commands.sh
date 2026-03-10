# create pgsc-calc docker applet
# 1. use app wizard
dx-app-wizard
# 2. add following options into app-wizard
App Name: pgsc_calc
Title []: pgsc_calc
Summary []: Runs PGSC-CALC pipeline on UKBiobank DNAnexus, made by Katie Cardone
Version [0.0.1]:
1st input name (<ENTER> to finish): input
Label (optional human-readable name) []: 
Choose a class (<TAB> twice for choices): array:file
This is an optional parameter [y/n]: n
2nd input name (<ENTER> to finish): command
Label (optional human-readable name) []: 
Choose a class (<TAB> twice for choices): string
This is an optional parameter [y/n]: n
3rd input name (<ENTER> to finish):
1st output name (<ENTER> to finish):
Timeout policy [48h]: 7d
Programming language: bash
Will this app need access to the Internet? [y/N]: y
Will this app need access to the parent project? [y/N]: y
Choose an instance type for your app [mem1_ssd1_v2_x4]: mem1_ssd1_v2_x4
# 3. made additional edits to applet scripts in terminal (see uploaded scripts)
## use vim
# 4. build applet
dx build pgsc_calc --overwrite
# 5. upload applet scripts
dx upload -r pgsc_calc --path :/Cardio_IRM/scripts/

# test PGSC-CALC applet
plink_input_dir=":/Cardio_IRM/output/plink"
other_input_dir=":/Cardio_IRM/input"
ancestry_input_dir=":/Cardio_IRM/input"
output_dir=":/Cardio_IRM/output/pgsc_calc"

input_flags=()
for file in $(dx ls "${plink_input_dir}" --brief)
do
input_flags+=("-iinput=${plink_input_dir}/${file}")
done

for file in $(dx ls "${other_input_dir}/score_files" --brief)
do
input_flags+=("-iinput=${other_input_dir}/score_files/${file}")
done

dx run pgsc_calc \
-y \
"${input_flags[@]}" \
-iinput="${other_input_dir}/UKBB.CAD.AFIB.PGSC_CALC.samplesheet.csv" \
-iinput="${ancestry_input_dir}/pgsc_HGDP+1kGP_v1.tar.zst" \
-icommand='ls input/score_files/' \
--brief \
--name "pgsc_calc" \
--destination ${output_dir} \
--instance-type mem2_ssd1_v2_x64

# extract pgs catalog variants and create plink files- HF
# 1. run the following code from ttyd terminal
bgen_input_dir=":/Bulk/Imputation/Imputation from genotype (TOPmed)"
variants_input_dir=":/Cardio_IRM/input"
output_dir=":/Cardio_IRM/output/plink"

for chr in $(seq 1 22) X
do
dx run app-swiss-army-knife \
-y \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.bgen.bgi" \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.bgen" \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.sample" \
-iin="${variants_input_dir}/HF.PGS005097.var_list.txt" \
--brief \
--name "extact_pgs_vars.chr${chr}" \
-icmd="plink2 --bgen ukb21007_c${chr}_b0_v1.bgen ref-first \
                --sample ukb21007_c${chr}_b0_v1.sample \
                --extract range HF.PGS005097.var_list.txt \
                --set-all-var-ids @:#:\\\$r:\\\$a \
                --new-id-max-allele-len 1000 \
                --make-pgen \
                --out ukb21007_c${chr}_b0_v1.topmed_imputed.hf_pgs_vars" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# extract pgs catalog variants and create plink files- CAD and AFIB
# 1. run the following code from ttyd terminal
bgen_input_dir=":/Bulk/Imputation/Imputation from genotype (TOPmed)"
variants_input_dir=":/Cardio_IRM/input"
output_dir=":/Cardio_IRM/output/plink"

for chr in $(seq 1 22)
do
dx run app-swiss-army-knife \
-y \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.bgen.bgi" \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.bgen" \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.sample" \
-iin="${variants_input_dir}/CAD.PGS005112.AFIB.PGS005072.var_list.txt" \
--brief \
--name "extact_pgs_vars.chr${chr}" \
-icmd="plink2 --bgen ukb21007_c${chr}_b0_v1.bgen ref-first \
                --sample ukb21007_c${chr}_b0_v1.sample \
                --extract range CAD.PGS005112.AFIB.PGS005072.var_list.txt \
                --set-all-var-ids @:#:\\\$r:\\\$a \
                --new-id-max-allele-len 1000 \
                --make-pgen \
                --out ukb21007_c${chr}_b0_v1.topmed_imputed.cad_afib_pgs_vars" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# submit pgsc-calc job from ukbb ttyd- HF
# 1. run the following code in ttyd terminal
plink_input_dir=":/Cardio_IRM/output/plink"
other_input_dir=":/Cardio_IRM/input/"
output_dir=":/Cardio_IRM/output/pgsc_calc"

input_flags=()
for file in $(dx ls "${plink_input_dir}" --brief)
do
input_flags+=("-iinput=${plink_input_dir}/${file}")
done

dx run pgsc_calc \
-y \
"${input_flags[@]}" \
-iinput="${other_input_dir}/UKBB.HF.PGSC_CALC.samplesheet.csv" \
-iinput="${other_input_dir}/pgsc_HGDP+1kGP_v1.tar.zst" \
-iinput="${other_input_dir}/PGS005097_hmPOS_GRCh38.txt.gz" \
-icommand="nextflow run pgscatalog/pgsc_calc -profile conda \
            --input input/UKBB.HF.PGSC_CALC.samplesheet.csv \
            --scorefile input/PGS005097_hmPOS_GRCh38.txt.gz \
            --target_build GRCh38 \
            --outdir output/ \
            --max_cpus 64 \
            --max_memory 256.GB \
            --min_overlap 0.0 \
            --max_time 240.h \
            --run_ancestry input/pgsc_HGDP+1kGP_v1.tar.zst \
            --keep_multiallelic True \
            --hwe_ref 0 \
            --pca_maf_target 0.05" \
--brief \
--name "pgsc_calc" \
--destination ${output_dir} \
--instance-type mem2_ssd1_v2_x64

# submit pgsc-calc job from ukbb ttyd- CAD and AFIB
# 1. run the following code in ttyd terminal
plink_input_dir=":/Cardio_IRM/output/plink"
other_input_dir=":/Cardio_IRM/input"
ancestry_input_dir=":/Cardio_IRM/input"
output_dir=":/Cardio_IRM/output/pgsc_calc"

input_flags=()
for file in $(dx ls "${plink_input_dir}" --brief)
do
input_flags+=("-iinput=${plink_input_dir}/${file}")
done

dx run pgsc_calc \
-y \
"${input_flags[@]}" \
-iinput="${other_input_dir}/UKBB.CAD.AFIB.PGSC_CALC.samplesheet.csv" \
-iinput="${ancestry_input_dir}/pgsc_HGDP+1kGP_v1.tar.zst" \
-icommand="nextflow run pgscatalog/pgsc_calc -profile conda \
            --input input/UKBB.CAD.AFIB.PGSC_CALC.samplesheet.csv \
            --pgs_id PGS005072,PGS005112 \
            --target_build GRCh38 \
            --outdir output/ \
            --max_cpus 64 \
            --max_memory 256.GB \
            --min_overlap 0.0 \
            --max_time 240.h \
            --run_ancestry input/pgsc_HGDP+1kGP_v1.tar.zst \
            --keep_multiallelic True \
            --hwe_ref 0 \
            --pca_maf_target 0.05" \
--brief \
--name "pgsc_calc" \
--destination ${output_dir} \
--instance-type mem2_ssd1_v2_x64

# catboost feature selection python script
# CAD
input_dir=":/Cardio_IRM/input/"
scripts_dir=":/Cardio_IRM/scripts/"

output_dir=":/Cardio_IRM/output/feature_selection/CAD/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/catboost_feature_selection.py" \
-iin="${input_dir}/UKBB.CAD.IRM.PXS_feature_selection_input.csv" \
--brief \
--name "feature_selection_${i}" \
-icmd="python catboost_feature_selection.py \
                --input UKBB.CAD.IRM.PXS_feature_selection_input.csv \
                --pheno CAD \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# AFIB
output_dir=":/Cardio_IRM/output/feature_selection/AFIB/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/catboost_feature_selection.py" \
-iin="${input_dir}/UKBB.AFIB.IRM.PXS_feature_selection_input.csv" \
--brief \
--name "feature_selection_${i}" \
-icmd="python catboost_feature_selection.py \
                --input UKBB.AFIB.IRM.PXS_feature_selection_input.csv \
                --pheno AFIB \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# HF
output_dir=":/Cardio_IRM/output/feature_selection/HF/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/catboost_feature_selection.py" \
-iin="${input_dir}/UKBB.HF.IRM.PXS_feature_selection_input.csv" \
--brief \
--name "feature_selection_${i}" \
-icmd="python catboost_feature_selection.py \
                --input UKBB.HF.IRM.PXS_feature_selection_input.csv \
                --pheno HF \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# eval python script
# CAD
input_dir=":/Cardio_IRM/input/"
weight_dir=":/Cardio_IRM/output/"
scripts_dir=":/Cardio_IRM/scripts/"

output_dir=":/Cardio_IRM/output/eval/CAD/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/IRM_eval_script_missing_data.py" \
-iin="${input_dir}/UKBB.CAD.IRM.eval_input.csv" \
-iin="${weight_dir}/UKBB.CAD.catboost.shap.selected_features.txt" \
--brief \
--name "eval_${i}" \
-icmd="python IRM_eval_script_missing_data.py \
                --input UKBB.CAD.IRM.eval_input.csv \
                --weight UKBB.CAD.catboost.shap.selected_features.txt \
                --pheno CAD \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# AFIB
output_dir=":/Cardio_IRM/output/eval/AFIB/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/IRM_eval_script_missing_data.py" \
-iin="${input_dir}/UKBB.AFIB.IRM.eval_input.csv" \
-iin="${weight_dir}/UKBB.AFIB.catboost.shap.selected_features.txt" \
--brief \
--name "eval_${i}" \
-icmd="python IRM_eval_script_missing_data.py \
                --input UKBB.AFIB.IRM.eval_input.csv \
                --weight UKBB.AFIB.catboost.shap.selected_features.txt \
                --pheno AFIB \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# HF
output_dir=":/Cardio_IRM/output/eval/HF/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/IRM_eval_script_missing_data.py" \
-iin="${input_dir}/UKBB.HF.IRM.eval_input.csv" \
-iin="${weight_dir}/UKBB.HF.catboost.shap.selected_features.txt" \
--brief \
--name "eval_${i}" \
-icmd="python IRM_eval_script_missing_data.py \
                --input UKBB.HF.IRM.eval_input.csv \
                --weight UKBB.HF.catboost.shap.selected_features.txt \
                --pheno HF \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# ROC/PRC curve plotting scripts
# CAD
python IRM_plot_script.py --input UKBB.CAD.IRM.eval_input.csv \
--pheno CAD \
--weight UKBB.CAD.catboost.shap.selected_features.txt \
--mean_metrics UKBB.CAD.IRM.population_performance.reformat.csv \
--output_dir plots/ \
--iter 1

# AFIB
python IRM_plot_script.py --input UKBB.AFIB.IRM.eval_input.csv \
--pheno AFIB \
--weight UKBB.AFIB.catboost.shap.selected_features.txt \
--mean_metrics UKBB.AFIB.IRM.population_performance.reformat.csv \
--output_dir plots/ \
--iter 1

# HF
python IRM_plot_script.py --input UKBB.HF.IRM.eval_input.csv \
--pheno HF \
--weight UKBB.HF.catboost.shap.selected_features.txt \
--mean_metrics UKBB.HF.IRM.population_performance.reformat.csv \
--output_dir plots/ \
--iter 1