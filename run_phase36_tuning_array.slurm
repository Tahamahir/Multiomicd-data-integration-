
# Active ton environnement conda

eval "$(conda shell.bash hook)"



# Change ce chemin selon ton environnement

conda activate /srv/lustre01/project/cbs_bioinf-0leti8ikwe4/shared/anaconda3/envs/python



# Va à la racine de ton projet

cd /srv/lustre01/project/cbs_bioinf-0leti8ikwe4/shared/Wheat/First_Data



python 10_analysis/scripts/36_tune_best_model_per_mb_slurm.py \

    --project-root . \

    --task-id ${SLURM_ARRAY_TASK_ID}



echo "End: $(date)"

