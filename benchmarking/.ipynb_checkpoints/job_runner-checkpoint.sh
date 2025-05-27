#!/bin/bash

source  /home/cmoelbe/bin/anaconda3/bin/activate python_env

NEW_CACHE=$TMPDIR/cache
mkdir -p $NEW_CACHE
if [ -z $XDG_CACHE_HOME ]; then
    XDG_CACHE_HOME=$HOME/.cache
fi
cp -r $XDG_CACHE_HOME/gimmemotifs $NEW_CACHE/
export XDG_CACHE_HOME=$NEW_CACHE
echo $XDG_CACHE_HOME

echo ${SGE_TASK_ID}
# Extract the line corresponding to the PBS_ARRAYID, skipping the header
line=$(sed -n "${SGE_TASK_ID}p" results/benchmark_todo.csv)
echo $line
# Parse the parameters from the line
IFS=',' read -r dataset networktype degs thres_access fpr edges_cutoff_expr max_tf  <<< "$line"

# Run the Python script with the parameters
python3 run_grn_construction.py --dataset "$dataset" \
                                --networktype "$networktype" \
                                --max_features "$degs" \
                                --thres_access "$thres_access"  \
                                --fpr "$fpr" \
                                --edges_cutoff_expr "$edges_cutoff_expr" \
                                --max_tf "$max_tf" \
                                --outfolder "benchmark_files"