#!/bin/bash -l

#SBATCH --job-name=multi_node
#SBATCH --partition=poti
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00

export SCRATCH="/scratch/rrdmatos"

echo "================================================================="
echo "Date: $(date)"
echo "Submission Host: $(hostname)"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "================================================================="
################################
# Change to working directory
################################

set -euxo pipefail

mkdir -p ~/logs

echo "Create TF_CONFIG"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

MASTER_PORT=29500

worker_list=""
for node in "${nodes_array[@]}"; do
    worker_list+="\"$node:$MASTER_PORT\","
done
worker_list=${worker_list%,}


srun --export=ALL,SCRATCH=$SCRATCH bash -c '
# This block of code is executed by each Slurm task on its respective node
echo "--- Running setup on $(hostname) ---"
cd $SCRATCH

rm -r *

cp -r ~/perf/tensorflow/create-env.sh ~/perf/tensorflow/train-* .
DATA_DIR="tiny-imagenet-200"

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q *.zip

.  ./create-env.sh
'

scontrol show hostnames $SLURM_JOB_NODELIST > /dev/null && echo "All nodes finished setup."

TO_RUN="train-sync.py"

if [ "$S_TYPE" == "ctl" ]; then
	TO_RUN="train-ctl.py"
fi

env


srun --export=ALL,SCRATCH=$SCRATCH,TO_RUN=$TO_RUN,worker_list=$worker_list bash -c '

cd $SCRATCH
env
source venv/bin/activate

TF_CONFIG=$(cat <<EOF
{
    "cluster": {
        "worker": [${worker_list}]
    },
    "task": {
        "type": "worker",
        "index": ${SLURM_PROCID}
    }
}
EOF
)
export TF_CONFIG

echo "Host: $(hostname) | Rank (PROCID): ${SLURM_PROCID} | TF_CONFIG: ${TF_CONFIG}"

echo "$TO_RUN,$PY_MODEL\n"
python $TO_RUN --model $PY_MODEL --batch $PY_BATCH_SIZE

LOGS_DIR="~/logs/${TO_RUN}_${SLURM_JOB_ID}_${SLURM_PROCID}"

mkdir -p $LOGS_DIR
cp -r "$SCRATCH/logs/*" $LOGS_DIR
'

echo "Starting distributed training with srun..."


echo "Training completed."
