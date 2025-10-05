#!/bin/bash -l

#SBATCH --job-name=multi_node
#SBATCH --partition=tupi
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00

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

echo "Create TF_CONFIG"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

MASTER_PORT=29500

worker_list=""
for node in "${nodes_array[@]}"; do
    worker_list+="\"$node:$MASTER_PORT\","
done
worker_list=${worker_list%,}


CMD="
cd $SCRATCH
cd comp-sys-perf-analysis-final-work/tensorflow || { git clone git@github.com:kenichi220/comp-sys-perf-analysis-final-work.git ; cd comp-sys-perf-analysis-final-work/tensorflow ; ./create-env.sh ; }

echo "Activate venv ---- add create venv if not has been exist"
source venv/bin/activate

TF_CONFIG=\$(cat <<EOF
{
    \"cluster\": {
        \"worker\": [${worker_list}]
    },
    \"task\": {
        \"type\": \"worker\",
        \"index\": \${SLURM_PROCID}
    }
}
EOF
)
export TF_CONFIG

echo \"Host: \$(hostname) | Rank (PROCID): \${SLURM_PROCID} | TF_CONFIG: \${TF_CONFIG}\"

python train-parallel.py

cp -r $SCRATCH/comp-sys-perf-analysis-final-work/tensorflow/logs/* ~/logs
"

echo "Starting distributed training with srun..."

srun --export=ALL bash -c "$CMD"

echo "Training completed."
