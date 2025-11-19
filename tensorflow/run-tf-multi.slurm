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


srun --export=ALL bash -c '
# This block of code is executed by each Slurm task on its respective node
echo "--- Running setup on $(hostname) ---"
cd $SCRATCH

REPO_DIR="comp-sys-perf-analysis-final-work"
if [ ! -d "$REPO_DIR" ]; then
    echo "Directory not found on $(hostname). Cloning repository..."
    git clone https://github.com/kenichi220/comp-sys-perf-analysis-final-work.git
else
    echo "Repository directory already exists on $(hostname)."
fi

DATA_DIR="tiny-imagenet-200"
if [ ! -d "$DATA_DIR" ]; then
	echo "Directory not found on $(hostname). Download data..."
	wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
	unzip -q *.zip
else
	echo "Data already exists on $(hostname)"
fi

cd $REPO_DIR/tensorflow

if [ ! -d "venv" ]; then
    echo "Virtual environment not found on $(hostname). Creating..."
    ./create-env.sh
else
    echo "Virtual environment already exists on $(hostname)."
fi
echo "--- Setup on $(hostname) complete ---"
'

scontrol show hostnames $SLURM_JOB_NODELIST > /dev/null && echo "All nodes finished setup."

TO_RUN = "train-sync.py"

if [ "$S_TYPE" == "ctl" ]; then
	echo "==============================================="
	echo "=============> Model $MODEL <=================="
	echo "==============================================="
	TO_RUN = "train-ctl.py"
fi

CMD="
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

python $TO_RUN --model $PY_MODEL --batch $PY_BATCH_SIZE

cp -r $SCRATCH/comp-sys-perf-analysis-final-work/tensorflow/logs/* ~/logs
"

echo "Starting distributed training with srun..."

srun --export=ALL bash -c "$CMD"

echo "Training completed."
