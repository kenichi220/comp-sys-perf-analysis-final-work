#!/bin/bash

INPUT_CSV="projeto_experimental.csv"

SLURM_SCRIPT="run-tf-multi.slurm"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Not found $INPUT_CSV"
    exit 1
fi

grep -v '^#' "$INPUT_CSV" | while IFS=',' read -r NODES TYPE MODEL BATCH_SIZE BLOCKS
do
    # Diff format, jump
    if [ -z "$NODES" ] || [ -z "$TYPE" ] || [ -z "$MODEL" ] || [ -z "$BATCH_SIZE" ]; then
        echo "Jump line: $NODES,$TYPE,$MODEL,$BATCH_SIZE,$BLOCKS"
        continue
    fi

    echo "=========================================================="
    echo "  Launch job"
    echo "  Nodes: $NODES"
    echo "  TYPE: $TYPE"
    echo "  MODEL: $MODEL"
    echo "  Batch Size: $BATCH_SIZE"
    echo "=========================================================="

    sbatch \
        --nodes=$NODES \
	--nodelist=poti1,poti2,poti3,poti5 \
        --job-name="train_${MODEL}_${TYPE}_${BATCH_SIZE}_${NODES}" \
        --export=ALL,S_TYPE=$TYPE,PY_BATCH_SIZE=$BATCH_SIZE,PY_MODEL=$MODEL \
        "$SLURM_SCRIPT"

    sleep 1

done

echo "All jobs finished"
