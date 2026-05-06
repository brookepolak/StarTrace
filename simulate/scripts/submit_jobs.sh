#!/bin/bash
#SBATCH --job-name=petar_node
#SBATCH --output=logs/node_%j.out
#SBATCH --error=logs/node_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0   # use full node memory

DIRS=($(ls -d NSC* | sort -V))

# Run 64 jobs in parallel
for i in $(seq 0 63)
do
    (
        DIR=${DIRS[$i]}
        echo "Running $DIR"
        cd $DIR
        petar.init ics.txt
        petar -u 1 -r 0.5 -t 500 ics.txt.input
    ) &
done

wait
echo "All jobs finished"