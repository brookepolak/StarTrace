#!/bin/bash

# total seeds per N_sc
NSEEDS=100

# loop over number of subclusters
for NSC in {1..8}
do
    for SEED in $(seq 0 $((NSEEDS-1)))
    do
        DIR="NSC${NSC}SEED${SEED}"
        
        echo "Setting up $DIR"
        mkdir -p $DIR

        # generate ICs (assumes your python script takes args)
        python generate_ics.py \
            --n_subclusters $NSC \
            --seed $SEED \
            --output ${DIR}/ics.txt \
            --radius 10

    done
done