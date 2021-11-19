#!/bin/bash

for WARMUP in 8000 10000; do
    for DROPOUT in 0.1 0.3 ; do
        for LR in 0.001 0.0005 0.0001 ; do
            bash baseline/en-ta.sh $WARMUP $DROPOUT $LR 0.1
        done
    done
done
