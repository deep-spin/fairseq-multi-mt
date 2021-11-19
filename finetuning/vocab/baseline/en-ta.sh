#!/bin/bash

WARMUP=$1
DROPOUT=$2
LR=$3
LS=$4

CHECKPOINT_PATH=/mnt/data/bpop/wmt-multi/bathwater/models/baseline/en-ta-$WARMUP-$DROPOUT-$LR-$LS

# The basic skeleton of command line scripts for finetuning the small M2M model
# for en<>ta. I would like to run this on 4 GPUs, but it may only be feasible
# to do 2.
fairseq-train \
    /mnt/data/bpop/wmt-multi/filtered-data/task2/normal-vocab/bin/ \
    --finetune-from-model /mnt/data/bpop/wmt-multi/flores101_mm100_175M/model.pt \
    --save-dir $CHECKPOINT_PATH \
    --task translation_multi_simple_epoch \
    --encoder-normalize-before \
    --langs $( cat /mnt/data/bpop/wmt-multi/flores101_mm100_175M/language_pairs.txt | tr "," "\n" | cut -f 1 -d "-" | sort | uniq | perl -pe 'chomp if eof' | tr "\n" "," ) \
    --lang-pairs "en-ta,ta-en" \
    --max-tokens 1024 \
    --update-freq 2 \
    --decoder-normalize-before \
    --sampling-method temperature \
    --sampling-temperature 5 \
    --encoder-langtok src \
    --decoder-langtok \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing $LS \
    --optimizer adam \
    --adam-eps 1e-08 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr $LR \
    --warmup-updates $WARMUP \
    --max-epoch 10 \
    --dropout $DROPOUT \
    --attention-dropout $DROPOUT \
    --weight-decay 0.0 \
    --keep-best-checkpoints 1 \
    --save-interval-updates 20000 \
    --seed 222 \
    --log-format simple \
    --log-interval 10 \
    --patience 10 \
    --arch transformer_wmt_en_de_big \
    --encoder-layers 6 \
    --decoder-layers 6 \
    --encoder-embed-dim 512 \
    --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --decoder-ffn-embed-dim 2048 \
    --encoder-layerdrop 0.0 \
    --decoder-layerdrop 0.0 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --fp16 \
    --memory-efficient-fp16 \
    --ddp-backend no_c10d \
    --tensorboard-logdir $CHECKPOINT_PATH/tb
