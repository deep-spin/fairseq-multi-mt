#!/bin/bash

CHECKPOINT_PATH=$1

# finetuning of en<->tl languages with adapters. No other parameters are updated.

fairseq-train \
    /mnt/data/bpop/wmt-multi/filtered-data/task2/normal-vocab/bin/ \
    --save-dir $CHECKPOINT_PATH \
    --task translation_multi_simple_epoch \
    --encoder-normalize-before \
    --langs $( cat /mnt/data/bpop/wmt-multi/flores101_mm100_175M/language_pairs.txt | tr "," "\n" | cut -f 1 -d "-" | sort | uniq | perl -pe 'chomp if eof' | tr "\n" "," ) \
    --lang-pairs "en-tl,tl-en" \
    --max-tokens 1024 \
    --update-freq 2 \
    --decoder-normalize-before \
    --sampling-method temperature \
    --sampling-temperature 5 \
    --encoder-langtok src \
    --decoder-langtok \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-eps 1e-08 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-04 \
    --warmup-updates 2500 \
    --max-epoch 5 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
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
    --tensorboard-logdir $CHECKPOINT_PATH/tb \
    --find-unused-parameters \
    --adapter-enc-dim 512 \
    --adapter-enc-type 'per_lang' \
    --adapter-dec-dim 512 \
    --adapter-dec-type 'per_lang' \
    --finetune-enc-modules adapter \
    --finetune-dec-modules adapter \
    --load-pretrained-encoder-from /mnt/data/bpop/wmt-multi/flores101_mm100_175M/model.pt \
    --load-pretrained-decoder-from /mnt/data/bpop/wmt-multi/flores101_mm100_175M/model.pt \
    --homogeneous-batch
