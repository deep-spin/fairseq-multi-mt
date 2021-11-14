#!/bin/bash

CHECKPOINT_PATH=$1

# From-scratch training of en<->tl with a new vocabulary consisting of separate 32k BPE
# segmentations for both languages. The resulting language-specific vocabularies
# are concatenated and duplicates removed. Embeddings from m2m100 are kept for
# subword types that exist in both this new vocabulary and the 256k baseline
# vocab.

# adapted from https://github.com/pytorch/fairseq/issues/3233#issuecomment-802020438
fairseq-train \
    /mnt/data/bpop/wmt-multi/filtered-data/task2/en-tl-32-cat/bin/ \
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
    --tensorboard-logdir $CHECKPOINT_PATH/tb
