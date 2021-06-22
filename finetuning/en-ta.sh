#!/bin/bash

MODEL_PATH=$1

# many of these options are copied from https://github.com/pytorch/fairseq/issues/3343
# adapted from https://github.com/pytorch/fairseq/issues/3233#issuecomment-802020438
fairseq-train \
    data_bin \
    --finetune-from-model $MODEL_PATH/model.pt \
    --save-dir /checkpoint \
    --task translation_multi_simple_epoch \
    --encoder-normalize-before \
    --langs $( cat $MODEL_PATH/language_pairs.txt ) \
    --lang-pairs "en-ta" \
    --max-tokens 1200 \
    --decoder-normalize-before \
    --sampling-method temperature \
    --sampling-temperature 1.5 \
    --encoder-langtok src \
    --decoder-langtok \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-eps 1e-08 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-05 \
    --warmup-updates 2500 \
    --max-update 40000 \
    --dropout 0.3 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --update-freq 2 \
    --save-interval 1 \
    --save-interval-updates 5000 \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --seed 222 \
    --log-format simple \
    --log-interval 2 \
    --patience 10 \
    --arch transformer_wmt_en_de_big \
    --encoder-layers 6 \
    --decoder-layers 6 \
    --encoder-layerdrop 0.05 \
    --decoder-layerdrop 0.05 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --ddp-backend no_c10d  # what on earth is that?




fairseq-train \
    $DATA_BIN \
    --arch transformer_wmt_en_de_big \
    --finetune-from-model $MODEL_PATH/model.pt \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --layernorm-embedding \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" \
    --sampling-temperature 1.5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-pairs "en-ta" \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-eps 1e-08 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-05 \
    --warmup-updates 2500 \
    --max-update 40000 \
    --dropout 0.3 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --max-tokens 1024 \
    --update-freq 2 \
    --save-interval 1 \
    --save-interval-updates 5000 \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --seed 222 \
    --log-format simple \
    --log-interval 2 \
    --fp16 \
    --memory-efficient-fp16
