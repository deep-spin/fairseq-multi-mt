#!/bin/bash

CHECKPOINT_PATH=$1

# many of these options are copied from https://github.com/pytorch/fairseq/issues/3343
# adapted from https://github.com/pytorch/fairseq/issues/3233#issuecomment-802020438
fairseq-train \
    /home/bpop/en-ta-sanity-bin \
    --save-dir $CHECKPOINT_PATH \
    --task translation_multi_simple_epoch \
    --encoder-normalize-before \
    --langs $( cat /home/bpop/flores101_mm100_175M/language_pairs.txt | tr "," "\n" | cut -f 1 -d "-" | sort | uniq | perl -pe 'chomp if eof' | tr "\n" "," ) \
    --lang-pairs "en-ta,ta-en" \
    --max-tokens 1024 \
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
    --encoder-embed-dim 512 \
    --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --decoder-ffn-embed-dim 2048 \
    --encoder-layerdrop 0.05 \
    --decoder-layerdrop 0.05 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --fp16 \
    --memory-efficient-fp16 \
    --ddp-backend no_c10d \
    --find-unused-parameters \
    --adapter-enc-dim 256 \
    --adapter-enc-type 'per_lang' \
    --adapter-dec-dim 256 \
    --adapter-dec-type 'per_lang' \
    --finetune-enc-modules adapter \
    --finetune-dec-modules adapter \
    --load-pretrained-encoder-from /home/bpop/flores101_mm100_175M/model.pt \
    --load-pretrained-decoder-from /home/bpop/flores101_mm100_175M/model.pt
