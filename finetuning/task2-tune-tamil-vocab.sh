#!/bin/bash
CHECKPOINT_PATH=$1

# basic finetuning of the task 2 languages. This model uses the provided vocabulary
# and updates all model parameters.
# The sampling temperature (5) is something I saw in the literature somewhere

# many of these options are copied from https://github.com/pytorch/fairseq/issues/3343
# adapted from https://github.com/pytorch/fairseq/issues/3233#issuecomment-802020438
fairseq-train \
    /mnt/data/bpop/wmt-multi/tamil-data/32k/bin \
    --save-dir $CHECKPOINT_PATH \
    --task translation_multi_simple_epoch \
    --encoder-normalize-before \
    --langs $( cat /mnt/data/bpop/wmt-multi/flores101_mm100_615M/language_pairs.txt | tr "," "\n" | cut -f 1 -d "-" | sort | uniq | perl -pe 'chomp if eof' | tr "\n" "," ) \
    --lang-pairs "en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta"  \
    --max-tokens 1024 \
    --decoder-normalize-before \
    --sampling-method uniform \
    --encoder-langtok src \
    --decoder-langtok \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-eps 1e-08 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-04 \
    --warmup-updates 8000 \
    --max-update 200000 \
    --dropout 0.3 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --update-freq 2 \
    --save-interval-updates 10000 \
    --no-epoch-checkpoints \
    --seed 222 \
    --log-format simple \
    --log-interval 10 \
    --patience 10 \
    --arch transformer_wmt_en_de_big \
    --encoder-layers 12 \
    --decoder-layers 12 \
    --encoder-embed-dim 1024 \
    --decoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 4096 \
    --decoder-ffn-embed-dim 4096 \
    --encoder-layerdrop 0.0 \
    --decoder-layerdrop 0.0 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --fp16 \
    --memory-efficient-fp16 \
    --ddp-backend no_c10d \
    --find-unused-parameters \
    --finetune-enc-modules embed_tokens \
    --finetune-dec-modules embed_tokens \
    --load-pretrained-encoder-from /mnt/data/bpop/wmt-multi/flores_big_task2_vocab/model.pt \
    --load-pretrained-decoder-from /mnt/data/bpop/wmt-multi/flores_big_task2_vocab/model.pt \
    --discard-pretrained-encoder-embeddings \
    --discard-pretrained-decoder-embeddings \
    --homogeneous-batch \
    --encoder-embed-path /mnt/data/bpop/wmt-multi/tamil-data/tamil-32k-embeddings.txt
