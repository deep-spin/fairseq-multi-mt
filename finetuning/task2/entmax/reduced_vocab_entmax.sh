CHECKPOINT_PATH=$1

# The question with this model is, can we train a model with monolingual adapters
# on all language pairs using the reduced-vocabulary checkpoint?

# many of these options are copied from https://github.com/pytorch/fairseq/issues/3343
# adapted from https://github.com/pytorch/fairseq/issues/3233#issuecomment-802020438
fairseq-train \
    /mnt/data/bpop/wmt-multi/filtered-data/task2/small-vocab/bin/ \
    --save-dir $CHECKPOINT_PATH \
    --tensorboard-logdir $CHECKPOINT_PATH/tensorboard-log \
    --task translation_multi_simple_epoch \
    --encoder-normalize-before \
    --langs $( cat /mnt/data/bpop/wmt-multi/flores_big_task2_vocab/language_pairs.txt | tr "," "\n" | cut -f 1 -d "-" | sort | uniq | perl -pe 'chomp if eof' | tr "\n" "," ) \
    --lang-pairs "en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta" \
    --max-tokens 1024 \
    --decoder-normalize-before \
    --sampling-method uniform \
    --encoder-langtok src \
    --decoder-langtok \
    --criterion entmax_loss \
    --loss-alpha 1.5 \
    --optimizer adam \
    --adam-eps 1e-08 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-05 \
    --warmup-updates 4000 \
    --max-update 100000 \
    --dropout 0.3 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --update-freq 2 \
    --keep-best-checkpoints 1 \
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
    --adapter-enc-dim 1024 \
    --adapter-enc-type 'per_lang' \
    --adapter-dec-dim 1024 \
    --adapter-dec-type 'per_lang' \
    --finetune-enc-modules adapter \
    --finetune-dec-modules adapter \
    --load-pretrained-encoder-from /mnt/data/bpop/wmt-multi/flores_big_task2_vocab/model.pt \
    --load-pretrained-decoder-from /mnt/data/bpop/wmt-multi/flores_big_task2_vocab/model.pt \
    --homogeneous-batch \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "entmax-alpha": 1.5}' \
    --eval-bleu-remove-bpe "sentencepiece"
