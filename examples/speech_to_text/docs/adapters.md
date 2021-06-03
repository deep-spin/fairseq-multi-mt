# Lightweight Adapter Tuning for Multilingual Speech Translation

This is the codebase for the paper [Lightweight Adapter Tuning for Multilingual Speech Translation](https://arxiv.org/abs/2106.3771637) (ACL-IJCNLP 2021).


## Data Preparation
All our experiments were performed on MuST-C. Please refer to the data preparation steps provided by [`fairseq S2T`](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text) for the MuST-C dataset [here](https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/docs/mustc_example.md#data-preparation).


## Training
### Multilingual training
To train a multilingual ST backbone model, please run the following command:
```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
  --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
  --save-dir ${MULTILINGUAL_BACKBONE} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_m --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${PRETRAINED_ASR}
```
where:
- `${MUSTC_ROOT}` is the path to MuST-C data.
- `${MULTILINGUAL_BACKBONE}` is the path to save the outputs of the experiments.
- `${PRETRAINED_ASR}` is the path to the pretrained ASR model used to initialize the ST encoder.

### Adapter-based finetuning
To perform multilingual finetuning using adapters, please run the following command:
```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
  --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
  --lang-pairs en-de,en-es,en-fr,en-it,en-nl,en-pt,en-ro,en-ru \
  --save-dir ${EXP_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_m --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --find-unused-parameters \
  --homogeneous-batch \
  --adapter-enc-dim 256 \
  --adapter-enc-type 'per_lang' \
  --adapter-dec-dim 256 \
  --adapter-dec-type 'per_lang' \
  --finetune-enc-modules adapter \
  --finetune-dec-modules adapter \
  --load-pretrained-encoder-from ${MULTILINGUAL_BACKBONE}/${CHECKPOINT_FILENAME} \
  --load-pretrained-decoder-from ${MULTILINGUAL_BACKBONE}/${CHECKPOINT_FILENAME}
```

### Full/Partial finetuning
For full finetuning on a specific language pair, for example, `en-de`:
```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st \
  --valid-subset dev_de_st \
  --save-dir ${EXP_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_m --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --finetune-from-model ${MULTILINGUAL_BACKBONE}/${CHECKPOINT_FILENAME}
```

For decoder-only finetuning:
```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st \
  --valid-subset dev_de_st \
  --save-dir ${EXP_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_m --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --finetune-dec-modules dropout_module,embed_tokens,embed_positions,layers,layer_norm,output_projection \
  --load-pretrained-encoder-from ${MULTILINGUAL_BACKBONE}/${CHECKPOINT_FILENAME} \
  --load-pretrained-decoder-from ${MULTILINGUAL_BACKBONE}/${CHECKPOINT_FILENAME}
```

## Inference and Evaluation
For decoding using trained models, run the following command:

```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${EXP_DIR} --num-epoch-checkpoints 10 \
  --output "${EXP_DIR}/${CHECKPOINT_FILENAME}"

for LANG in de nl es fr it pt ro ru; do
  fairseq-generate ${MUSTC_ROOT} \
    --config-yaml config_st.yaml --gen-subset tst-COMMON_${LANG}_st --task speech_to_text \
    --prefix-size 1 --path ${EXP_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring sacrebleu
done
```

# Results
| Model  | Params (M) (trainable/total) | En-De | En-Es | En-Fr | En-It | En-Nl | En-Pt | En-Ro | En-Ru |
|---|---|---|---|---|---|---|---|---|---|
| Multilingual baseline | 76.3/76.3 | 24.18 | 28.28 | **34.98** | 24.62 | **28.80** | **31.13** | 23.22 | 15.88 |
| Best adapting | 8 x 4.8/76.3 | **24.63** | **28.73** | 34.75 | **24.96** | **28.80** | 30.96 | 23.70 | **16.36** |
| Best finetuning | 8 x 35.5/8 x 76.3 | 24.50 | 28.67 | 34.89 | 24.82 | 28.38 | 30.73 | **23.78** | 16.23 |