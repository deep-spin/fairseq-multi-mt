MODEL_PATH=/home/bpop/flores101_mm100_175M/
DICT=$MODEL_PATH/dict.txt
FLORES_PATH=/home/bpop/flores_tasklangs/

for PAIR in data-bin/* ; do
    LANG1=$(echo $PAIR | cut -f 1 -d "-")
    LANG2=$(echo $PAIR | cut -f 2 -d "-")
    echo $LANG1 $LANG2
    fairseq-generate $PAIR --batch-size 16 --path $MODEL_PATH/model.pt --fixed-dictionary $DICT -s $LANG1 -t $LANG2 --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs $MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test | tail -1
    echo $LANG2 $LANG1
    fairseq-generate $PAIR --batch-size 16 --path $MODEL_PATH/model.pt --fixed-dictionary $DICT -s $LANG2 -t $LANG1 --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs $MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test | tail -1
    echo
done
