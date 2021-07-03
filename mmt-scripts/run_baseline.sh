#MODEL_PATH=/home/bpop/flores101_mm100_175M/
MODEL_PATH=$1
FLORES_MODEL_PATH=$2
DATA_BIN_PATH=$3
RESULTS_PATH=$4
DICT=$FLORES_MODEL_PATH/dict.txt
FLORES_PATH=/home/bpop/flores101_dataset/
SCRIPTS_PATH=/home/bpop/fairseq-multi-mt/mmt-scripts/
BEAM=5

mkdir -p $RESULTS_PATH

PAIRS=$(ls $DATA_BIN_PATH/test* | cut -f 2 -d "." | sort | uniq)

for PAIR in $PAIRS ; do
    LANG1=$(basename $PAIR | cut -f 1 -d "-")
    LANG2=$(basename $PAIR | cut -f 2 -d "-")
    FLORES_LANG1=$(echo $LANG1 | python $SCRIPTS_PATH/flores2m2m.py flores)
    FLORES_LANG2=$(echo $LANG2 | python $SCRIPTS_PATH/flores2m2m.py flores)
    echo $LANG1 $LANG2
    results_dir=$RESULTS_PATH/hypotheses.$LANG1-$LANG2.$LANG2
    fairseq-generate $DATA_BIN_PATH --batch-size 32 --path $MODEL_PATH --fixed-dictionary $DICT -s $LANG1 -t $LANG2 --remove-bpe 'sentencepiece' --beam $BEAM --task translation_multi_simple_epoch --lang-pairs $FLORES_MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test --results-path $results_dir | tail -1
    cat $results_dir/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $FLORES_PATH/devtest/$FLORES_LANG2.devtest --tokenize spm
    echo $LANG2 $LANG1
    results_dir=$RESULTS_PATH/hypotheses.$LANG2-$LANG1.$LANG1
    fairseq-generate $DATA_BIN_PATH --batch-size 32 --path $MODEL_PATH --fixed-dictionary $DICT -s $LANG2 -t $LANG1 --remove-bpe 'sentencepiece' --beam $BEAM --task translation_multi_simple_epoch --lang-pairs $FLORES_MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test --results-path $results_dir | tail -1
    cat $results_dir/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $FLORES_PATH/devtest/$FLORES_LANG1.devtest --tokenize spm
    echo
done
