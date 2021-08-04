#MODEL_PATH=/home/bpop/flores101_mm100_175M/
FLORES_MODEL_PATH=$1
DATA_BIN_PATH=$2
RESULTS_PATH=$3
ALPHA=$4
DICT=$FLORES_MODEL_PATH/dict.txt
FLORES_PATH=/home/bpeters/wmt-multi/flores101_dataset/
SCRIPTS_PATH=/home/bpeters/fairseq-multi-mt/mmt-scripts/
BEAM=5

CHECKPOINTS=/home/bpeters/wmt-multi/x-centric-checkpoints

mkdir -p $RESULTS_PATH

PAIRS=$(ls $DATA_BIN_PATH/test* | cut -f 2 -d "." | sort | uniq)

for PAIR in $PAIRS ; do
    LANG1=$(basename $PAIR | cut -f 1 -d "-")
    LANG2=$(basename $PAIR | cut -f 2 -d "-")
    FLORES_LANG1=$(echo $LANG1 | python $SCRIPTS_PATH/flores2m2m.py flores)
    FLORES_LANG2=$(echo $LANG2 | python $SCRIPTS_PATH/flores2m2m.py flores)
    echo $LANG1 $LANG2
    LANG1_CENTRIC=$CHECKPOINTS/$LANG1/checkpoint_last.pt
    LANG2_CENTRIC=$CHECKPOINTS/$LANG2/checkpoint_last.pt
    LANG1_SRC=$CHECKPOINTS/unidirectional/src/$LANG1/checkpoint_last.pt
    LANG2_TGT=$CHECKPOINTS/unidirectional/tgt/$LANG2/checkpoint_last.pt
    results_dir=$RESULTS_PATH
    fairseq-generate $DATA_BIN_PATH --entmax-alpha $ALPHA --batch-size 32 --path $LANG1_CENTRIC:$LANG2_CENTRIC:$LANG1_SRC:$LANG2_TGT --fixed-dictionary $DICT -s $LANG1 -t $LANG2 --remove-bpe 'sentencepiece' --beam $BEAM --task translation_multi_simple_epoch --lang-pairs $FLORES_MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test --results-path $results_dir/$LANG1-$LANG2-x-centric-ensemble | tail -1
    cat $results_dir/$LANG1-$LANG2-x-centric-ensemble/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $FLORES_PATH/devtest/$FLORES_LANG2.devtest --tokenize spm
    echo $LANG2 $LANG1
    results_dir=$RESULTS_PATH
    LANG1_TGT=$CHECKPOINTS/unidirectional/tgt/$LANG1/checkpoint_last.pt
    LANG2_SRC=$CHECKPOINTS/unidirectional/src/$LANG2/checkpoint_last.pt
    fairseq-generate $DATA_BIN_PATH --entmax-alpha $ALPHA --batch-size 32 --path $LANG1_CENTRIC:$LANG2_CENTRIC:$LANG1_TGT:$LANG2_SRC --fixed-dictionary $DICT -s $LANG2 -t $LANG1 --remove-bpe 'sentencepiece' --beam $BEAM --task translation_multi_simple_epoch --lang-pairs $FLORES_MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test --results-path $results_dir/$LANG2-$LANG1-x-centric-ensemble | tail -1
    cat $results_dir/$LANG2-$LANG1-x-centric-ensemble/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $FLORES_PATH/devtest/$FLORES_LANG1.devtest --tokenize spm
    echo
done
