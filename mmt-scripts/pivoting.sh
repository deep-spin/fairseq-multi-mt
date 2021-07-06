#MODEL_PATH=/home/bpop/flores101_mm100_175M/
MODEL_PATH=$1
FLORES_MODEL_PATH=$2
RESULTS_PATH=$3
PIVOT=$4
DICT=$FLORES_MODEL_PATH/dict.txt
FLORES_PATH=/home/bpop/flores101_dataset/
FAIRSEQ_PATH=/home/bpop/fairseq-multi-mt/
SCRIPTS_PATH=$FAIRSEQ_PATH/mmt-scripts/
BEAM=5


# select pivot language
PIVOT_DIR="pivoting"

mkdir -p $PIVOT_DIR/data/$PIVOT
mkdir -p $PIVOT_DIR/data/$PIVOT/spm
mkdir -p $PIVOT_DIR/data/$PIVOT/bin
mkdir -p $PIVOT_DIR/results/$PIVOT/

LANGUAGES=$( ls $RESULTS_PATH | cut -f 3 -d "." | sort | uniq )

# find the results for translating into $PIVOT (this presupposes you've done all the direct translations, so you have that),
for SRC in $LANGUAGES ; do
    if [ $SRC != $PIVOT ] ; then
        for TGT in $LANGUAGES ; do
            if [ $TGT != $SRC ] && [ $TGT != $PIVOT ] ; then
                echo $SRC-$TGT
                # decide what results_dir should be.
                # resegment and copy src-pivot hypotheses to pivoting/hyp-data/$PIVOT/spm/$SRC.$PIVOT-$TGT.$PIVOT
                results_dir=$PIVOT_DIR/results/$PIVOT/hypotheses.$SRC-$TGT.$TGT
                cat $RESULTS_PATH/hypotheses.$SRC-$PIVOT.$PIVOT/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | python $FAIRSEQ_PATH/scripts/spm_encode.py --model $FLORES_MODEL_PATH/sentencepiece.bpe.model > $PIVOT_DIR/data/$PIVOT/spm/$SRC.$PIVOT-$TGT.$PIVOT
                FLORES_TGT=$(echo $TGT | python $SCRIPTS_PATH/flores2m2m.py flores)
                fairseq-preprocess -s $PIVOT -t $TGT --testpref $PIVOT_DIR/data/$PIVOT/spm/$SRC.$PIVOT-$TGT --destdir $PIVOT_DIR/data/$PIVOT/bin/$SRC.$PIVOT-$TGT --srcdict $DICT --tgtdict $DICT --only-source 1>&2
                fairseq-generate $PIVOT_DIR/data/$PIVOT/bin/$SRC.$PIVOT-$TGT --batch-size 32 --path $MODEL_PATH --fixed-dictionary $DICT -s $PIVOT -t $TGT --remove-bpe 'sentencepiece' --beam $BEAM --task translation_multi_simple_epoch --lang-pairs $FLORES_MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test --results-path $results_dir | tail -1
                cat $results_dir/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $FLORES_PATH/devtest/$FLORES_TGT.devtest --tokenize spm
            fi
        done
    fi
done
