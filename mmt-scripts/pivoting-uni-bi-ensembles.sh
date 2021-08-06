FLORES_MODEL_PATH=$1
RESULTS_PATH=$2
PIVOT=$3
PIVOT_DIR=$4
DICT=$FLORES_MODEL_PATH/dict.txt
FLORES_PATH=/home/bpeters/wmt-multi/flores101_dataset/
FAIRSEQ_PATH=/home/bpeters/fairseq-multi-mt/
SCRIPTS_PATH=$FAIRSEQ_PATH/mmt-scripts/
BEAM=5

CHECKPOINTS=/home/bpeters/wmt-multi/x-centric-checkpoints

# select pivot language

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
                PIVOT_CENTRIC=$CHECKPOINTS/$PIVOT/checkpoint_last.pt
                TGT_CENTRIC=$CHECKPOINTS/$TGT/checkpoint_last.pt
                PIVOT_SRC=$CHECKPOINTS/unidirectional/src/$PIVOT/checkpoint_last.pt
                TGT_TGT=$CHECKPOINTS/unidirectional/tgt/$TGT/checkpoint_last.pt
                
                results_dir=$PIVOT_DIR/results/$PIVOT/hypotheses.$SRC-$TGT.$TGT
                cat $RESULTS_PATH/hypotheses.$SRC-$PIVOT.$PIVOT/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | python $FAIRSEQ_PATH/scripts/spm_encode.py --model $FLORES_MODEL_PATH/sentencepiece.bpe.model > $PIVOT_DIR/data/$PIVOT/spm/$SRC.$PIVOT-$TGT.$PIVOT
                FLORES_TGT=$(echo $TGT | python $SCRIPTS_PATH/flores2m2m.py flores)
                fairseq-preprocess -s $PIVOT -t $TGT --testpref $PIVOT_DIR/data/$PIVOT/spm/$SRC.$PIVOT-$TGT --destdir $PIVOT_DIR/data/$PIVOT/bin/$SRC.$PIVOT-$TGT --srcdict $DICT --tgtdict $DICT --only-source 1>&2
                fairseq-generate $PIVOT_DIR/data/$PIVOT/bin/$SRC.$PIVOT-$TGT --batch-size 32 --path $PIVOT_CENTRIC:$TGT_CENTRIC:$PIVOT_SRC:$TGT_TGT --fixed-dictionary $DICT -s $PIVOT -t $TGT --remove-bpe 'sentencepiece' --beam $BEAM --task translation_multi_simple_epoch --lang-pairs $FLORES_MODEL_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --skip-invalid-size-inputs-valid-test --results-path $results_dir | tail -1
                cat $results_dir/generate-test.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $FLORES_PATH/devtest/$FLORES_TGT.devtest --tokenize spm
            fi
        done
    fi
done
