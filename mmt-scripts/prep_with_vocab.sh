#!/bin/bash

# these should be alphabetized, otherwise no paths will match
PAIR=$1
MODEL_PATH=$2
TRAIN_PATH=$3
FLORES_PATH=$4
DATA_DIR=$5
VOCAB_PATH=$6
DICT=$7

LANG1=$( echo $PAIR | cut -f 1 -d "-")
LANG2=$( echo $PAIR | cut -f 2 -d "-")

mkdir -p $DATA_DIR/spm
mkdir -p $DATA_DIR/bin


FAIRSEQ_PATH=~/fairseq-multi-mt
SCRIPTS_PATH=$FAIRSEQ_PATH/mmt-scripts

TRAINPREF=$DATA_DIR/spm/train.$LANG1-$LANG2
VALIDPREF=$DATA_DIR/spm/dev.$LANG1-$LANG2
TESTPREF=$DATA_DIR/spm/test.$LANG1-$LANG2

# segment corpora and write them to DATA_DIR
# train corpora are specific to each language PAIR (en data is different for en-ta than en-sr, for example)
for lang in $LANG1 $LANG2 ; do
    cat $TRAIN_PATH/*$LANG1-$LANG2.$lang | python $FAIRSEQ_PATH/scripts/spm_encode.py --model $MODEL_PATH/sentencepiece.bpe.model > $TRAINPREF.$lang
done

# dev/devtest data are NOT pair-dependent (it's the same English set no matter what the other language is)
for lang in $LANG1 $LANG2 ; do
    flores_lang=$(echo $lang | python $SCRIPTS_PATH/flores2m2m.py flores)
    for split in dev devtest ; do
        outsplit=$(echo $split | sed "s/devtest/test/g")
        cat $FLORES_PATH/$split/$flores_lang.$split | python $FAIRSEQ_PATH/scripts/spm_encode.py --model $MODEL_PATH/sentencepiece.bpe.model > $DATA_DIR/spm/$outsplit.$LANG1-$LANG2.$lang
    done
done

# if you've made it this far, you have the segmented corpora for a language pair in DATA_DIR
# The next step is to call fairseq-preprocess on it
destdir=$DATA_DIR/bin/$LANG1-$LANG2
fairseq-preprocess \
    --source-lang $LANG1 --target-lang $LANG2 \
    --trainpref $TRAINPREF --validpref $VALIDPREF --testpref $TESTPREF \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir $destdir --srcdict $DICT --tgtdict $DICT
