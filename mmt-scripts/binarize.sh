LANGUAGES=$1
MODEL_PATH=/home/bpop/flores101_mm100_175M/
DICT=$MODEL_PATH/dict.txt
FLORES_PATH=/home/bpop/flores101_dataset/
FAIRSEQ_PATH=..

mkdir -p ../data
mkdir -p ../data-bin

# this part should probably be done by a different script
for SPLIT in dev devtest ; do

    # segment each language's dev and devtest
    for LANGUAGE in $( cat $LANGUAGES ) ; do
        inp=$FLORES_PATH/$SPLIT/$LANGUAGE.$SPLIT
        outp=$FAIRSEQ_PATH/data/spm.$SPLIT.$LANGUAGE
        echo "encoding"
        echo $inp
        echo $outp
        spm_encode --model=$MODEL_PATH/sentencepiece.bpe.model --output_format=piece < $inp > $outp
        echo $LANGUAGE
    done
done

testpref=$FAIRSEQ_PATH/data/spm.devtest
for PAIR in $( cat $LANGUAGES | python pairs.py) ; do
    LANG1=$(echo $PAIR | cut -f 1 -d "-")
    LANG2=$(echo $PAIR | cut -f 2 -d "-")
    
    # now binarize
    destdir=$FAIRSEQ_PATH/data-bin/$PAIR/
    fairseq-preprocess --source-lang $LANG1 --target-lang $LANG2 --testpref $testpref --thresholdsrc 0 --thresholdtgt 0 --destdir $destdir --srcdict $DICT --tgtdict $DICT
    # fairseq-preprocess --source-lang $LANG2 --target-lang $LANG1 --testpref $testpref --thresholdsrc 0 --thresholdtgt 0 --destdir $destdir --srcdict $DICT --tgtdict $DICT
done
