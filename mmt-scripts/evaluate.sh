HYP_PATH=$1  # should end in generate-test.txt
GOLD_PATH=$2  # a file from flores101_dataset, for example

cat $HYP_PATH | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sacrebleu $GOLD_PATH --tokenize spm

