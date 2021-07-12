#!/usr/bin/env python

# input: a stream of vocabulary entries (with counts) as extracted from
# spm_encode --generate_vocabulary

import sys
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_format", default="tab")
parser.add_argument("--out_format", default="tab")
parser.add_argument("--min_count", default=1, type=int)
opt = parser.parse_args()

total_counts = Counter()

input_sep = "\t" if opt.in_format == "tab" else " "
output_sep = "\t" if opt.out_format == "tab" else " "

for line in sys.stdin:
    word, count = line.rstrip().split(input_sep)
    total_counts[word] += int(count)

for word, count in total_counts.most_common():
    if count < opt.min_count:
        break
    sys.stdout.write(output_sep.join([word, str(count)]) + "\n")

