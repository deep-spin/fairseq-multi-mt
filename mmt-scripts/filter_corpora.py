#!/usr/bin/env/python

import sys

import argparse
import langid

parser = argparse.ArgumentParser()
parser.add_argument("lang1")
parser.add_argument("lang2")
parser.add_argument("--min_length", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)  # what would be sensible
parser.add_argument("--filter_identical", action="store_true")
parser.add_argument("--check_langid", action="store_true")
opt = parser.parse_args()

num_written = 0
for i, line in enumerate(sys.stdin):
    # a line should be tab-separated parallel corpora, such as you would get
    # from calling paste
    stripped_line = line.strip()
    if stripped_line:
        seq1, seq2 = stripped_line.split("\t")  # could generalize to more than 2?

        if opt.filter_identical and seq1 == seq2:
            # remove identical
            continue
        
        split_seq1, split_seq2 = seq1.split(), seq2.split()
        if len(split_seq1) < opt.min_length or len(split_seq2) < opt.min_length:
            continue
        if len(split_seq1) > opt.max_length or len(split_seq2) > opt.max_length:
            continue

        if opt.check_langid:
            # this will slow the code down a lot.
            # Therefore it would be better to handle quicker forms of filtering
            # first
            seq1_lang, seq1_score = langid.classify(seq1)
            if seq1_lang != opt.lang1:
                continue
            seq2_lang, seq2_score = langid.classify(seq2)
            if seq2_lang != opt.lang2:
                continue

        num_written += 1
        
        if num_written > 0 and num_written % 100 == 0:
            sys.stderr.write("Seen {}, kept {}\n".format(i, num_written))
    sys.stdout.write(line)
