#!/usr/bin/env python

"""
Although filter_corpora.py provides the --check_langid option, this is very
slow because it does not support performing langid in batches. this script
requires my fork of langid, which adds an additional classify_batch function.
"""

import sys
import argparse
import langid  # https://github.com/bpopeters/langid.py


parser = argparse.ArgumentParser()
parser.add_argument("lang1")
parser.add_argument("lang2")
parser.add_argument("--batch_size", type=int, default=256)
opt = parser.parse_args()


def filter_batch(batch, lang1, lang2):
    seq1_batch, seq2_batch = zip(*[b_line.split("\t") for b_line in batch])
    lang1_preds = langid.classify_batch(seq1_batch)
    lang2_preds = langid.classify_batch(seq2_batch)
    for lang1_pred, lang2_pred, b_line in zip(lang1_preds, lang2_preds, batch):
        seq1_score, seq1_lang = lang1_pred
        seq2_score, seq2_lang = lang2_pred
        if seq1_lang == lang1 and seq2_lang == lang2:
            yield b_line
            # sys.stdout.write(b_line)

# now...we need to iterate over sys.stdin in batches
num_written = 0
lang1_batch, lang2_batch = [], []
batch = []
for i, line in enumerate(sys.stdin):
    # a line should be tab-separated parallel corpora, such as you would get
    # from calling paste
    stripped_line = line.strip()
    if stripped_line:
        batch.append(line)

        if len(batch) == opt.batch_size:
            # this is where the magic happens
            for filt_line in filter_batch(batch, opt.lang1, opt.lang2):
                sys.stdout.write(filt_line)
                num_written += 1
            sys.stderr.write("Seen {}, kept {}\n".format(i, num_written))
            batch = []

if batch:
    for filt_line in filter_batch(batch, opt.lang1, opt.lang2):
        sys.stdout.write(filt_line)
        num_written += 1
        sys.stderr.write("Seen {}, kept {}\n".format(i, num_written))
