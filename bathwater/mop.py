#!/usr/bin/env python

"""
Minimally-Obtrusive Pruning: given a corpus which has already been segmented,
return the subset of its vocabulary containing only types which occur at least
k times in the segmented corpus.
"""

import sys
import argparse
from collections import Counter

from change_vocab import write_vocab


def main(args):
    counts = Counter()
    for line in sys.stdin:
        counts.update(line.strip().split())
    # now: filter
    filtered_counts = Counter(
        {t: count for t, count in counts.items() if count >= args.k}
    )

    # will this have complete coverage over the alphabet? no guarantee.
    sorted_vocab = [t for t, count in filtered_counts.most_common()]
    write_vocab(sorted_vocab, opt.out)

    # so, we need a method that takes a model, vocabulary, and corpus, and
    # returns some statistics about the corpus (such as whether it differs
    # from the original)

    # so, we can use SentencePieceProcessor's set_vocabulary method (argument
    # is a list) in order to segment differently.
    # I think it would be best to evaluate on the dev sets.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("-k", type=int, default=5)
    opt = parser.parse_args()
    main(opt)
