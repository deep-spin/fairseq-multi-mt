#!/usr/bin/env python

"""
Minimally-Obtrusive Pruning: given a corpus which has already been segmented,
return the subset of its vocabulary containing only types which occur at least
k times in the segmented corpus.
"""

import sys
import argparse
from collections import Counter
from itertools import chain

from change_vocab import write_vocab


def main(args):
    # this should keep track of characters as well, because they should always
    # be included
    alphabet = set()
    counts = Counter()
    for line in sys.stdin:
        tokens = line.strip().split()
        counts.update(tokens)
        alphabet.update(chain.from_iterable(tokens))
    # now: filter
    filtered_counts = Counter(
        {t: count for t, count in counts.items() if count >= args.k}
    )
    # after filtering, add all symbols from the alphabet
    filtered_counts.update(alphabet)

    # will this have complete coverage over the alphabet? no guarantee.
    sorted_vocab = [t for t, count in filtered_counts.most_common()]
    write_vocab(sorted_vocab, opt.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("-k", type=int, default=5)
    opt = parser.parse_args()
    main(opt)
