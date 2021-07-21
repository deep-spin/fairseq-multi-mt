#!/usr/bin/env python

"""
Given a Tamil corpus segmented according to some Tamil-specific technique, what
percentage of tokens (or types) are covered by the multilingual M2M-100
vocabulary?
"""

import argparse
from itertools import Counter


def count_tokens(corpora):
    counts = Counter()
    for corpus in corpora:
        with open(corpus) as f:
            for line in f:
                counts.update(line.strip().split())
    return counts


def read_dict(path):
    with open(path) as f:
        return {line.rstrip().split()[0] for line in f}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_dict")
    parser.add_argument("--corpora", nargs="*", help="arbitrarily many paths")
    opt = parser.parse_args()
    # so, given these corpora, count all of their subword occurrences
    pretrained_vocab = read_dict(opt.pretrained_dict)
    token_counts = count_tokens(opt.corpora)
    n_tokens = sum(token_counts.values())
    n_seen_tokens = sum(v for k, v in token_counts.items()
                        if k in pretrained_vocab)
    print("Pretrained vocab size: {}".format(len(pretrained_vocab)))
    print("Unique types with new segmentation: {}".format(len(token_counts)))
    print("Percentage of tokens seen in pretrained vocab: {}".format(n_seen_tokens / n_tokens * 100))


if __name__ == "__main__":
    main()
