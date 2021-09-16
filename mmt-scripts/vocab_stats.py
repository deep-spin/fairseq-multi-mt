#!/usr/bin/env python

import argparse
import sentencepiece as spm


def read_corpus(path):
    with open(path) as f:
        return [line.strip() for line in f]


def word_count(corpus):
    return sum(len(sent) for sent in corpus)


def average_length(corpus):
    return word_count(corpus) / len(corpus)



def pieces_per_word(corpus):
    # needs to be spm-encoded already
    space_char = "▁"
    pieces = word_count(corpus)
    words = sum(sum(t.startswith(space_char) for t in sent) for sent in corpus)
    return pieces / words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--models", nargs="+")
    opt = parser.parse_args()

    raw_corpus = read_corpus(opt.corpus)
    tokenized_corpus = [sent.split() for sent in raw_corpus]
    # words per sequence
    unseg_len = average_length(tokenized_corpus)

    # for each model
    stats = dict()
    for model in opt.models:
        encoder = spm.SentencePieceProcessor(model_file=model)
        segmented_corpus = [encoder.encode(sent, out_type=str)
                            for sent in raw_corpus]
        seg_len = average_length(segmented_corpus)
        avg_pieces = seg_len / unseg_len
        stats[model] = avg_pieces
    ranking = sorted(stats.items(), key=lambda x: x[1])
    for model in ranking:
        print(model)


if __name__ == "__main__":
    main()
