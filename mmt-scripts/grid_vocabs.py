#!/usr/bin/env python

import argparse
from os.path import join, exists
import sentencepiece as spm


def read_vocab(path):
    # is it possible to do a vocab with counts?
    with open(path) as f:
        vocab = {line.split()[0] for line in f}
        for symbol in ["<s>", "</s>", "<unk>"]:
            if symbol in vocab:
                vocab.remove(symbol)
        return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("baseline_vocab")
    parser.add_argument("model_dir")
    parser.add_argument("--model_type", default="unigram")
    parser.add_argument("--vocab_sizes", type=int, nargs="+")
    opt = parser.parse_args()

    # run spm_train (through the python interface) on the same corpus several
    # times with varying hyperparameters
    # compare the similarity between this vocabulary and the original one
    baseline_vocab = read_vocab(opt.baseline_vocab)

    for vocab_size in opt.vocab_sizes:
        prefix = join(opt.model_dir, "{}-{}".format(opt.model_type, vocab_size))
        vocab_path = prefix + ".vocab"
        if not exists(vocab_path):
            # if it already exists, don't make it. Just print the results
            spm.SentencePieceTrainer.train(
                input=opt.corpus,
                model_prefix=prefix,
                vocab_size=vocab_size,
                model_type=opt.model_type
            )
        vocab = read_vocab(vocab_path)
        print(sum(v not in baseline_vocab for v in vocab), vocab_size)


if __name__ == "__main__":
    main()
