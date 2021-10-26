#!/usr/bin/env python

import argparse
import os
import sentencepiece as spm


def read_vocab(path):
    # is it possible to do a vocab with counts?
    with open(path) as f:
        vocab = {line.split(" ")[0] for line in f}
        for symbol in ["<s>", "</s>", "<unk>"]:
            if symbol in vocab:
                vocab.remove(symbol)
        return vocab


def main(args):
    if args.baseline_vocab is not None:
        foo = None
    language = args.corpus.split(".")[-1]
        
    for vocab_size in args.vocab_sizes:
        model_dir = "vocabs/{}/{}/vocab-{}".format(language, args.model_type, vocab_size)
        os.makedirs(model_dir, exist_ok=True)
        prefix = os.path.join(model_dir, "m")
        spm.SentencePieceTrainer.train(
            input=args.corpus,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type=args.model_type
        )
        if args.validate:
            # now, compute some stats about this vocab and write it to a file in
            # the same directory. Right?
            # The question is what kind of statistics to compute
            valid_corpus = args.validate_corpus if args.validate_corpus is not None else args.corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="training corpus for segmentation")
    parser.add_argument("--valid_corpus", default=None, help="""
                        corpus for computing model stats (if none, use train
                        corpus)""")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--baseline_vocab", default=None)  # used for computing overlap
    parser.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--vocab_sizes", type=int, nargs="+")
    opt = parser.parse_args()
    main(opt)
