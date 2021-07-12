#!/usr/bin/env python

import argparse
import sys
import torch


def read_vocab(path):
    with open(path) as f:
        return [line.rstrip().split(" ")[0] for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dict")
    opt = parser.parse_args()

    vocab = read_vocab(opt.dict)
    m = torch.load(opt.model)  # a dict
    # we assume there is only the one embedding matrix
    src_emb_matrix = m['model']['encoder.embed_tokens.weight']
    V, d = src_emb_matrix.size()

    sys.stderr.write("Embedding matrix size {}, vocab size {}\n".format(V, len(vocab)))
    for i in range(V):
        word_type = vocab[i]
        vec = [str(x_j) for x_j in src_emb_matrix[i].tolist()]
        sys.stdout.write(" ".join([word_type] + vec) + "\n")
        

if __name__ == "__main__":
    main()
