#!/usr/bin/env python

import argparse
import sys


def read_vocab(path):
    vocab = {"<s>", "<pad>", "</s>", "<unk>"}
    with open(path) as f:
        for line in f:
            vocab.add(line.rstrip().split(" ")[0])
    return vocab


# reading the text file into memory seems bad, it's probably better to do this
# directly from the model's embedding matrix
def read_embeddings(path):
    with open(path) as f:
        embs = dict()
        old_V, d = f.readline().strip().split(" ")
        for line in f:
            word_type = line.split(" ", 1)[0]
            embs[word_type] = line
        return embs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("new_dict")
    opt = parser.parse_args()

    emb_matrix = read_embeddings(opt.embeddings)

    # so, the embeddings are in a huge file, where each line (except the first)
    #
    for special in ["<s>", "<pad>", "</s>", "<unk>"]:
        sys.stdout.write(emb_matrix[special])
    with open(opt.new_dict) as f:
        for line in f:
            word_type = line.split(" ", 1)[0]
            sys.stdout.write(emb_matrix[word_type])


if __name__ == "__main__":
    main()
