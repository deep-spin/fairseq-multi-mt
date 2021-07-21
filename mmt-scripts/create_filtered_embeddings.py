#!/usr/bin/env python

import argparse
import sys
import numpy as np


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
        f.readline()
        for line in f:
            word_type = line.split(" ", 1)[0]
            embs[word_type] = line
        return embs


def get_embedding_dim(path):
    with open(path) as f:
        f.readline()
        return len(f.readline().split(" ")) - 1


def generate_new_embedding(word_type, dim):
    """
    This will need to be formatted the same way as a line
    """
    # nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    emb = [str(x) for x in np.random.normal(size=dim, loc=0, scale=dim**-0.5)]
    return " ".join([word_type] + emb) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("new_dict")
    opt = parser.parse_args()

    emb_dim = get_embedding_dim(opt.embeddings)

    emb_matrix = read_embeddings(opt.embeddings)
    language_embeddings = {k: v for k, v in emb_matrix.items() if k.startswith("__")}

    # so, the embeddings are in a huge file, where each line (except the first)
    #
    sys.stdout.write("stupid obligatory header\n")
    for special in ["<s>", "<pad>", "</s>", "<unk>"]:
        sys.stdout.write(emb_matrix[special])
    with open(opt.new_dict) as f:
        for line in f:
            word_type = line.split(" ", 1)[0]
            if word_type in emb_matrix:
                sys.stdout.write(emb_matrix[word_type])
            else:
                # create new embedding using same strategy as fairseq
                sys.stdout.write(generate_new_embedding(word_type, emb_dim))
        for k, v in language_embeddings.items():
            # make sure not to exclude the language embeddings, which are
            # included in the flores model's dict but not included in
            sys.stdout.write(v)


if __name__ == "__main__":
    main()
