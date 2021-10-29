#!/usr/bin/env python

"""
Inputs to this script are:
- a model
- a new vocabulary (one type per line)
- parameters for what to do about unseen vocabulary
"""

import argparse
import sys
import os
from os.path import join
import re
from itertools import count

import torch


def read_vocab(path, delim):
    specials = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(path) as f:
        # move the specials to the front
        # delimiter could be tab or space (spm does one, fairseq the other)
        types = [line.rstrip().split(delim)[0] for line in f]
        return specials + [t for t in types if t not in specials]


def export_vocab(model_path, old_dict, new_dict, new_delim="\t"):
    """
    Inputs:
    - a model (.pt file path)
    - old_dict (the model's existing dictionary file, in two columns, second
    column is frequency or log likelihood, depending on the type of the spm
    - new_dict: a new vocabulary. For overlapping types, keep old embeddings.
            For new types, do something else (easiest is random initialization)

    model_dict should be the vocabulary that the model uses:
    V should equal this length of this vocab (throw an error if it doesn't)
    """
    # which old indices do we keep?
    old_i2s = read_vocab(old_dict, " ")

    # The language embeddings should be kept whether they are in new_dict or
    # not. Therefore, they need to be added to the new vocabulary
    lang_emb = {t for t in old_i2s if re.search(r'__[a-z][a-z]+__', t)}

    # old_s2i: map strings to indices in the original vocabulary
    old_s2i = {t: i for i, t in enumerate(old_i2s)}
    # get the langu

    # new_vocab = set(read_vocab(new_dict, new_delim))
    new_i2s = read_vocab(new_dict, new_delim)
    new_s2i = {t: i for i, t in enumerate(new_i2s)}
    # The language embeddings should be kept whether they are in new_dict or
    # not. Therefore, they need to be added to the new vocabulary
    raw_new_size = len(new_i2s)
    new_lang_ix = count(len(new_i2s))
    for lang in lang_emb:
        if lang not in new_s2i:
            new_i2s.append(lang)
            new_s2i[lang] = next(new_lang_ix)
    sys.stderr.write("Added {} new language embedding types to the new vocabulary\n".format(len(new_i2s) - raw_new_size))

    model = torch.load(model_path)  # a dict
    old_matrix = model['model']['encoder.embed_tokens.weight']  # shared
    del model
    old_vocab_size, dim = old_matrix.size()
    sys.stderr.write("Original embedding matrix size {}\n".format(old_vocab_size))
    sys.stderr.write("Original vocab size {}\n".format(len(old_i2s)))
    new_vocab_size = len(new_i2s)
    # instantiate the new embedding matrix randomly.
    new_matrix = old_matrix.new_empty(new_vocab_size, dim)
    torch.nn.init.normal_(new_matrix, mean=0, std=dim ** -0.5)

    # Iterate over new vocabulary. If the type exists in the old vocabulary,
    # use it instead of the random embedding.
    # We could also be clever to avoid the for-loop, but I'm sick of cleverness
    for i, t in enumerate(new_i2s):
        if t in old_s2i:
            old_i = old_s2i[t]
            new_matrix[i] = old_matrix[old_i]
    return new_matrix, new_i2s


def write_vocab(word_types, path):
    """
    word_types: the vocabulary, presented as a list, already including all
            special characters
    """
    specials = {"<s>", "<pad>", "</s>", "<unk>"}
    with open(path, "w") as f:
        for word_type in word_types:
            if word_type not in specials:
                f.write(" ".join([word_type, "1"]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("old_dict")
    parser.add_argument("new_dict")
    parser.add_argument("out_dir")
    parser.add_argument("--new_delim", default="tab", choices=["tab", "space"])
    opt = parser.parse_args()
    new_delim = "\t" if opt.new_delim == "tab" else " "

    # 1. make out_dir: it's where the resulting dictionary and pt files will go
    sys.stderr.write("Making new model directory at {}\n".format(opt.out_dir))
    os.mkdir(opt.out_dir)

    # 2. Read the old model's embedding matrix. Keep rows that correspond to
    #    the new vocabulary.
    # 3. Add new rows for any items in the new vocabulary that the old vocabulary
    #    does not cover. Initially, just do this randomly.
    sys.stderr.write("Extracting embedding matrix from {}\n".format(opt.model))
    new_emb, new_vocab = export_vocab(
        opt.model, opt.old_dict, opt.new_dict, new_delim=new_delim
    )

    # Write the new vocabulary to the output path
    vocab_outpath = join(opt.out_dir, "dict.txt")
    sys.stderr.write("Writing vocab to {}\n".format(vocab_outpath))
    write_vocab(new_vocab, vocab_outpath)

    # splice new_emb onto the existing model (import some code from fairseq?)
    model_out = join(opt.out_dir, "model.pt")
    sys.stderr.write("Reloading model and saving it to {}\n".format(model_out))
    model = torch.load(opt.model)
    model['model']['encoder.embed_tokens.weight'] = new_emb
    model['model']['decoder.embed_tokens.weight'] = new_emb
    # is this sufficient? I hope so.
    torch.save(model, model_out)


if __name__ == "__main__":
    main()
