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

import torch


def read_vocab(path):
    specials = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(path) as f:
        # move the specials to the front
        types = [line.rstrip().split(None, 1)[0] for line in f]
        return specials + [t for t in types if t not in specials]


def export_vocab(model_path, old_dict, new_dict):
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
    old_vocab = read_vocab(old_dict)
    new_vocab = set(read_vocab(new_dict))
    language_embeddings = {t for t in old_vocab if t.startswith("__")}
    new_vocab.update(language_embeddings)

    # other things that need to be kept: language embedding
    kept_ix = [i for i, t in enumerate(old_vocab) if t in new_vocab]
    kept_vocab = [t for t in old_vocab if t in new_vocab]
    new_types = list(new_vocab - set(old_vocab))

    model = torch.load(model_path)  # a dict
    # we assume there is only the one embedding matrix
    src_emb_matrix = model['model']['encoder.embed_tokens.weight']
    del model
    V, d = src_emb_matrix.size()
    sys.stderr.write("Original embedding matrix size {}\n".format(V))
    sys.stderr.write("Original vocab size {}\n".format(len(old_vocab)))

    kept_emb = src_emb_matrix[kept_ix]

    # add new types at the end
    unseen_emb = kept_emb.new_empty(len(new_types), d)
    torch.nn.init.normal_(unseen_emb, mean=0, std=d ** -0.5)
    new_emb = torch.cat([kept_emb, unseen_emb])
    final_vocab = kept_vocab + new_types
    assert len(final_vocab) == new_emb.size(0), \
        "Embeddings and vocab do not match"
    return new_emb, final_vocab


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
    opt = parser.parse_args()

    # 1. make out_dir: it's where the resulting dictionary and pt files will go
    sys.stderr.write("Making new model directory at {}\n".format(opt.out_dir))
    os.mkdir(opt.out_dir)

    # 2. Read the old model's embedding matrix. Keep rows that correspond to
    #    the new vocabulary.
    # 3. Add new rows for any items in the new vocabulary that the old vocabulary
    #    does not cover. Initially, just do this randomly.
    sys.stderr.write("Extracting embedding matrix from {}\n".format(opt.model))
    new_emb, new_vocab = export_vocab(opt.model, opt.old_dict, opt.new_dict)

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
