#!/usr/bin/env python

from os.path import join
from itertools import permutations
from random import shuffle
import pickle
import argparse


def create_examples(multitext_ex, n_pairs):
    # shuffle the languages in the example, yield n_pairs permutations
    languages = [k for k in multitext_ex if k != "uid"]
    assert n_pairs <= len(languages) * (len(languages) - 1)
    shuffle(languages)
    all_pairs = permutations(languages, 2)
    for i in range(n_pairs):
        src, tgt = next(all_pairs)
        ret = {"uid": multitext_ex["uid"]}
        ret["sourceLanguage"] = src
        ret["targetLanguage"] = tgt
        ret["sourceText"] = multitext_ex[src]
        yield ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flores_path")
    parser.add_argument("out_path")
    parser.add_argument("--n_examples", dtype=int, default=100)
    parser.add_argument("--pairs_per", dtype=int, default=2)
    opt = parser.parse_args()
    languages = ["eng", "ind", "jav", "msa", "tam", "tgl"]

    multitext = [{"uid": str(i)} for i in range(opt.n_examples)]

    for language in languages:
        with open(join(opt.flores_path, language + ".dev")) as f:
            for i in range(opt.n_examples):
                lang_ex = f.readline().strip()
                multitext[i][language] = lang_ex

    output = []
    for example in multitext:
        for ex in create_examples(example, opt.pairs_per):
            output.append(ex)
    with open(opt.out_path, "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()
