#!/usr/bin/env python

from os.path import join
import sys
from itertools import permutations
from random import shuffle


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
    flores_path = sys.argv[1]  # path to dev data
    languages = ["eng", "ind", "jav", "msa", "tam", "tgl"]

    n_examples = 100
    multitext = [{"uid": str(i)} for i in range(n_examples)]

    for language in languages:
        with open(join(flores_path, language + ".dev")) as f:
            for i in range(n_examples):
                lang_ex = f.readline().strip()
                multitext[i][language] = lang_ex

    for example in multitext:
        for ex in create_examples(example, 3):
            print(ex)


if __name__ == "__main__":
    main()
