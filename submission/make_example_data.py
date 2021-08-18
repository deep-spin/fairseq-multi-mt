#!/usr/bin/env python

from os.path import join
import sys

flores_path = sys.argv[1] # path to dev data
languages = ["eng", "ind", "jav", "msa", "tam", "tgl"]

n_examples = 100
multitext = [{"uid": str(i)} for i in range(n_examples)]

for language in languages:
    with open(join(flores_path, language + ".dev")) as f:
        for i in range(n_examples):
            lang_ex = f.readline().strip()
            multitext[i][language] = lang_ex

for example in multitext:
    print(example)
