#!/usr/bin/env python

import sys
import re
import argparse
from collections import defaultdict


def n_directions(n):
    return n * (n - 1)


def parse_results(lines):

    results = defaultdict(dict)

    src, trg = None, None
    for line in lines:
        line = line.strip()
        if line:
            if not line.startswith("BLEU"):
                try:
                    src, trg = line.split()
                except ValueError:
                    src, trg = line.split("-")
            else:
                assert src is not None and trg is not None
                # parse BLEU score
                # bleu = re.search(r'(?<=BLEU4 = )[^,]*', line).group(0)
                bleu = re.search(r'(?<=BLEU\+case\.mixed\+numrefs\.1\+smooth\.exp\+tok\.spm\+version\.1\.5\.0 = )([0-9]|\.)*', line).group(0)
                results[src][trg] = bleu
    return results


def print_results(results, languages):
    bleu_sum = 0.0
    for s in languages:
        # print a line for each
        trg_bleus = [results[s][t] if t != s else "-" for t in languages]
        bleu_sum += sum(float(tb) if tb != "-" else 0.0 for tb in trg_bleus)
        print(" & ".join([s] + trg_bleus) + "\\\\")
    # now get the mean
    print(bleu_sum / n_directions(len(languages)))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--pivot_pairs", default=None)
    opt = parser.parse_args()
    task_langs = {"eur": ["en", "et", "hr", "hu", "mk", "sr"],
                  "sea": ["en", "id", "jv", "ms", "ta", "tl"]}
    assert all(task in task_langs for task in opt.tasks)
    results = parse_results(sys.stdin)
    if opt.pivot_pairs is not None:
        with open(opt.pivot_pairs) as f:
            pivot_results = parse_results(f)
        for src, src_dict in pivot_results.items():
            for trg, score in src_dict.items():
                results[src][trg] = score
    for task in opt.tasks:
        print(task)
        print_results(results, task_langs[task])
