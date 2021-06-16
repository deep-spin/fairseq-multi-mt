#!/usr/bin/env python

import sys
import re
from collections import defaultdict


def n_directions(n):
    return n * (n - 1)


def parse_results(lines):

    results = defaultdict(dict)

    src, trg = None, None
    for line in lines:
        line = line.strip()
        if line:
            if not line.startswith("Generate"):
                src, trg = line.split()
            else:
                assert src is not None and trg is not None
                # parse BLEU score
                bleu = re.search(r'(?<=BLEU4 = )[^,]*', line).group(0)
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
    eur = ["en", "et", "hr", "hu", "mk", "sr"]
    sea = ["en", "id", "jv", "ms", "ta", "tl"]
    results = parse_results(sys.stdin)
    print_results(results, eur)
    print_results(results, sea)
            
