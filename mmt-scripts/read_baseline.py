#!/usr/bin/env python

import sys
import re
from collections import defaultdict

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
    for s in languages:
        # print a line for each
        trg_bleus = [results[s][t] if t != s else "-" for t in languages]
        print(" & ".join([s] + trg_bleus) + "\\\\")


if __name__ == "__main__":
    eur = ["en", "et", "hr", "hu", "mk", "sr"]
    sea = ["en", "id", "jv", "ms", "ta", "tl"]
    results = parse_results(sys.stdin)
    print_results(results, eur)
    print_results(results, sea)
            
