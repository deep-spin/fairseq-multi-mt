#!/usr/bin/env python

from collections import Counter
import sys


counts = Counter()
for line in sys.stdin:
    tokens = line.strip().split()
    counts.update(tokens)

ordered = counts.most_common()
print("Number of types: {}".format(len(counts)))
print("Number of tokens: {}".format(sum(counts.values())))
print("F95: {}".format(ordered[int(len(counts) * 0.95)]))
