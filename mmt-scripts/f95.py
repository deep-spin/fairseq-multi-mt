#!/usr/bin/env python

from collections import Counter
import sys


counts = Counter()
for line in sys.stdin:
    tokens = line.strip().split()
    counts.update(tokens)

ordered = counts.most_common()
index_with_100 = max(i for (i, (k, v)) in enumerate(ordered) if v >= 100)
print("Number of types: {}".format(len(counts)))
print("Number of tokens: {}".format(sum(counts.values())))
f_index = int(len(counts) * 0.95)
print("F95 index: {}".format(f_index))
print("F95: {}".format(ordered[f_index]))
print("index with 100: {}".format(index_with_100))
