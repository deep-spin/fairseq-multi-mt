#!/usr/bin/env python

from itertools import combinations
import sys

pairs = combinations((line.strip() for line in sys.stdin), 2)

for pair in pairs:
    sys.stdout.write("-".join(pair) + "\n")
