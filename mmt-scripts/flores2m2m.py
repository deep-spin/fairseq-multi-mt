import sys
import re
from os.path import basename, dirname, join

# flores codes seem to be ISO 639-3. M2M codes seem to be ISO 639-1. In lieu
# of a principled solution, I have written a regular expression that works only
# for the set of languages included in small tasks 1 and 2.

def read_codes(path, direction):
    with open(path) as f:
        lookup = dict()
        for line in f:
            name, flores_code, mm_code = line.strip().split("\t")
            if direction == "mm":
                lookup[flores_code] = mm_code
            else:
                lookup[mm_code] = flores_code
    return lookup


if __name__ == "__main__":
    if len(sys.argv) > 1:
        direction = sys.argv[1]
        assert direction in ["flores", "mm"]
    else:
        direction = "mm"
    lookup = read_codes(
        join(dirname(sys.argv[0]), "flores101_codes.txt"), direction
    )
    pattern = '|'.join(sorted(re.escape(k) for k in lookup))  # why the sort?

    for line in sys.stdin:
        fname = basename(line)
        normalized = re.sub(pattern, lambda m: lookup.get(m.group(0)), fname)
        sys.stdout.write(join(dirname(line), normalized))
