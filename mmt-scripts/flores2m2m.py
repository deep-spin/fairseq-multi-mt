import sys
import re
from os.path import basename, dirname, join

# flores codes seem to be ISO 639-3. M2M codes seem to be ISO 639-1. In lieu
# of a principled solution, I have written a regular expression that works only
# for the set of languages included in small tasks 1 and 2.

flores = """
eng
est
hrv
hun
mkd
srp
ind
jav
msa
tam
tgl
""".split()

m2m = """
en
et
hr
hu
mk
sr
id
jv
ms
ta
tl
""".split()

lookup = dict(zip(flores, m2m))

pattern = '|'.join(sorted(re.escape(k) for k in lookup))  # why the sort?

for line in sys.stdin:
    fname = basename(line)
    normalized = re.sub(pattern, lambda m: lookup.get(m.group(0)), fname)
    sys.stdout.write(join(dirname(line), normalized))
