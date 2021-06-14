import sys
import argparse
import cyrtranslit as ct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language")
    parser.add_argument("to_script")
    opt = parser.parse_args()

    transliterator = ct.to_latin if opt.to_script == "latin" else ct.to_cyrillic

    for line in sys.stdin:
        sys.stdout.write(transliterator(line, opt.language))
