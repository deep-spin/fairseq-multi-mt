#!/usr/bin/env python

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    opt = parser.parse_args()

    state_dict = torch.load(opt.model)
    adapters = {k: v for k, v in state_dict["model"].items() if "adapter" in k}
    torch.save(adapters, opt.out)

if __name__ == "__main__":
    main()
