#!/usr/bin/env python

import argparse
import torch

"""
Our baseline model (flores_big_task2_vocab, which is basically the same as the
shared task's official baseline) was not trained with adapters. We would like
to "plug in" the adapters we used in our finetuning experiments. In order to
do so, we would like to overwrite the model config values relating to adapters
so that the model is built with them.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_model")
    parser.add_argument("adapter_model")
    parser.add_argument("out")
    opt = parser.parse_args()

    baseline = torch.load(opt.baseline_model)
    adapter = torch.load(opt.adapter_model)
    baseline["cfg"]["model"].adapter_enc_type = adapter["cfg"]["model"].adapter_enc_type
    baseline["cfg"]["model"].adapter_dec_type = adapter["cfg"]["model"].adapter_dec_type
    baseline["cfg"]["model"].adapter_enc_dim = adapter["cfg"]["model"].adapter_enc_dim
    baseline["cfg"]["model"].adapter_dec_dim = adapter["cfg"]["model"].adapter_dec_dim
    baseline["cfg"]["model"].adapter_keys = adapter["cfg"]["model"].adapter_keys
    torch.save(baseline, opt.out)


if __name__ == "__main__":
    main()
