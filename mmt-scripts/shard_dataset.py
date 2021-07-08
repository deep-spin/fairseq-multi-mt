import argparse
from os import makedirs
from os.path import basename, join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")  # not a parallel corpus, just a single one
    parser.add_argument("out_dir")
    parser.add_argument("num_shards", type=int)
    opt = parser.parse_args()

    for i in range(opt.num_shards):
        makedirs(join(opt.out_dir, "shard-{}".format(i)), exist_ok=True)
    # we will be writing to num_shards files
    corpus_name = basename(opt.corpus)
    shards = [open(join(opt.out_dir, "shard-{}".format(i), corpus_name), "w")
              for i in range(opt.num_shards)]
    with open(opt.corpus) as f:
        for i, line in enumerate(f):
            cur_shard = i % opt.num_shards  # round robin
            cur_file = shards[cur_shard]
            cur_file.write(line)


if __name__ == "__main__":
    main()
