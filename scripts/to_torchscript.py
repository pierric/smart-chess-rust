import argparse

import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=9)
    parser.add_argument("-e", "--end", type=int, default=99)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    for n in range(args.start, args.end+1, args.step):
        nn.export(f"epoch-{n}.ckpt", f"epoch-{n}.pt")


if __name__ == "__main__":
    main()
