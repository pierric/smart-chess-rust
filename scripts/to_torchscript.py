#!/usr/bin/env python
import argparse
import os

import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", nargs="*", action="extend")
    args = parser.parse_args()

    print(args)

    for path in args.checkpoint:
        stamm, _ = os.path.splitext(path)
        nn.export(path, f"{stamm}.pt")


if __name__ == "__main__":
    main()
