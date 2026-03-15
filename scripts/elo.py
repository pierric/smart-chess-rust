import argparse
import math


def main():
    parser = argparse.ArgumentParser("EloCalc")
    parser.add_argument("result", type=str)

    args = parser.parse_args()
    items = args.result.split("/")

    if len(items) != 3:
        raise ValueError("Input should be in the form Total/WhiteWin/BlackWin")

    total, win, lost = [int(v.strip()) for v in items]

    score = win + (total - win - lost) / 2
    s = score / total
    d = 400 * math.log(s / (1 - s), 10)

    print(f"ELO: {d:+0.2f}")


if __name__ == "__main__":
    main()
