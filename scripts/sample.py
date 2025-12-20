import os
import json
import argparse
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", required=True)
    subparser = parser.add_subparsers(dest="command", required=True)
    pmix = subparser.add_parser("mix")
    pmix.add_argument("-r", "--ratio", default=1.0, type=float)
    psplit = subparser.add_parser("split")
    psplit.add_argument("-r", "--ratio", default=0.1, type=float)
    psplit.add_argument("-v", "--val", required=True)
    psplit.add_argument("-t", "--train", required=True)
    psplit.add_argument("--factor-draw", type=float, default=1.0)
    psplit.add_argument(
        "--sample-draw", choices=["sample", "keep-all"], default="sample"
    )

    args = parser.parse_args()

    if args.command == "mix":
        mix(args)

    else:
        split(args)


def mix(args):
    assert args.ratio > 0

    with open(args.list, "r") as fp:
        files = list(filter(None, map(lambda l: l.strip(), fp.readlines())))

    collection = []

    for f in files:
        comps = f.split(os.sep)
        external = comps[0] == ".."

        index = 0
        while index < len(comps) and (
            comps[index] in [".", ".."] or comps[index].startswith("L")
        ):
            index += 1
        comps = comps[index:]

        try:
            runs_idx = comps.index("runs")
        except ValueError:
            run_id = int(comps[0])
        else:
            run_id = int(comps[runs_idx + 1])

        collection.append(
            {
                "run_id": run_id,
                "file": f,
                "external": external,
            }
        )

    df = pd.DataFrame(collection)
    edf = df[df.external]
    cdf = df[~df.external]

    latest_run_id = cdf.run_id.max()
    older_runs = pd.concat([edf, cdf[cdf.run_id < latest_run_id]])
    latest_runs = cdf[cdf.run_id == latest_run_id]

    assert len(older_runs) >= args.ratio * len(
        latest_runs
    ), f"to sample the runs, there need more older runs. num old: {len(older_runs)}"

    selection = pd.concat(
        (older_runs.sample(n=int(args.ratio * len(latest_runs))), latest_runs)
    )

    for f in selection.file:
        print(f)


def split(args):
    with open(args.list, "r") as fp:
        files = list(filter(None, map(lambda l: l.strip(), fp.readlines())))

    def _get_result(f):
        try:
            o = json.load(open(f))
        except:
            print(f"Failed to load file: {f}")
            raise
        o = o.get("outcome")

        if o is None:
            return "Unfinished"

        if w := o.get("winner"):
            return w

        return o.get("termination")

    df = pd.DataFrame([{"filename": f, "result": _get_result(f)} for f in tqdm(files)])
    df["result_"] = df["result"].map(
        {
            "White": "White",
            "Black": "Black",
            "InsufficientMaterial": "draw",
            "Stalemate": "draw",
            "Unfinished": "draw",
        }
    )
    df.dropna(inplace=True)
    groups = df.groupby(by="result_")

    group_keys = groups.groups.keys()

    print("Counting")
    print(groups.count().to_string(header=False))
    print("--------")

    if "White" in group_keys and "Black" in group_keys:
        nmax = groups.count().result.loc[["White", "Black"]].max().item()

        w = groups.get_group("White")
        b = groups.get_group("Black")

        w = w.sample(frac=1).reset_index(drop=True)
        b = b.sample(frac=1).reset_index(drop=True)

        groups_col = [w, b]

        if "draw" in group_keys:
            if args.sample_draw == "sample":
                if args.factor_draw is not None:
                    total_draw = len(groups.get_group("draw"))
                    nmax = min(total_draw, int(args.factor_draw * nmax))
                d = groups.get_group("draw").sample(n=nmax)
            elif args.sample_draw == "keep-all":
                d = groups.get_group("draw")
                d = d.sample(frac=1).reset_index(drop=True)
            else:
                assert "bad value for --sample-draw"

            groups_col.append(d)

    else:
        assert args.sample_draw == "keep-all"
        groups_col = [
            groups.get_group(g).sample(frac=1).reset_index(drop=True)
            for g in group_keys
        ]

    print("Selected")
    for c, g in zip("wbd", groups_col):
        print(c, ":", len(g))
    print("--------")

    def _split(d):
        nval = int(len(d) * args.ratio)
        assert nval > 0
        return d.iloc[:nval], d.iloc[nval:]

    val, train = zip(*[_split(g) for g in groups_col])

    val = pd.concat(val).sample(frac=1).reset_index(drop=True)
    train = pd.concat(train).sample(frac=1).reset_index(drop=True)

    print(f"val split: {len(val)} samples")
    print(f"train split: {len(train)} samples")

    assert set(val.filename).isdisjoint(set(train.filename))

    open(args.train, "w").writelines("\n".join(train.filename))
    open(args.val, "w").writelines("\n".join(val.filename))


if __name__ == "__main__":
    main()
