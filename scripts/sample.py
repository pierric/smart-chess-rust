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
            }
        )

    df = pd.DataFrame(collection)

    latest_run_id = df.run_id.max()
    older_runs = df[df.run_id < latest_run_id]
    latest_runs = df[df.run_id == latest_run_id]

    assert len(older_runs) >= len(
        latest_runs
    ), "to sample the runs, there need more older runs."

    selection = pd.concat(
        (older_runs.sample(n=int(args.ratio * len(latest_runs))), latest_runs)
    )

    for f in selection.file:
        print(f)


def split(args):
    with open(args.list, "r") as fp:
        files = list(filter(None, map(lambda l: l.strip(), fp.readlines())))

    def _get_result(f):
        o = json.load(open(f))
        return (o.get("outcome") or {}).get("winner", None) or "draw"

    df = pd.DataFrame([{"filename": f, "result": _get_result(f)} for f in tqdm(files)])
    groups = df.groupby(by="result")
    nmax = groups.count().loc[["White", "Black"]].max().item()

    w = groups.get_group("White")
    b = groups.get_group("Black")

    w = w.sample(frac=1).reset_index(drop=True)
    b = b.sample(frac=1).reset_index(drop=True)

    if args.sample_draw == "sample":
        d = groups.get_group("draw").sample(n=nmax)
    elif args.sample_draw == "keep-all":
        d = groups.get_group("draw")
        d = d.sample(frac=1).reset_index(drop=True)
    else:
        assert "bad value for --sample-draw"

    def _split(d):
        nval = int(len(d) * args.ratio)
        assert nval > 0
        return d.iloc[:nval], d.iloc[nval:]

    val_d, train_d = _split(d)
    val_w, train_w = _split(w)
    val_b, train_b = _split(b)

    val = pd.concat([val_d, val_w, val_b])
    train = pd.concat([train_d, train_w, train_b])

    print(f"val split: {len(val)} samples")
    print(f"train split: {len(train)} samples")

    assert set(val.filename).isdisjoint(set(train.filename))

    open(args.train, "w").writelines("\n".join(train.filename))
    open(args.val, "w").writelines("\n".join(val.filename))


if __name__ == "__main__":
    main()
