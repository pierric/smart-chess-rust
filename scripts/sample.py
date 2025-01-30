import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", required=True)
    parser.add_argument("-r", "--ratio", default=1.0, type=float)
    args = parser.parse_args()

    assert args.ratio > 0

    with open(args.list, "r") as fp:
        files = list(filter(None, map(lambda l: l.strip(), fp.readlines())))

    collection = []

    for f in files:
        comps = f.split(os.sep)
        runs_idx = comps.index("runs")
        assert runs_idx >= 0
        collection.append(
            {
                "run_id": int(comps[runs_idx + 1]),
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


if __name__ == "__main__":
    main()
