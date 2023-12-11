export ROOT=`cargo metadata --format-version 1 | jq -r ".workspace_root"`
export PYTHONPATH="$ROOT/py"
ARGS=$@
N=10

for run_idx in `seq $N`; do
    echo "Run index" $run_idx
    cargo r --release -- -t "trace${run_idx}.json" $ARGS
done

for run_idx in `seq $N`; do
    echo -n "$run_idx: "
    cat "trace${run_idx}.json" | jq -r ".outcome.winner"
done