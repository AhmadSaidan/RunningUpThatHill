#!/bin/bash
# Runs inside a network-isolated SageMaker job on one L4.
# 1. Trains the submission as an unprivileged user who cannot read the validation text.
# 2. Scores it with the trusted eval.py from main.
# Usage: sandbox.sh <max_bytes>
set -uo pipefail
MAX_BYTES=$1
IN=/opt/ml/input/data  # channels: submission/ (from the PR), referee/ (from main), train/, val/
OUT=/opt/ml/model      # uploaded to S3 when the job ends

# --- 1. train ---
useradd --create-home student
chmod 700 $IN/val $IN/referee  # root only
if runuser -u student -- cat $IN/val/val.txt > /dev/null 2>&1; then
    echo "sandbox error: the student user can read the validation text" | tee $OUT/train.log
    exit 1
fi
mkdir -p /work/data
cp $IN/submission/*.py /work/
cp $IN/train/train.txt /work/data/
chown -R student /work
cd /work
timeout 900 runuser -u student -- python train.py > $OUT/train.log 2>&1
TRAIN_EXIT=$?

# Nothing the student started may survive into scoring.
LEFT=$(pgrep -u student | wc -l)
pkill -9 -u student; sleep 1
if pgrep -u student > /dev/null; then
    echo "sandbox error: could not stop the student's processes" | tee -a $OUT/train.log
    exit 1
fi
echo "sandbox: stopped $LEFT leftover student process(es)" >> $OUT/train.log
if [ -d /work/out ] && find /work/out -type l | grep -q .; then
    echo "sandbox error: out/ may not contain symlinks" | tee -a $OUT/train.log
    TRAIN_EXIT=1
fi

# --- 2. score ---
EVAL_EXIT=1
if [ $TRAIN_EXIT -eq 0 ]; then
    rm -rf /score  # in case the student pre-created it
    mkdir -m 700 /score && mkdir /score/data
    cp $IN/referee/eval.py /score/
    cp $IN/submission/model.py $IN/submission/tokenizer.py /score/
    cp -r /work/out /score/out
    cp $IN/val/val.txt /score/data/
    cd /score
    timeout 900 python eval.py "$MAX_BYTES" > $OUT/eval.log 2>&1
    EVAL_EXIT=$?
    if [ $EVAL_EXIT -eq 0 ]; then  # a rollout for the leaderboard; failure here doesn't affect the score
        cp $IN/referee/sample.py /score/
        timeout 120 python sample.py > $OUT/sample.txt 2> $OUT/sample.log || true
    fi
fi

python - "$TRAIN_EXIT" "$EVAL_EXIT" > $OUT/results.json <<'EOF'
import json, os, re, sys
def read(f):
    return open(f).read() if os.path.exists(f) else ""
def find(pattern, text):  # last match: eval.py prints its genuine results at the end
    m = re.findall(pattern, text)
    return m[-1].replace(",", "") if m else None
train, ev = read("/opt/ml/model/train.log"), read("/opt/ml/model/eval.log")
print(json.dumps({
    "train_exit": int(sys.argv[1]), "eval_exit": int(sys.argv[2]),
    "bpb": find(r"val bpb: ([\d.]+)", ev), "size": find(r"size: ([\d,]+)", ev),
    "params": find(r"([\d,]+) parameters", train), "steps": find(r"done: (\d+) steps", train),
    "tokens": find(r"done: \d+ steps, ([\d,]+) tokens", train),
    "bytes_per_token": find(r"bytes per token: ([\d.]+)", ev),
    "sample": read("/opt/ml/model/sample.txt").strip()[:600] or None,
    # training curve from train.py's "step N | loss L | Ts" lines: [seconds, step, loss in nats/token]
    "curve": [[int(t), int(s), float(l)] for s, l, t in re.findall(r"step (\d+) \| loss ([\d.]+) \| (\d+)s", train)],
}))
EOF
cat $OUT/results.json
