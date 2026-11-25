#!/bin/bash
# Score a submission on one L4 in a network-isolated SageMaker job, then print a Markdown report.
# Run from the repo root (main). Works from GitHub Actions or locally with AWS credentials.
# Usage: scoring/submit.sh <dir containing model.py tokenizer.py train.py> <name>
set -euo pipefail
SUB=$1
NAME=$2
: "${AWS_REGION:=eu-west-2}" "${BUCKET:=gpu-poor-golf-970450051890-eu-west-2}" "${MAX_BYTES:=16000000}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE=763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker
JOB=golf-$NAME-$(date -u +%m%d-%H%M%S)
S3=s3://$BUCKET/submissions/$JOB

aws s3 cp --recursive --quiet "$SUB" "$S3/submission/"
aws s3 cp --quiet eval.py "$S3/referee/eval.py"
aws s3 cp --quiet scoring/sandbox.sh "$S3/referee/sandbox.sh"
aws s3 cp --quiet scoring/sample.py "$S3/referee/sample.py"

channel() { echo "{\"ChannelName\":\"$1\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$2\",\"S3DataDistributionType\":\"FullyReplicated\"}}}"; }
CHANNELS="[$(channel submission "$S3/submission/"),$(channel referee "$S3/referee/"),$(channel train "s3://$BUCKET/data/train/"),$(channel val "s3://$BUCKET/data/val/")]"

# Only one L4 job can run at a time, so wait for a free slot.
until err=$(aws sagemaker create-training-job --training-job-name "$JOB" \
    --role-arn "arn:aws:iam::$ACCOUNT:role/GpuPoorGolfSageMaker" \
    --algorithm-specification "{\"TrainingImage\":\"$IMAGE\",\"TrainingInputMode\":\"File\",\"ContainerEntrypoint\":[\"bash\",\"/opt/ml/input/data/referee/sandbox.sh\",\"$MAX_BYTES\"]}" \
    --input-data-config "$CHANNELS" \
    --output-data-config "{\"S3OutputPath\":\"s3://$BUCKET/output/\"}" \
    --resource-config "{\"InstanceType\":\"ml.g6.xlarge\",\"InstanceCount\":1,\"VolumeSizeInGB\":30,\"KeepAlivePeriodInSeconds\":${KEEP_ALIVE:-0}}" \
    --stopping-condition '{"MaxRuntimeInSeconds":2400}' \
    --enable-network-isolation 2>&1 >/dev/null); do
    case "$err" in
        *ResourceLimitExceeded*) echo "GPU busy, retrying in 60s" >&2; sleep 60 ;;
        *) echo "$err" >&2; exit 1 ;;
    esac
done
echo "started $JOB" >&2
aws sagemaker wait training-job-completed-or-stopped --training-job-name "$JOB" || true  # a failed job still gets a report

TMP=$(mktemp -d)
aws s3 cp --quiet "s3://$BUCKET/output/$JOB/output/model.tar.gz" "$TMP/" && tar -xzf "$TMP/model.tar.gz" -C "$TMP" || true
python3 - "$TMP" "$JOB" <<'EOF'
import json, os, sys
d, job = sys.argv[1], sys.argv[2]
def read(f):
    p = os.path.join(d, f)
    return open(p).read() if os.path.exists(p) else ""
r = json.loads(read("results.json") or "{}")
r["job"] = job
json.dump(r, open("result.json", "w"))  # for scoring/publish.py
print(f"### GPU-Poor Golf score\n")
if r.get("bpb"):
    print(f"**val bpb: {float(r['bpb']):.4f}**\n")
    print("| size (bytes) | parameters | steps | tokens |\n|---|---|---|---|")
    print(f"| {int(r['size']):,} | {int(r['params'] or 0):,} | {r['steps']} | {int(r['tokens'] or 0):,} |")
    if r.get("sample"):
        print("\n**Rollout** (temperature 0.8):\n\n```text\n" + r["sample"].replace("`", "'") + "\n```")
else:
    stage = "training" if r.get("train_exit", 1) != 0 else "scoring"
    log = read("train.log" if stage == "training" else "eval.log")
    tail = "\n".join(log.strip().splitlines()[-25:])[-3000:]
    print(f"**Failed during {stage}.** Last lines of the log:\n\n```\n{tail}\n```")
print(f"\n<sub>job `{job}`</sub>")
EOF
