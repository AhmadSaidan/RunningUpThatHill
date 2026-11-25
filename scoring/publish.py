"""Update the leaderboard site (the gh-pages branch, checked out at <site>).

Add a scored run:   python scoring/publish.py <site> add result.json --pr 12 --sha abc1234 --author someone --title "..." --change "..."
Mark a run merged:  python scoring/publish.py <site> merged --pr 12 --sha abc1234
"""
import argparse
import datetime
import json
import math
import os

p = argparse.ArgumentParser()
p.add_argument("site")
p.add_argument("action", choices=["add", "merged"])
p.add_argument("result", nargs="?")
p.add_argument("--pr", type=int, required=True)
p.add_argument("--sha", required=True)
p.add_argument("--author", default="")
p.add_argument("--title", default="")
p.add_argument("--change", default="")
args = p.parse_args()

runs_path = os.path.join(args.site, "runs.json")
runs = json.load(open(runs_path)) if os.path.exists(runs_path) else []


def best_record(runs):
    merged = [r for r in runs if r["merged"] and r["bpb"] is not None]
    return min(merged, key=lambda r: r["bpb"]) if merged else None


if args.action == "merged":
    matches = [r for r in runs if r["pr"] == args.pr and r["sha"].startswith(args.sha[:7])]
    if not matches:
        raise SystemExit(f"no scored run for PR #{args.pr} at {args.sha[:7]}: rescore before merging")
    matches[-1]["merged"] = True
else:
    r = json.load(open(args.result))
    bpb = float(r["bpb"]) if r.get("bpb") else None
    bytes_per_token = float(r.get("bytes_per_token") or 1)
    entry = {
        "job": r["job"],
        "pr": args.pr,
        "sha": args.sha[:7],
        "author": args.author,
        "title": args.title,
        "change": args.change or args.title,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "merged": False,
        "bpb": bpb,
        "size": int(r["size"]) if r.get("size") else None,
        "params": int(r["params"]) if r.get("params") else None,
        "steps": int(r["steps"]) if r.get("steps") else None,
        "tokens": int(r["tokens"]) if r.get("tokens") else None,
        "sample": r.get("sample"),
        # training loss converted to bits per byte, so tokenizers are comparable: [seconds, bpb]
        "curve": [[t, round(loss / math.log(2) / bytes_per_token, 4)] for t, _, loss in r.get("curve", [])],
    }
    record = best_record(runs)
    runs = [x for x in runs if x["job"] != entry["job"]] + [entry]

    # Loss-curve image for the PR comment: this run against the current record.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    if record and record["curve"]:
        t, y = zip(*record["curve"])
        ax.plot(t, y, color="0.6", ls="--", label=(f"record: #{record['pr']} @{record['author']}" if record["pr"] else f"record: {record.get('label') or 'baseline'} baseline") + f" ({record['bpb']:.4f})")
    if entry["curve"]:
        t, y = zip(*entry["curve"])
        score = f"{bpb:.4f}" if bpb is not None else "failed"
        ax.plot(t, y, color="#2563eb", label=f"this PR: #{args.pr} @{args.author} ({score})")
    ax.set(xlabel="training time (s)", ylabel="train loss (bits per byte)", title=entry["change"][:80])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.join(args.site, "plots"), exist_ok=True)
    fig.savefig(os.path.join(args.site, "plots", f"{entry['job']}.png"))

json.dump(runs, open(runs_path, "w"), indent=1)
print(f"{len(runs)} runs in {runs_path}")
