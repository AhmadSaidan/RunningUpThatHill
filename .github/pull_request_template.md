One change: <!-- one line, shown on the leaderboard, e.g. "learning rate 6e-4 -> 1e-3" -->

## Motivation

<!-- Why should this help in a 5-minute budget? Point to the lecture idea or estimate behind it
     (FLOPs, memory, tokens seen, number formats, ...). A few sentences is plenty. -->

## Local results

<!-- Run the baseline and your change on the same machine with the same settings, so the comparison is fair.
     Train loss: the last "step ... | loss ..." line of train.py. Val bpb: the "val bpb:" line of eval.py. -->

Machine (e.g. M2 laptop, Colab T4): 
Training time used (`time_budget`): 

| | Steps | Final train loss | Val bpb |
|---|---|---|---|
| Baseline (current `main`) | | | |
| My change | | | |

## Checklist

- [ ] One change only
- [ ] Only `model.py`, `tokenizer.py` and/or `train.py` changed
- [ ] `eval.py` runs and the size is under 16 MB
- [ ] My branch is up to date with the latest `main`
