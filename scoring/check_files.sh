#!/bin/bash
# Fails if PR $PR changes anything other than the three files students may edit.
set -euo pipefail
gh pr diff "$PR" --repo "$GITHUB_REPOSITORY" --name-only > changed.txt
cat changed.txt
if grep -vxE 'model\.py|tokenizer\.py|train\.py' changed.txt; then
    echo "::error::Only model.py, tokenizer.py and train.py may be changed."
    exit 1
fi
