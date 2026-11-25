"""Download the data, a fixed slice of FineWeb (one 2 GB shard). Run once.

  data/val.txt   - the first ~2 MB of documents (held out, only eval.py reads this)
  data/train.txt - the next documents, ~500 MB by default

Usage: python prepare.py [train_mb]
"""
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

train_mb = float(sys.argv[1]) if len(sys.argv) > 1 else 500
path = hf_hub_download("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet", repo_type="dataset")
docs = (doc for batch in pq.ParquetFile(path).iter_batches(columns=["text"]) for doc in batch.column(0).to_pylist())

os.makedirs("data", exist_ok=True)
for name, mb in [("val", 2), ("train", train_mb)]:
    size = 0
    with open(f"data/{name}.txt", "wb") as f:
        while size < mb * 1e6:
            doc = next(docs).encode() + b"\n\n"
            f.write(doc)
            size += len(doc)
    print(f"data/{name}.txt: {size:,} bytes")
