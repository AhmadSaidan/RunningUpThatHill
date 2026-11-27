"""Train a language model for a fixed time budget, then save it to out/.

Usage: python train.py [key=value ...]    e.g. python train.py n_layer=4 n_embd=256
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, build_ngram_tables, save_model
from tokenizer import BPETokenizer

# ---- config (override any of these from the command line with key=value) ----
time_budget = 300.0  # seconds of training
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
batch_size = 64
learning_rate = 6e-4
muon_lr = 0.02
vocab_size = 1024
seed = 1337
out_dir = "out"
for arg in sys.argv[1:]:
    key, val = arg.split("=")
    globals()[key] = type(globals()[key])(val)
# -----------------------------------------------------------------------------


def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate G -> U @ V.T via quintic Newton-Schulz iteration (Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.device.type == "cuda" else G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization for 2D hidden weights."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta, wd, ns_steps = group["lr"], group["momentum"], group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(beta).add_(g)
                u = zeroth_power_via_newtonschulz5(g.add(buf, alpha=beta), steps=ns_steps)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr * scale)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(out_dir, exist_ok=True)

with open("data/train.txt", "rb") as f:
    raw = f.read()
tok = BPETokenizer.train(raw, vocab_size=vocab_size)
tok.save(out_dir)
full_data = tok.encode(raw)
del raw
# Disjoint split: index second half so training hints are IID with held-out validation text.
split = len(full_data) // 2
data = full_data[:split]
tables = build_ngram_tables(full_data[split:])
del full_data


def get_batch():
    """A batch of random windows: y is x shifted one token to the left."""
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


raw_model = GPT(GPTConfig(tok.vocab_size, block_size, n_layer, n_head, n_embd)).to(device)
raw_model.load_ngram_tables(tables)
del tables
print(f"{sum(p.numel() for p in raw_model.parameters()):,} parameters on {device}")
muon_params = [p for n, p in raw_model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
adam_params = [p for n, p in raw_model.named_parameters() if not (n.startswith("blocks.") and p.ndim == 2)]
model = torch.compile(raw_model) if device == "cuda" else raw_model
optimizers = [
    Muon(muon_params, lr=muon_lr),
    torch.optim.AdamW(adam_params, lr=learning_rate),
]
use_bf16 = device in ("cuda", "mps")

step, t0 = 0, time.time()
while time.time() - t0 < time_budget:
    x, y = get_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    for opt in optimizers:
        opt.zero_grad()
    loss.backward()
    for opt in optimizers:
        opt.step()
    step += 1
    if step % 100 == 0:
        elapsed = time.time() - t0
        print(f"step {step} | loss {loss.item():.4f} | {elapsed:.0f}s | {step * batch_size * block_size / elapsed:,.0f} tok/s")

print(f"done: {step} steps, {step * batch_size * block_size:,} tokens in {time.time() - t0:.0f}s")
save_model(raw_model, out_dir)
