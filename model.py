"""A vanilla GPT-2 style transformer. Adapted from nanoGPT (github.com/karpathy/nanoGPT, MIT)."""
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-scale exact-match n-gram tables: (order_k, log2_slots).
# 524,288 int32 slots = 2.10 MB -> model + tables ~15.55 MB (< 16 MB).
# 31-bit packed entry: tag[0:16] | next_byte[16:24] | conf[24:31] (always non-negative int32).
NGRAM_SPECS = ((4, 17), (6, 18), (8, 17))
HASH_MULT = 16777619
HASH_MASK = (1 << 36) - 1


def _hash_ngrams_np(tokens: np.ndarray, k: int) -> np.ndarray:
    n = len(tokens) - k
    h = np.zeros(n, dtype=np.int64)
    for j in range(k):
        h = (h * HASH_MULT + tokens[j : j + n].astype(np.int64)) & HASH_MASK
        h = h ^ (h >> 17)
    return h


def build_ngram_tables(tokens: np.ndarray, max_positions: int = 25_000_000):
    """Build heavy-hitter modal next-byte + confidence hash tables in pure NumPy."""
    if len(tokens) > max_positions + 16:
        tokens = tokens[: max_positions + 16]
    tables = []
    for k, bits in NGRAM_SPECS:
        slots = 1 << bits
        h = _hash_ngrams_np(tokens, k)
        slot = h & (slots - 1)
        tag = (h >> bits) & 0xFFFF
        tag = np.where(tag == 0, 1, tag)
        nxt = tokens[k : k + len(h)].astype(np.int64) & 0xFF

        packed = (slot << 24) | (tag << 8) | nxt
        packed.sort()
        diff = np.empty(len(packed), dtype=np.bool_)
        diff[0] = True
        diff[1:] = packed[1:] != packed[:-1]
        starts = np.flatnonzero(diff)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = len(packed)
        counts = ends - starts
        uniq = packed[starts]

        st_key = uniq >> 8
        st_diff = np.empty(len(st_key), dtype=np.bool_)
        st_diff[0] = True
        st_diff[1:] = st_key[1:] != st_key[:-1]
        st_starts = np.flatnonzero(st_diff)
        totals = np.add.reduceat(counts, st_starts)

        order = np.argsort(counts, kind="mergesort")
        gid = np.cumsum(st_diff) - 1
        best_idx = np.empty(len(st_starts), dtype=np.int64)
        best_idx[gid[order]] = order

        best_slot = uniq[best_idx] >> 24
        best_tag = (uniq[best_idx] >> 8) & 0xFFFF
        best_nxt = uniq[best_idx] & 0xFF
        best_cnt = counts[best_idx]
        purity = best_cnt.astype(np.float32) / np.maximum(totals.astype(np.float32), 1.0)
        support = 1.0 - 1.0 / (totals.astype(np.float32) + 1.0)
        best_conf = np.clip(np.round(127.0 * purity * support), 1, 127).astype(np.int64)

        packed_entry = (best_tag | (best_nxt << 16) | (best_conf << 24)).astype(np.int32)
        slot_order = np.argsort(totals, kind="mergesort")
        tbl = np.zeros(slots, dtype=np.int32)
        tbl[best_slot[slot_order]] = packed_entry[slot_order]
        tables.append(tbl)
    return tables


def _hash_ngrams_torch(idx: torch.Tensor, k: int) -> torch.Tensor:
    """Strictly causal rolling hash matching _hash_ngrams_np (no int64 overflow)."""
    B, T = idx.shape
    h = torch.zeros((B, T), dtype=torch.int64, device=idx.device)
    for j in range(k):
        shift_left = k - 1 - j
        tok = F.pad(idx[:, :-shift_left], (shift_left, 0)) if shift_left > 0 else idx
        h = (h * HASH_MULT + tok) & HASH_MASK
        h = h ^ (h >> 17)
    return h


@dataclass
class GPTConfig:
    vocab_size: int = 257
    block_size: int = 256  # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))  # no peeking at the future
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        for lvl, (_, bits) in enumerate(NGRAM_SPECS):
            self.register_buffer(f"ng_tbl_{lvl}", torch.zeros(1 << bits, dtype=torch.int32))
        self.ng_emb_gate = nn.Parameter(torch.zeros(len(NGRAM_SPECS)))
        self.ng_logit_gate = nn.Parameter(torch.full((len(NGRAM_SPECS),), 0.5))

    def load_ngram_tables(self, tables):
        for lvl, tbl in enumerate(tables):
            getattr(self, f"ng_tbl_{lvl}").copy_(torch.from_numpy(tbl))

    def _lookup_hints(self, idx):
        B, T = idx.shape
        best_nxt = torch.zeros((B, T), dtype=torch.long, device=idx.device)
        best_conf = torch.zeros((B, T), dtype=torch.float32, device=idx.device)
        best_lvl = torch.full((B, T), -1, dtype=torch.long, device=idx.device)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        for lvl, (k, bits) in enumerate(NGRAM_SPECS):
            if T < k:
                continue
            h = _hash_ngrams_torch(idx, k)
            slot = h & ((1 << bits) - 1)
            tag = (h >> bits) & 0xFFFF
            tag = torch.where(tag == 0, torch.ones_like(tag), tag)
            entry = getattr(self, f"ng_tbl_{lvl}")[slot].to(torch.int64)
            hit = ((entry & 0xFFFF) == tag) & (pos >= k - 1)
            nxt = (entry >> 16) & 0xFF
            conf = ((entry >> 24) & 0x7F).to(torch.float32) / 127.0
            best_nxt = torch.where(hit, nxt, best_nxt)
            best_conf = torch.where(hit, conf, best_conf)
            best_lvl = torch.where(hit, torch.full_like(best_lvl, lvl), best_lvl)
        return best_nxt, best_conf, best_lvl

    def forward(self, idx):
        """idx: (B, T) token ids -> logits: (B, T, vocab_size)"""
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        hint_nxt, hint_conf, hint_lvl = self._lookup_hints(idx)
        has_hint = hint_lvl >= 0
        safe_lvl = hint_lvl.clamp_min(0)
        emb_w = torch.where(has_hint, self.ng_emb_gate[safe_lvl] * hint_conf, torch.zeros_like(hint_conf))
        x = x + emb_w.to(x.dtype).unsqueeze(-1) * self.tok_emb(hint_nxt)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        logit_w = torch.where(has_hint, self.ng_logit_gate[safe_lvl] * hint_conf, torch.zeros_like(hint_conf))
        return logits.scatter_add(2, hint_nxt.unsqueeze(-1), logit_w.to(logits.dtype).unsqueeze(-1))


def save_model(model, out_dir):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, os.path.join(out_dir, "model.pt"))


def load_model(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
