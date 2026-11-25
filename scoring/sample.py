"""Writes a short continuation of a fixed prompt, for the leaderboard. Run after eval.py, in the same folder.
Uses only the interface eval.py relies on: load_model(out_dir), load_tokenizer(out_dir), model(x) -> logits.

Usage: python sample.py > sample.txt
"""
import torch

from model import load_model
from tokenizer import load_tokenizer

PROMPT = "Deep Learning is all about"
MAX_BYTES = 300  # of generated text
TEMPERATURE = 0.8

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = load_tokenizer("out")
model = load_model("out").to(device).eval()
T = model.block_size

prompt_ids = [int(i) for i in tok.encode(PROMPT.encode())]
ids = [tok.bos] + prompt_ids
with torch.no_grad():
    for _ in range(4 * MAX_BYTES):  # enough tokens for any sensible tokenizer
        x = torch.tensor([ids[-T:]], device=device)
        logits = model(x)[0, -1, : tok.vocab_size].float()
        logits[tok.bos] = -float("inf")  # no new documents
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        ids.append(int(torch.multinomial(probs, 1)))
        if len(tok.decode(ids[1 + len(prompt_ids) :])) >= MAX_BYTES:
            break

text = tok.decode(ids[1:]).decode("utf-8", errors="replace")
print(text[: len(PROMPT) + MAX_BYTES])
