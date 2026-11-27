import math
import torch
from data import train_data, val_data, vocab_size

counts = torch.ones(vocab_size, vocab_size)
for a, b in zip(train_data[:-1].tolist(), train_data[1:].tolist()):
    counts[a, b] += 1
probs = counts / counts.sum(1, keepdim=True)

nll = 0.0
pairs = list(zip(val_data[:-1].tolist(), val_data[1:].tolist()))
for a, b in pairs:
    nll -= math.log(probs[a, b].item())
loss = nll / len(pairs)
print(f"bigram val loss {loss:.4f}  perplexity {math.exp(loss):.2f}  bits/char {loss / math.log(2):.3f}")
print(f"uniform guess loss {math.log(vocab_size):.4f}")
