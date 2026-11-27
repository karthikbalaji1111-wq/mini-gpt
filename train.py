import json
import math
import time
import torch
from data import get_batch, train_data, val_data, vocab_size, decode
from model import GPT

batch_size = 64
block_size = 128
max_iters = 3000
eval_every = 250
eval_batches = 40
lr_max = 1e-3
lr_min = 1e-4
warmup = 100

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(vocab_size, block_size=block_size, n_layer=4, n_head=4, n_embd=256, dropout=0.1).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params / 1e6:.2f}M parameters on {device}")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)


def get_lr(it):
    if it < warmup:
        return lr_max * (it + 1) / warmup
    progress = (it - warmup) / (max_iters - warmup)
    return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for name, source in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            x, y = get_batch(source, batch_size, block_size, device)
            losses[k] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


history = []
start = time.time()
for it in range(max_iters + 1):
    if it % eval_every == 0:
        losses = estimate_loss()
        history.append({"iter": it, "train": losses["train"], "val": losses["val"]})
        print(f"iter {it}  train {losses['train']:.4f}  val {losses['val']:.4f}  {time.time() - start:.0f}s")
    if it == max_iters:
        break
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it)
    x, y = get_batch(train_data, batch_size, block_size, device)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

final = history[-1]
results = {
    "params": n_params,
    "device": device,
    "minutes": round((time.time() - start) / 60, 1),
    "final_train_loss": final["train"],
    "final_val_loss": final["val"],
    "val_perplexity": math.exp(final["val"]),
    "val_bits_per_char": final["val"] / math.log(2),
    "history": history,
}
json.dump(results, open("results.json", "w"), indent=2)
torch.save(model.state_dict(), "model.pt")

model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = decode(model.generate(ctx, 500, temperature=0.8, top_k=40)[0].tolist())
open("sample.txt", "w").write(sample)
print(sample)
