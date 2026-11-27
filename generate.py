import sys
import torch
from data import vocab_size, encode, decode
from model import GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=256, dropout=0.1).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "\n"
ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
out = model.generate(ctx, 400, temperature=0.8, top_k=40)
print(decode(out[0].tolist()))
