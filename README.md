# mini-gpt

A small GPT written from scratch in PyTorch and trained on tiny Shakespeare (about 1.1M characters). I wrote the attention myself instead of using `nn.MultiheadAttention`, mostly to make sure I actually understood it.

- `model.py` has causal self-attention, the transformer block (pre-LayerNorm, GELU MLP, residuals) and the GPT with learned position embeddings and tied input/output embeddings.
- `test_model.py` checks that my attention matches `F.scaled_dot_product_attention`, that changing future tokens never changes earlier predictions, and that the untrained loss is close to ln(vocab).
- `train.py` is a plain training loop: AdamW, linear warmup then cosine decay, gradient clipping.
- `baseline.py` is a bigram model to compare against.

## Setup that I used

Character level tokenizer (65 characters), 90/10 train/val split, 4 layers, 4 heads, 256 dims, context of 128, dropout 0.1. 3.2M parameters, 3000 iterations at batch size 64, about 9 minutes on an M2 MacBook Air.


![loss curve](loss.png)


