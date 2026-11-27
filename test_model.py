import torch
import torch.nn.functional as F
from model import Attention, GPT

torch.manual_seed(0)


def test_attention_matches_torch():
    attn = Attention(n_embd=64, n_head=4, block_size=32, dropout=0.0).eval()
    x = torch.randn(2, 20, 64)
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=2)
    q, k, v = [t.view(B, T, 4, C // 4).transpose(1, 2) for t in (q, k, v)]
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = attn.proj(expected.transpose(1, 2).contiguous().view(B, T, C))
    assert torch.allclose(attn(x), expected, atol=1e-5)


def test_no_peeking_at_future_tokens():
    model = GPT(vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0).eval()
    a = torch.randint(0, 30, (1, 16))
    b = a.clone()
    b[0, 10:] = torch.randint(0, 30, (6,))
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 12:], lb[0, 12:], atol=1e-5)


def test_initial_loss_is_near_uniform():
    import math
    model = GPT(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    x = torch.randint(0, 65, (8, 16))
    y = torch.randint(0, 65, (8, 16))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(65)) < 0.3


if __name__ == "__main__":
    test_attention_matches_torch()
    test_no_peeking_at_future_tokens()
    test_initial_loss_is_near_uniform()
    print("all passed")
