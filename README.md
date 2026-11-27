mini-gpt

This is a small GPT that I wrote from scratch in PyTorch and trained on tiny Shakespeare, which is about 1.1 million characters. I wrote the attention myself instead of using nn.MultiheadAttention because I wanted to make sure I actually understood it.

model.py has the causal self-attention, the transformer block and the GPT itself. test_model.py checks three things: my attention gives the same output as PyTorch's scaled_dot_product_attention, changing future tokens never changes the earlier predictions, and the loss of an untrained model is close to ln(vocab size). train.py is a plain training loop with AdamW, warmup, cosine decay and gradient clipping. baseline.py is a simple bigram model to compare against.

It is character level with 65 characters. I used a 90/10 train and validation split, 4 layers, 4 heads, 256 dimensions, a context of 128 and dropout of 0.1. That comes to 3.2 million parameters. Training for 3000 iterations at batch size 64 took about 9 minutes on my MacBook Air.

The final validation loss was 1.48, which is a perplexity of 4.4. The bigram baseline got 2.48, a perplexity of 12, and guessing at random would give 4.17. So it learned something real, but the generated text is still nonsense. It gets the play format and the speaker names right and the grammar is roughly there, but the sentences don't mean anything.

Validation loss was lowest around iteration 2500 and then started going up a little while the training loss kept dropping, so it was starting to overfit. I kept the final checkpoint instead of the best one, which was lazy. More dropout or early stopping is what I would try next. The loss curve is in loss.png and a sample is in sample.txt.

To run it, download the tiny Shakespeare file from Karpathy's char-rnn repo and save it as input.txt, install torch, numpy and matplotlib, then run test_model.py, baseline.py, train.py and generate.py in that order. generate.py takes a prompt like "ROMEO:" as an argument.
