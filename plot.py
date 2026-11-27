import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = json.load(open("results.json"))["history"]
plt.figure(figsize=(6, 4))
plt.plot([r["iter"] for r in h], [r["train"] for r in h], label="train")
plt.plot([r["iter"] for r in h], [r["val"] for r in h], label="val")
plt.xlabel("iteration")
plt.ylabel("cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss.png", dpi=120)
