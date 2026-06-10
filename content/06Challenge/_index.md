---
title: "Final Challenge"
linkTitle: "Challenge"
weight: 80
---

Welcome to the final challenge! You have built and trained a generative language model from scratch.
Now your goal is to improve the **quality of the text it generates** — measured by how low a
validation loss (and perplexity) you can reach on held-out text — and then submit your model.

A better-trained model produces more coherent, corpus-like text. Your job is to tune the model and
training so it learns the language as well as possible without overfitting.

{{% notice tip %}}
Several variables were left open for tuning across the lab:
- `embed_dim`
- `num_heads`
- `ff_dim`
- `num_blocks`
- `block_size`
- `dropout_rate`
- `learning_rate`
- `steps` (training duration)
- `batch_size`
- Sampling at inference: `temperature`, `top_k`
{{% /notice %}}

## Tasks

- [ ] Review the lab material and the roadmap
- [ ] Tune your model architecture and training to lower validation loss / perplexity
- [ ] Generate a sample and confirm the text looks coherent and corpus-like
- [ ] Upload your final model to get validated

## Hold Out a Validation Set

So the challenge measures *generalization* (not memorization), evaluate on text the model didn't train
on. Reserve the last 10% of the corpus as validation.

```python
#@title Create a train/validation split of the corpus
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print("Train tokens:", len(train_data), "| Val tokens:", len(val_data))

def estimate_val_loss(model, val_data, iters=50):
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(val_data, batch_size, block_size)
        logits = model(xb, training=False)
        losses.append(float(loss_fn(yb, logits)))
    return float(np.mean(losses))
```

{{% notice info %}}
Remember to train on `train_data` (not the full `data`) when running the challenge, then call
`estimate_val_loss(model, val_data)` to see how well it generalizes. Lower is better.
{{% /notice %}}

## How to Submit Your Model

To submit your final model for validation, use the following snippet in your Colab notebook:

```python
#@title Submit Final Model for Validation
import requests, zipfile, os, json, time

PROVISIONER = "http://genai-workshop.ftntlab.tech/claim"

claim = requests.post(PROVISIONER, timeout=20).json()
submit_url = claim["submit_url"]
token = claim["submit_token"]

print("[*] Your submit URL:", submit_url)
print("[*] Your token:", token)

model.save("submission.keras")
!zip submission.zip submission.keras
print("[*] Model saved and zipped.")

time.sleep(5)

with open("submission.zip", "rb") as f:
    r = requests.post(
        submit_url,
        files={"file": f},
        headers={"X-Submit-Token": token},
        timeout=180
    )
print("[*] Your Result:\n")
if r.headers.get("content-type","").startswith("application/json"):
    print(json.dumps(r.json(), indent=2))
```

## Hints and Tips

{{% expand title="Hint 1" %}}
The fastest quality win is usually **more training**. Increase `steps` to 5000-8000 and watch the
validation loss keep dropping before it plateaus.
{{% /expand %}}

{{% expand title="Hint 2" %}}
A model that's too small can't capture the language. Try increasing `num_blocks` to 6 and `embed_dim`
to 128 — but watch training time, since cost grows with both.
{{% /expand %}}

{{% expand title="Hint 3" %}}
Widen the feed-forward network: `ff_dim` of 256 -> 512 gives each block more capacity to store
patterns.
{{% /expand %}}

{{% expand title="Hint 4" %}}
A larger `block_size` (context window), e.g. 192-256, lets the model use more history when predicting
the next token, which often improves coherence. This raises memory use noticeably.
{{% /expand %}}

{{% expand title="Hint 5" %}}
If validation loss starts *rising* while training loss keeps falling, you are **overfitting**.
Increase `dropout_rate` toward 0.2-0.3 or stop training earlier.
{{% /expand %}}

{{% expand title="Hint 6" %}}
`num_heads` must divide evenly into `embed_dim`. With `embed_dim=128`, try `num_heads=8`. More heads
let the model track more types of relationships in parallel.
{{% /expand %}}

{{% expand title="Hint 7" %}}
For the *generated sample* (not the loss), medium `temperature` (0.7-0.9) with `top_k=40` usually
reads best. This won't change your loss score, but it makes your demo output look much sharper.
{{% /expand %}}
