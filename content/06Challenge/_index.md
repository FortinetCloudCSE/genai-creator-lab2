---
title: "Final Challenge"
linkTitle: "Challenge"
weight: 80
---

Welcome to the final challenge! You have built and pre-trained a generative language model from
scratch. Now your goal is to make it **write better** — and to prove it with a number rather than a
feeling.

**Approx. time:** 20 minutes

## The Objective

Get the **validation loss as low as you can** without overfitting, then generate a sample that reads
convincingly like the corpus.

Validation loss is the right score here because it measures the model on text it has never been
trained on. A model that merely memorized Tiny Shakespeare would show a low *training* loss and a
high *validation* loss.

```python
#@title Score your model
val = estimate_val_loss(iters=50)
print(f"Validation loss : {val:.4f}")
print(f"Perplexity      : {np.exp(val):.2f}")
```

## Target Tiers

The default settings from Phase 3 land at a validation loss of about **1.70**. Beat that.

| Tier | Validation loss | Perplexity | How it is reached |
| ---- | --------------- | ---------- | ----------------- |
| Untrained | ~4.17 | ~65 | — |
| Baseline (Phase 3 defaults) | ~1.70 | ~5.5 | 3000 steps, `embed_dim=64` |
| 🥉 Bronze | < 1.65 | < 5.2 | More training steps alone will get you here |
| 🥈 Silver | < 1.58 | < 4.9 | A bigger model — start with `embed_dim` and `ff_dim` |
| 🥇 Gold | < 1.55 | < 4.7 | A bigger model **and** a longer run |

{{% notice info %}}
These tiers are measured, not invented. A reference run with `embed_dim=128`, `num_heads=8`,
`ff_dim=512` and everything else at the defaults reached **1.577** after 3000 steps and **1.546**
after 6000 — at which point the curve had essentially flattened while the *training* loss kept
falling to 1.271. That widening gap is overfitting, and it is roughly where this model size runs out
of room. Getting far below 1.5 needs a genuinely larger model than a free Colab session will train
comfortably.
{{% /notice %}}

{{% notice info %}}
These targets assume the Tiny Shakespeare corpus with the 90/10 split from Phase 1. If you swapped in
your own corpus the absolute numbers will be different — a corpus with a larger vocabulary always
starts at a higher loss. In that case, compete against your *own* baseline instead.
{{% /notice %}}

## Tasks

- [ ] Re-read the roadmap and note which knob affects which part of the pipeline
- [ ] Change **one** hyperparameter at a time, retrain, and record the validation loss
- [ ] Reach at least the Bronze tier
- [ ] Generate a 400-character sample and compare it with your first one from Phase 4
- [ ] Save the model and confirm it reloads

{{% notice tip %}}
Every variable left open for tuning:

| Where | Parameter | Default |
| ----- | --------- | ------- |
| Phase 1 | `block_size` | 128 |
| Phase 3 | `embed_dim` | 64 |
| Phase 3 | `num_heads` | 4 |
| Phase 3 | `ff_dim` | 256 |
| Phase 3 | `num_blocks` | 4 |
| Phase 3 | `dropout_rate` | 0.1 |
| Phase 3 | `learning_rate` | 1e-3 |
| Phase 3 | `batch_size` | 64 |
| Phase 3 | `steps` | 3000 |
| Phase 4 | `temperature`, `top_k`, `top_p` | 0.8 / none / none |
{{% /notice %}}

{{% notice info %}}
**Retraining is not optional.** Changing any architecture parameter (`embed_dim`, `num_heads`,
`ff_dim`, `num_blocks`, `block_size`) means you must re-run `build_gpt()`, the compile cell **and**
the training cell. Changing only `learning_rate`, `batch_size` or `steps` still requires re-running
the compile and training cells. Sampling parameters (`temperature`, `top_k`, `top_p`) change nothing
about the model and need no retraining at all — so they can never improve your loss.
{{% /notice %}}

## Submit Your Result

There is no submission server for this lab — you score yourself. Bring the following to the group
discussion at the end of the workshop:

1. Your best **validation loss** and **perplexity**
2. The **hyperparameters** that produced it
3. A **400-character sample** generated at `temperature=0.8, top_k=40`
4. One sentence on **which change helped most** — and one on which change surprisingly did not

```python
#@title Produce your final report
print("Validation loss :", round(estimate_val_loss(iters=50), 4))
print("Hyperparameters :", dict(block_size=block_size, embed_dim=embed_dim,
                                num_heads=num_heads, ff_dim=ff_dim,
                                num_blocks=num_blocks, dropout_rate=dropout_rate,
                                learning_rate=learning_rate, batch_size=batch_size,
                                steps=steps))
print("Parameters      :", model.count_params())
print("\n--- Sample ---\n")
print(generate(model, "ROMEO:", max_new_tokens=400, temperature=0.8, top_k=40))
```

## Hints and Tips

{{% expand title="Hint 1 - the cheapest win" %}}
**Train longer.** Increase `steps` from 3000 to 6000-8000 and watch the validation loss keep falling
before it plateaus. Nothing else is this simple. On a T4 GPU 8000 steps is still under 15 minutes.
{{% /expand %}}

{{% expand title="Hint 2 - the model is too small" %}}
A model that is too small cannot represent the language. `embed_dim=128` with `num_heads=8` is the
single most effective change — on its own it takes the 3000-step result from 1.70 to about 1.58.
Remember: **`num_heads` must divide `embed_dim` evenly**, or the block will not build.
{{% /expand %}}

{{% expand title="Hint 3 - widen the feed-forward layer" %}}
Most of a Transformer's parameters live in the FFN. Raising `ff_dim` from 256 to 512 gives each block
substantially more capacity to store patterns, at a modest speed cost.
{{% /expand %}}

{{% expand title="Hint 4 - give it more history" %}}
A larger `block_size` (192-256) lets the model use more preceding text when predicting the next
character, which mostly helps coherence across a line. Memory use grows with the *square* of
`block_size`, so raise it carefully.
{{% /expand %}}

{{% expand title="Hint 5 - reading the overfitting signal" %}}
Watch the **gap** between training and validation loss, not just the validation number. In the
reference run above it grew from 0.16 at 3000 steps to 0.275 at 6000 while validation barely moved —
the model had started memorizing rather than generalizing. When you see that, raise `dropout_rate`
to 0.2-0.3, or simply stop training earlier. More steps stop helping long before they start hurting.
{{% /expand %}}

{{% expand title="Hint 6 - learning rate" %}}
`1e-3` is a good default at this size. If you make the model much larger, `1e-3` can become unstable —
if your loss spikes or turns into `nan`, drop to `5e-4` or `3e-4`. If training seems to crawl, the
learning rate is likely too low.
{{% /expand %}}

{{% expand title="Hint 7 - make the sample look good" %}}
For the *generated sample* (not the loss), `temperature=0.8` with `top_k=40` or `top_p=0.9` usually
reads best. This cannot change your score, but it makes your demo output far sharper.
{{% /expand %}}

{{% expand title="Hint 8 - change the personality" %}}
For a completely different result, point `CORPUS_URL` at a different text and retrain. The model will
learn that style instead. Compare how quickly the loss falls on structured text (source code, logs)
versus prose — structured text is far more predictable, so the loss goes much lower.
{{% /expand %}}
