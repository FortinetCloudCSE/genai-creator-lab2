---
title: "Causal Masking - Why It Can't See the Future"
linkTitle: "Causal Masking"
weight: 3
---

## The Single Most Important Difference vs. AI 101

The classifier in AI 101 used an **encoder**: every token could attend to every other token, in both
directions, because the whole input was available at once.

A generative model must not do that. When it is learning to predict the token at position 5, it must
**only** be allowed to look at positions 1–4. If it could peek at position 5 (or beyond), it would
"cheat" — it would just copy the answer, learn nothing, and be useless at generation time when the
future genuinely doesn't exist yet.

We enforce this with a **causal mask** (also called a *look-ahead* or *triangular* mask).

```mermaid
graph TD
    subgraph "Allowed attention for token 4"
    T1[Token 1] --> T4
    T2[Token 2] --> T4
    T3[Token 3] --> T4
    T4[Token 4] --> T4
    T5[Token 5 - BLOCKED] -.x.-> T4
    end
```

## What the Mask Looks Like

For a 5-token sequence, the mask blocks (sets to `-infinity` before softmax) every position to the
*right* of the current one. A `1` means "block this":

```text
            attend to: t1 t2 t3 t4 t5
predict t1:            0  1  1  1  1
predict t2:            0  0  1  1  1
predict t3:            0  0  0  1  1
predict t4:            0  0  0  0  1
predict t5:            0  0  0  0  0
```

After softmax, the blocked positions get weight 0, so each token's new representation depends only on
itself and the tokens before it.

```python
#@title Build a causal mask
def causal_mask(seq_len):
    # 1 where attention should be BLOCKED (upper triangle, excluding diagonal)
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # shape (seq_len, seq_len)

print(causal_mask(5).numpy().astype(int))
```

{{% notice info %}}
This one design choice — causal masking — is what makes a Transformer "decoder-only" / "autoregressive"
/ "GPT-style." **Autoregressive** simply means *each output depends on the outputs that came before
it.* Every token is predicted from its left context, then fed back in to help predict the next one.
{{% /notice %}}

{{% notice tip %}}
The clever payoff: during *training*, the causal mask lets the model learn to predict **all positions
at once in a single forward pass** (each position only sees its own past). During *generation*, the
same mask means it can produce one token, append it, and roll forward. Same machinery, two modes.
{{% /notice %}}
