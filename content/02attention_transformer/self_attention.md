---
title: "Self-Attention Intuition"
linkTitle: "Self-Attention Intuition"
weight: 2
---

## The Core Idea

To predict the next token well, the model must decide **which earlier tokens matter right now**.
In the sentence *"The server crashed because it ran out of memory"*, to understand *"it"* the model
must look back at *"server"*. **Self-attention** is the mechanism that lets every token look at every
other token and decide how much to pay attention to each.

You saw self-attention in AI 301 at a high level. Here we open it up.

## Query, Key, Value

Every token produces three vectors, all learned:

- **Query (Q):** "what am I looking for?"
- **Key (K):** "what do I offer / what am I about?"
- **Value (V):** "what information do I pass on if attended to?"

A token's query is compared against every token's key. High similarity → high attention weight →
that token's value contributes more to the result.

```mermaid
graph TD
    A[Token vector] --> Q[Query]
    A --> K[Key]
    A --> V[Value]
    Q -->|compare with all Keys| S[Attention scores]
    S -->|softmax| W[Attention weights]
    W -->|weighted sum of Values| O[Output: context-aware vector]
```

## The Attention Formula

The famous equation from *"Attention Is All You Need"* (2017):

```math
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$
```

In plain language:

1. **`Q Kᵀ`** — for each token, score how relevant every other token is (dot product = similarity).
2. **`/ √dₖ`** — scale down so the scores don't explode as vectors get larger.
3. **`softmax`** — turn scores into weights that sum to 1 (a probability distribution over tokens).
4. **`× V`** — take a weighted average of the value vectors. The result is each token "rewritten" in
   the context of the tokens it found relevant.

```python
#@title Scaled dot-product attention, by hand
def scaled_dot_product_attention(q, k, v, mask=None):
    """q, k, v shapes: (batch, seq_len, d_k). Returns (output, attention_weights)."""
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)   # (batch, seq, seq)
    if mask is not None:
        scores += (mask * -1e9)        # push masked positions to -infinity before softmax
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, v), weights


# Run it on 4 random tokens and look at the attention weights
demo = tf.random.normal((1, 4, 8))
out, w = scaled_dot_product_attention(demo, demo, demo)
print("Output shape           :", out.shape)          # (1, 4, 8) - same shape as the input
print("Attention weight matrix:\n", np.round(w[0].numpy(), 2))
print("Every row sums to 1    :", np.allclose(w[0].numpy().sum(axis=-1), 1.0))
```

Each **row** of that matrix is one token's answer to "how much do I care about each token in the
sequence?" — and because of the softmax, every row sums to exactly 1. Right now the weights are
meaningless (the input was random noise); after training they encode real linguistic relationships.

## Multi-Head Attention

One set of Q/K/V can only capture one kind of relationship. **Multi-head attention** runs several
attention operations in parallel, each with its own learned Q/K/V, then concatenates the results.

As in the AI 301 classifier:

- Head 1 might track subject ↔ pronoun links
- Head 2 might track verb tense
- Head 3 might track topic / named entities

```mermaid
graph LR
    X[Input] --> H1[Head 1]
    X --> H2[Head 2]
    X --> H3[Head 3]
    H1 --> C[Concatenate]
    H2 --> C
    H3 --> C
    C --> P[Linear projection]
```

{{% notice info %}}
Keras gives us a battle-tested `layers.MultiHeadAttention`, so in the actual model we will use that
rather than the hand-written version above. The hand-written version exists so you can see there is
no magic inside — it is dot products, a scale, a softmax, and a weighted sum.
{{% /notice %}}

<!-- Renders the Mermaid diagrams on this page.
     The fortinet-hugo image sets `mermaid = false` in its generated hugo.toml, which in
     Relearn 8 disables the theme's Mermaid dependency entirely: the diagram markup is
     emitted, but no Mermaid library is ever loaded and the theme's CSS keeps every
     .mermaid block at `visibility: hidden`. Remove this block once the image is fixed. -->
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  if (!window.__ftntMermaidLoaded) {
    window.__ftntMermaidLoaded = true;
    mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: "default" });
    for (const el of document.querySelectorAll("pre.mermaid")) {
      try {
        await mermaid.run({ nodes: [el] });
      } catch (e) {
        console.error("Mermaid failed to render a diagram on this page:", e);
      }
      // Relearn only un-hides a diagram once its own script adds .mermaid-render.
      el.classList.add("mermaid-render");
    }
  }
</script>
