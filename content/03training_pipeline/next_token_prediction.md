---
title: "Next-Token Prediction & Training Data"
linkTitle: "Next-Token Prediction"
weight: 2
---

## The One Objective Behind Every LLM

An LLM is trained to do exactly one thing: **given a sequence of tokens, predict the next one.**
That's it. The astonishing range of LLM behavior — translation, summarization, coding, reasoning —
all emerges from getting very, very good at this single objective on enough data.

```mermaid
graph LR
    A["Input (context): 'To be or not to b'"] --> M[Model]
    M --> B["Output: probability for EVERY possible next token"]
    B --> C["Highest: 'e'  (0.91)"]
```

## Turning One Stream Into Millions of Examples

Recall we have one long array of token IDs and **no labels**. Here is the trick that creates training
data for free: slide a window of length `block_size` over the corpus. For each window, the **input**
is the window and the **target** is the *same window shifted one position to the right*. Position by
position, the target tells the model "this is the token that actually came next."

```text
context (x):  T  o     b  e     o  r
target  (y):  o     b  e     o  r     n
              ^each x position must predict the y in the same column
```

So a window of 8 tokens is really 8 prediction problems at once:

- given `T` → predict `o`
- given `T o` → predict ` `
- given `T o ` → predict `b`
- ... and so on.

The causal mask you built is what makes this safe to do in a single pass: each position can only see
its own past, so it can't cheat by reading the target.

```python
#@title Build (input, target) pairs from the corpus
batch_size = 64          # how many windows per training step
# block_size = 128 was already defined with the embedding layer

def get_batch(source, batch_size, block_size):
    """Cut batch_size random windows out of `source`; y is x shifted one token right."""
    starts = np.random.randint(0, len(source) - block_size - 1, size=batch_size)
    x = np.stack([source[i     : i + block_size    ] for i in starts])
    y = np.stack([source[i + 1 : i + block_size + 1] for i in starts])   # shifted by 1
    return tf.constant(x), tf.constant(y)

xb, yb = get_batch(train_data, batch_size, block_size)
print("Input  batch shape:", xb.shape)   # (64, 128)
print("Target batch shape:", yb.shape)   # (64, 128)
print("\nFirst example, decoded input :", repr(decode(xb[0][:40].numpy())))
print("First example, decoded target:", repr(decode(yb[0][:40].numpy())))
```

{{% notice tip %}}
Compare the printed input and target: the target is just the input nudged one character forward. That
one-character shift is the entire supervision signal of a language model. No humans labeled anything.
{{% /notice %}}

## The Loss Function

For each position the model outputs a probability distribution over the whole vocabulary. We compare
it to the true next token using **cross-entropy loss** (the same loss family used for classification
in AI 301 — predicting the next token *is* a classification problem with `vocab_size` classes).

```math
$$
\mathrm{Loss} = -\frac{1}{N}\sum_{i=1}^{N} \log P_\theta(\text{true next token}_i \mid \text{context}_i)
$$
```

In words: the loss is low when the model assigned a high probability to the token that actually came
next, and high when it was "surprised." Training drives this surprise down.

{{% notice info %}}
You may have heard the word **perplexity**. It is simply the exponential of this loss — an intuitive
measure of "on average, how many tokens is the model choosing between?" Lower perplexity = a more
confident, better language model.
{{% /notice %}}

{{% notice tip %}}
A useful reference point before you start: an *untrained* model assigns equal probability to all 65
characters, so its loss is `ln(65) ≈ 4.17` and its perplexity is 65 — it is guessing blindly. Watch
for that number in the first training step, and treat everything below it as learning.
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
