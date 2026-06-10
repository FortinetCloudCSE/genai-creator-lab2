---
title: "Glossary - Terms, Sorted Out"
linkTitle: "Glossary"
weight: 4
---

A precise reference for the terms used in this lab and in everyday GenAI conversations, grouped so
the easily-confused ones sit side by side. Where two terms are commonly mixed up, the entry says so
explicitly.

## Core Concepts

**Artificial Intelligence (AI)** — The broad field of building systems that perform tasks requiring
human-like intelligence. The umbrella over everything below.

**Machine Learning (ML)** — A subset of AI: systems that learn patterns from data instead of being
explicitly programmed. (Covered in AI 101.)

**Deep Learning (DL)** — A subset of ML using multi-layer neural networks. Transformers are deep
learning.

**Generative AI (GenAI)** — Models that produce *new* content (text, images, code, audio) by learning
the distribution of their training data. Contrast with **discriminative** models, which classify
existing inputs.

**Large Language Model (LLM)** — A large generative model trained on text, almost always a
decoder-only Transformer, whose sole training objective is next-token prediction.

## Architecture & Mechanics

**Token** — The atomic unit of text a model reads/writes (a character, word, or subword). Text is
converted to a sequence of integer token IDs.

**Tokenization** — Splitting text into tokens and mapping them to IDs. Production models use **subword
/ Byte-Pair Encoding (BPE)**.

**Embedding** — A learned vector representing a token (or other item) such that similar items have
similar vectors. Encodes meaning numerically.

**Positional encoding** — Information added to embeddings telling the model *where* each token sits,
since attention itself is order-blind.

**Transformer** — The neural architecture built on self-attention. **Encoder** variants (AI 101) read
bidirectionally for understanding/classification; **decoder** variants (this lab) read left-to-right
for generation.

**Self-attention** — The mechanism by which each token weighs the relevance of every other token
(via Query/Key/Value) to build a context-aware representation.

**Causal / look-ahead mask** — The constraint that a token may only attend to earlier tokens. What
makes a Transformer "decoder-only" and able to generate.

**Autoregressive** — Generating output one token at a time, each conditioned on all previously
generated tokens.

**Context window (block size)** — The maximum number of tokens a model can consider at once. Older
content "falls out" of the window.

**Parameters / weights** — The learned numbers inside the model. "175B parameters" = 175 billion such
numbers.

## Training

**Pre-training** — The first, expensive stage: self-supervised next-token prediction on huge text.
Produces a **foundation model**.

**Self-supervised learning** — Training where the labels come from the data itself (the next token),
needing no human annotation.

**Next-token prediction** — The single objective of LLM pre-training: predict the following token from
the preceding ones.

**Loss / cross-entropy** — The measure of how "surprised" the model is by the true next token.
Training minimizes it. **Perplexity** is its exponential, an intuitive "how many options is it
choosing among" score.

**Backpropagation & gradient descent** — How weights are updated to reduce loss (same as AI 101).

**Foundation model** — A large model pre-trained on broad data, reused as a base for many tasks via
prompting, RAG, or fine-tuning. Examples: GPT, Claude, Gemini, Llama.

**Fine-tuning** — Continuing to train a foundation model on task-specific data so new behavior is
baked into the weights. Best for *style/format/behavior*, not for facts.

- **SFT (Supervised Fine-Tuning)** — fine-tuning on (prompt → ideal answer) pairs.
- **LoRA / PEFT** — fine-tuning only a small set of added parameters; cheap and standard.
- **RLHF** — aligning a model to human preferences; turns a raw completer into a helpful assistant.

## Using the Model

**Inference / inferencing** — *Using* a trained model to produce output (weights fixed). Every prompt
you send is an inference call. Contrast with *training*.

**Sampling** — How the next token is chosen from the model's output probabilities.

- **Temperature** — scales randomness; low = focused/deterministic, high = creative/risky.
- **Top-k** — sample only from the k most likely tokens.
- **Top-p (nucleus)** — sample from the smallest set of tokens whose probabilities sum to p.

**Hallucination** — Fluent but false or unsupported output. A direct consequence of optimizing for
*plausibility*, not *truth*.

**Prompt** — The input text given to the model. **Few-shot prompting** includes examples in the
prompt; **system prompt** sets persistent instructions/behavior.

## The Ecosystem (Most Confused Terms)

**Grounding** — Constraining/informing output with authoritative external information to reduce
hallucination. The *goal*.

**RAG (Retrieval-Augmented Generation)** — The most common *technique* for grounding: retrieve
relevant documents and inject them into the prompt at inference time. **Does not change model
weights.**

{{% notice info %}}
**Grounding vs. RAG:** grounding is the *objective* (tie output to real sources); RAG is the most
common *method* of achieving it. **RAG vs. fine-tuning:** RAG adds knowledge at query time without
touching weights; fine-tuning changes weights and is for behavior/style. Use RAG for facts,
fine-tuning for behavior.
{{% /notice %}}

**Vector database** — Stores embeddings so the most *semantically similar* documents to a query can be
retrieved quickly. The retrieval engine behind RAG.

**Tool / function calling** — An external capability (API, DB, search, code) the model can invoke. The
model emits a structured call; the app executes it; the result returns to the model's context.

**Agentic AI / agent** — A system where the model operates in a loop — plan, act (often via tools),
observe, repeat — with autonomy over the sequence of steps toward a goal. More than a single
question-and-answer.

**MCP (Model Context Protocol)** — An *open standard / protocol* for connecting models to tools and
data sources. The "USB-C for AI tools." An **MCP server** exposes a tool/data source; an **MCP
client** consumes it.

{{% notice tip %}}
**The three most-confused terms, settled:**
- **Tool** = a capability the model can call (the *hands*).
- **Agent** = a model that decides *when/how* to use tools in a loop (the *decision-maker*).
- **MCP** = the standardized *plug* connecting models to tools (the *socket*).
A tool is not an agent; MCP is neither — it's the wiring standard between them.
{{% /notice %}}

## Security-Relevant Terms (bonus for this audience)

**Prompt injection** — An attack where malicious instructions are hidden in text the model reads
(a document, web page, retrieved RAG chunk) to hijack its behavior.

**Jailbreak** — A prompt crafted to bypass a model's safety alignment.

**Data exfiltration via tools/agents** — Risk that an over-permissioned agent or tool leaks or
misuses data. Expands the trust boundary to every connected tool and data source.

**Model context cutoff** — The date after which a model has no training knowledge; a key reason to use
RAG/tools for current information.
