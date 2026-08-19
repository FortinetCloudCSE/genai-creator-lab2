---
title: "AI 302 - Train Your Own Small LLM"
linkTitle: "AI 302 - Train Your Own Small LLM"
weight: 10
archetype: "home"
description: "GenAI Lab - From Tokens to Transformers to a Language Model You Trained Yourself"
---

# Welcome to the GenAI Creator Lab!

This lab is the direct follow-up to
[**AI 301 - Building Supervised ML Models**](https://fortinetcloudcse.github.io/genai-creator-lab/).
In AI 301 you built a Transformer-based **classifier** that produced one deterministic answer:
*spam* or *ham*. Here we keep the same Transformer machinery and the same Keras stack, but change
the **objective**: instead of classifying text, your model will **generate** text — one token at a
time. That single shift in objective is the whole story of how a Large Language Model (LLM) works.

By the end of the lab you will have pre-trained a small GPT-style model from scratch, in your own
Google Colab notebook, and watched it learn to write.

{{% notice info %}}
**Prerequisite:** AI 301, or equivalent familiarity with Keras, embeddings, Transformer blocks and
the training loop. If you want the security-side view of what people *build* with LLMs — agents,
tools, MCP and the attacks against them — see
[AI 101 - Agents, MCP & the Agentic Security Model](https://fortinetcloudcse.github.io/ai-101/index.html).
{{% /notice %}}

## In this lab you will

- Turn raw text into **tokens** and then into **embeddings**
- Build the **self-attention** mechanism that lets a model weigh context
- Understand the **causal mask** that makes generation possible
- Assemble a small **decoder-only Transformer** (the GPT-style architecture)
- **Pre-train** it on real text so it learns to predict the next token
- Run **inference** with temperature, top-k and top-p sampling and watch it generate
- See where **foundation models, fine-tuning, RAG, grounding, agents, tools and MCP** fit in
- Walk away with a precise **glossary** so the buzzwords stop being fuzzy

## What This Lab Covers — and What It Does Not

- This lab focuses on **generative** language models (the decoder-only "GPT" family).
- You will train a *small* model from scratch so every step is visible and runs on free hardware.
- This is **not** a guide to training GPT-class models — the principles are identical, only the
  data, parameters and compute differ by many orders of magnitude.
- We use a **character-level** model for clarity; production LLMs use subword tokenization at
  massive scale.
- You will do **pre-training** and **inference** hands-on. Fine-tuning, RLHF and RAG are explained
  but not executed — they need far more compute (or infrastructure) than a free Colab session.

## Learning Goals

By the end of this lab you will be able to:

- Explain the difference between a **classifier** and a **generative** model
- Describe the full LLM build pipeline: data → tokenize → embed → attention → train → infer
- Explain **next-token prediction** and why it is the core training objective of every LLM
- Explain what **causal masking** does and why a decoder needs it
- Tune sampling parameters (**temperature**, **top-k**, **top-p**) and predict their effect
- Reason about *why* LLMs hallucinate and how grounding mitigates it
- Define **foundation model**, **inferencing**, **grounding**, **fine-tuning**, **RAG**,
  **agentic AI**, **tools** and **MCP** — and tell them apart

## Lab Structure and Timing

The lab is designed for **about 3 hours** including the training run.

| Phase | Title | Approx. time | What you build / learn |
| ----- | ----- | ------------ | ---------------------- |
| 0 | Introduction | 20 min | What GenAI is, classifier vs generator, the build roadmap, Colab setup |
| 1 | Tokenization & Embeddings | 25 min | Turn text into numbers the model can learn from |
| 2 | Attention & the Transformer | 30 min | Self-attention, causal masking, the decoder block |
| 3 | The Training Pipeline | 40 min | Next-token prediction, loss, the training run |
| 4 | Inference & Grounding | 25 min | Sampling, hallucination, grounding, RAG |
| 5 | The GenAI Ecosystem | 20 min | Foundation models, fine-tuning, agents, tools, MCP, glossary |
| 6 | Final Challenge | 20 min | Tune your model and beat the target loss |

{{% notice tip %}}
Phase 3 contains a training run: a few minutes on a Colab GPU, considerably longer on CPU. Start
it, then read the Phase 4 theory while it runs — the notebook keeps training in the background.
{{% /notice %}}
