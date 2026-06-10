---
title: "AI 102 - How LLMs Are Built & What GenAI Means"
linkTitle: "AI 102 - How LLMs Are Built & What GenAI Means"
weight: 10
archetype: "home"
description: "GenAI Lab - From Tokens to Transformers to a Generating Language Model"
---

# Welcome to the GenAI Creator Lab!

This is the follow-up to **AI 101 - Building Supervised ML Models**. In AI 101 you built a
Transformer-based classifier that produced a single deterministic answer: *spam* or *ham*.
In this lab we keep the same Transformer machinery but change the **objective**: instead of
classifying text, the model will **generate** text — one token at a time. That single shift in
objective is the whole story of how a Large Language Model (LLM) works.

## In this lab you will

- Understand how raw text becomes **tokens** and then **embeddings**
- Build the **self-attention** mechanism that lets a model weigh context
- Assemble a small **decoder-only Transformer** (the GPT-style architecture)
- **Train** it on real text so it learns to predict the next token
- Run **inference** with temperature, top-k and top-p sampling and watch it generate
- See where **foundation models, fine-tuning, RAG, grounding, agents, tools and MCP** fit in
- Walk away with a precise **glossary** so the buzzwords stop being fuzzy

## What This Lab Covers — and What It Does Not

- This lab focuses on **generative** language models (the "GPT" family of decoder-only Transformers).
- You will train a *small* model from scratch so every step is visible and runs on free hardware.
- This is **not** a guide to training GPT-4-scale models — the principles are identical, only the
  data, parameters and compute differ by many orders of magnitude.
- We use a character/word-level model for clarity; production LLMs use subword tokenization at massive scale.

## Learning Goals

By the end of this lab you will be able to:

- Explain the difference between a **classifier** and a **generative** model
- Describe the full LLM build pipeline: data → tokenize → embed → attention → train → infer
- Explain **next-token prediction** and why it is the core training objective of every LLM
- Define **foundation model**, **inferencing**, **grounding**, **fine-tuning**, **RAG**, **agentic AI**, **tools** and **MCP** — and tell them apart
- Tune sampling parameters (**temperature**, **top-k**, **top-p**) and predict their effect
- Reason about *why* LLMs hallucinate and how grounding mitigates it

## Lab Structure (Phases)

This lab is organized into six phases plus a final challenge:

| Phase | Title | What you build / learn |
| ----- | ----- | ---------------------- |
| 0 | Introduction | What GenAI is, classifier vs generator, the build roadmap |
| 1 | Tokenization & Embeddings | Turn text into numbers the model can learn from |
| 2 | Attention & the Transformer | Build self-attention and a decoder block by hand |
| 3 | The Training Pipeline | Next-token prediction, loss, training loop |
| 4 | Inference & Grounding | Sampling, hallucination, RAG, grounding |
| 5 | The GenAI Ecosystem | Foundation models, fine-tuning, agents, tools, MCP + glossary |
| 6 | Final Challenge | Improve generation quality and submit your model |
