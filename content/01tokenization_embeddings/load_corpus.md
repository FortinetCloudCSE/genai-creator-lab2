---
title: "Load the Text Corpus"
linkTitle: "Load the Text Corpus"
weight: 2
---

## Why an LLM Needs a Corpus, Not Labels

In AI 101 every email came with a **label** (spam/ham) that a human had assigned. Training a
generative model needs no labels at all — the text is its own supervision. The "correct answer" for
any position is simply *the token that actually came next* in the real text. This is what makes
LLM training **self-supervised** and is why models can be trained on essentially the whole internet
without anyone hand-labeling it.

## Load a Small Corpus

We will use a small, clean, public-domain text corpus so training is fast and the output is
recognizable. The [`tiny_shakespeare`](https://huggingface.co/datasets/tiny_shakespeare) dataset is a
classic choice — it is small, stylistically consistent, and the model learns to imitate it quickly.

```python
#@title Load the tiny_shakespeare corpus
dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)

text = dataset["train"][0]["text"]
print("Corpus length (characters):", len(text))
print("\n--- First 300 characters ---\n")
print(text[:300])
```

{{% notice tip %}}
You can swap in any plain-text corpus you like — a book, your own documents, log files, even ABAP or
Python source. The model will learn to generate text in the style of whatever you feed it. The
corpus *is* the personality of your model.
{{% /notice %}}

## Inspect the Vocabulary

Before tokenizing, let's see what raw symbols the corpus contains. For a character-level model the
"vocabulary" is just the set of unique characters.

```python
#@title Inspect the character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Number of unique characters (vocab size):", vocab_size)
print("Characters:", "".join(chars))
```

{{% notice info %}}
We use **character-level** tokenization in this lab because it is the easiest to see and reason
about: the vocabulary is tiny (~65 symbols) and there is nothing hidden. Real LLMs use **subword**
tokenization (Byte-Pair Encoding, WordPiece, SentencePiece), which strikes a balance between
character-level (too many steps) and word-level (vocabulary too large, can't handle new words). The
*principle* — text becomes a sequence of integers — is identical. We cover subword tokenization on
the next page.
{{% /notice %}}
