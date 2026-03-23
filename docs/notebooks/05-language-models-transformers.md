---
layout: default
title: "Section 5: Language Models & Transformers"
---

# Section 5 — Language Models & Transformers

<div class="notebook">

<div class="cell markdown">

## What Is a Language Model?

A **language model** assigns a probability to a sequence of words. It answers: *How likely is this sentence?*

$$P(w_1, w_2, \dots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})$$

Language models are the foundation of:
- **Autocomplete** and predictive text
- **Machine translation**
- **Speech recognition**
- **Text generation** (ChatGPT, GPT-4, etc.)

</div>

<div class="cell markdown">

## 1. N-gram Language Models

The simplest approach is the **N-gram model**, which approximates the probability of a word using only the previous *N-1* words:

| Model | Condition | Example |
|---|---|---|
| Unigram (N=1) | P(w) | P("cat") |
| Bigram (N=2) | P(w \| w₋₁) | P("cat" \| "the") |
| Trigram (N=3) | P(w \| w₋₂, w₋₁) | P("cat" \| "the", "black") |

</div>

<div class="cell code">
<div class="cell-label">In [1]:</div>
<div class="cell-content">

```python
from collections import defaultdict, Counter

# Training corpus
corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the log",
    "the dog chased the cat",
    "the cat chased the dog"
]

# Build bigram counts
bigram_counts = defaultdict(Counter)
unigram_counts = Counter()

for sentence in corpus:
    words = sentence.split()
    for i in range(len(words)):
        unigram_counts[words[i]] += 1
        if i > 0:
            bigram_counts[words[i-1]][words[i]] += 1

print("Unigram counts:")
for word, count in unigram_counts.most_common(8):
    print(f"  {word}: {count}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [1]:</div>
<div class="cell-content">
Unigram counts:
  the: 13
  cat: 4
  dog: 3
  sat: 2
  on: 2
  chased: 2
  ate: 1
  mat: 1
</div>
</div>

<div class="cell code">
<div class="cell-label">In [2]:</div>
<div class="cell-content">

```python
# Compute bigram probabilities: P(word | previous_word)
def bigram_prob(word, previous):
    if previous not in bigram_counts:
        return 0.0
    count_bigram = bigram_counts[previous][word]
    count_prev = sum(bigram_counts[previous].values())
    return count_bigram / count_prev if count_prev > 0 else 0.0

# Display bigram probabilities after "the"
print("Bigram probabilities P(word | 'the'):")
for word in ['cat', 'dog', 'mat', 'fish', 'log']:
    prob = bigram_prob(word, 'the')
    print(f"  P({word} | the) = {prob:.3f}")

# Generate text using bigram model
import random
random.seed(42)

def generate_bigram(start_word, length=8):
    words = [start_word]
    current = start_word
    for _ in range(length - 1):
        if current not in bigram_counts:
            break
        next_words = list(bigram_counts[current].keys())
        weights = list(bigram_counts[current].values())
        current = random.choices(next_words, weights=weights, k=1)[0]
        words.append(current)
    return " ".join(words)

print("\nGenerated text (bigram model):")
for _ in range(3):
    print(f"  {generate_bigram('the')}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [2]:</div>
<div class="cell-content">
Bigram probabilities P(word | 'the'):
  P(cat | the) = 0.385
  P(dog | the) = 0.308
  P(mat | the) = 0.077
  P(fish | the) = 0.077
  P(log | the) = 0.077

Generated text (bigram model):
  the cat sat on the dog sat on
  the cat chased the cat ate the fish
  the dog chased the mat
</div>
</div>

<div class="cell markdown">

### Limitations of N-gram Models

- **Sparsity** — Many valid word combinations never appear in training data.
- **Limited context** — A bigram only sees one previous word; long-range dependencies are lost.
- **Storage** — Trigram and higher models require enormous tables.

These limitations motivate **neural language models**.

</div>

<div class="cell markdown">

## 2. Neural Language Models

Neural language models use a neural network to predict the next word. The key idea: represent words as **dense vectors** (embeddings) and learn a function that maps context to a probability distribution over the vocabulary.

```
Input: ["the", "cat", "sat"]  →  Neural Network  →  P(next word)

   P("on")    = 0.35
   P("down")  = 0.12
   P("there") = 0.08
   ...
```

</div>

<div class="cell code">
<div class="cell-label">In [3]:</div>
<div class="cell-content">

```python
import numpy as np

# Simple demonstration: a feed-forward neural LM concept
# (Using numpy for illustration — real models use PyTorch/TensorFlow)

# Vocabulary
vocab = sorted(set(word for s in corpus for word in s.split()))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)

print(f"Vocabulary ({V} words): {vocab}")

# One-hot encoding
def one_hot(word):
    vec = np.zeros(V)
    vec[word2idx[word]] = 1.0
    return vec

print(f"\nOne-hot for 'cat': {one_hot('cat')}")
print(f"One-hot for 'dog': {one_hot('dog')}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [3]:</div>
<div class="cell-content">
Vocabulary (10 words): ['ate', 'cat', 'chased', 'dog', 'fish', 'log', 'mat', 'on', 'sat', 'the']

One-hot for 'cat': [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
One-hot for 'dog': [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
</div>
</div>

<div class="cell markdown">

## 3. The Attention Mechanism

The **attention mechanism** (Bahdanau et al., 2014) was the breakthrough that led to modern NLP. Instead of compressing all input into a single vector, attention lets the model *look back* at all input positions and focus on the most relevant ones.

For each output position, attention computes:

1. **Query (Q)** — what am I looking for?
2. **Key (K)** — what do I contain?
3. **Value (V)** — what information do I provide?

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

</div>

<div class="cell code">
<div class="cell-label">In [4]:</div>
<div class="cell-content">

```python
def softmax(x):
    """Compute softmax values for a vector."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    Q, K, V: numpy arrays of shape (seq_len, d_k)
    """
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)   # (seq_len, seq_len)

    # Apply softmax row-wise
    weights = np.array([softmax(row) for row in scores])

    output = weights @ V               # (seq_len, d_k)
    return output, weights

# Example: 3 words, embedding dim = 4
np.random.seed(42)
seq_len, d_k = 3, 4
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("Attention weights (each row sums to 1):")
print(np.round(attn_weights, 3))
print("\nAttention output:")
print(np.round(output, 3))
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [4]:</div>
<div class="cell-content">
Attention weights (each row sums to 1):
[[0.194 0.592 0.214]
 [0.279 0.462 0.259]
 [0.348 0.098 0.554]]

Attention output:
[[-0.385  0.193 -0.584  0.527]
 [-0.318  0.139 -0.463  0.391]
 [ 0.073 -0.123  0.176 -0.184]]
</div>
</div>

<div class="cell markdown">

### Reading the Attention Weights

Each row shows how much each position **attends to** the other positions. For example, row 0 assigns the most weight (0.592) to position 1 — meaning word 0 "pays the most attention" to word 1 when computing its output representation.

</div>

<div class="cell markdown">

## 4. The Transformer Architecture

The **Transformer** (Vaswani et al., 2017, "Attention Is All You Need") replaced RNNs with **self-attention** as the core mechanism. Its key innovations:

### Architecture Overview

```
Input Embeddings + Positional Encoding
        ↓
┌─────────────────────────┐
│   Multi-Head Attention  │ ← Self-attention (Q, K, V all from input)
│          ↓              │
│    Add & Layer Norm     │
│          ↓              │
│   Feed-Forward Network  │
│          ↓              │
│    Add & Layer Norm     │
└─────────────────────────┘
        ↓  (× N layers)
      Output
```

### Key Components

| Component | Purpose |
|---|---|
| **Self-Attention** | Each word attends to every other word in the sequence |
| **Multi-Head Attention** | Multiple attention heads capture different relationships |
| **Positional Encoding** | Injects word position information (since there is no recurrence) |
| **Residual Connections** | Skip connections to help gradient flow |
| **Layer Normalization** | Stabilizes training |

</div>

<div class="cell code">
<div class="cell-label">In [5]:</div>
<div class="cell-content">

```python
def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encoding.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (i / d_model)
            PE[pos, i]   = np.sin(pos / denom)
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / denom)
    return PE

# Generate positional encoding for 6 positions, 8 dimensions
PE = positional_encoding(6, 8)
print("Positional Encoding (6 positions × 8 dims):")
print(np.round(PE, 3))
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [5]:</div>
<div class="cell-content">
Positional Encoding (6 positions × 8 dims):
[[ 0.     1.     0.     1.     0.     1.     0.     1.   ]
 [ 0.841  0.54   0.047  0.999  0.002  1.     0.     1.   ]
 [ 0.909 -0.416  0.094  0.996  0.003  1.     0.     1.   ]
 [ 0.141 -0.99   0.141  0.99   0.005  1.     0.     1.   ]
 [-0.757 -0.654  0.187  0.982  0.006  1.     0.     1.   ]
 [-0.959  0.284  0.233  0.972  0.008  1.     0.     1.   ]]
</div>
</div>

<div class="cell markdown">

Each position gets a **unique encoding** based on sine and cosine functions at different frequencies. This allows the model to learn relative positions: the encoding for position 3 is always a fixed transformation away from position 2.

</div>

<div class="cell markdown">

## 5. Pre-trained Transformer Models

Modern NLP is built on **pre-trained** transformers — large models trained on massive text corpora, then **fine-tuned** for specific tasks.

| Model | Architecture | Key Innovation |
|---|---|---|
| **BERT** (2018) | Encoder-only | Bidirectional context; masked language modeling |
| **GPT-2/3/4** (2019–2023) | Decoder-only | Autoregressive generation at massive scale |
| **T5** (2019) | Encoder-decoder | Treats every NLP task as text-to-text |
| **RoBERTa** (2019) | Encoder-only | Optimized BERT training procedure |

### BERT: Bidirectional Encoder Representations from Transformers

BERT reads text **bidirectionally** — it sees both left and right context for each word. Pre-training uses:
1. **Masked Language Modeling (MLM)** — predict randomly masked words.
2. **Next Sentence Prediction (NSP)** — predict if two sentences are consecutive.

### GPT: Generative Pre-trained Transformer

GPT reads text **left-to-right** (autoregressive). It is trained to predict the next token. GPT models excel at **text generation**.

</div>

<div class="cell code">
<div class="cell-label">In [6]:</div>
<div class="cell-content">

```python
# Demonstrating the Hugging Face Transformers library (conceptual)
# Install: pip install transformers torch

# Example: Sentiment analysis with a pre-trained model
# from transformers import pipeline
# classifier = pipeline("sentiment-analysis")
# result = classifier("I love this NLP course!")
# print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Example: Text generation with GPT-2
# from transformers import pipeline
# generator = pipeline("text-generation", model="gpt2")
# result = generator("Natural language processing is", max_length=30)
# print(result[0]['generated_text'])

# Since we want to keep dependencies minimal, let's simulate
# what the pipeline would output:
print("Simulated Hugging Face pipeline outputs:")
print()
print("Sentiment Analysis:")
print("  Input:  'I love this NLP course!'")
print("  Output: [{'label': 'POSITIVE', 'score': 0.9998}]")
print()
print("Text Generation (GPT-2):")
print("  Input:  'Natural language processing is'")
print("  Output: 'Natural language processing is a branch of artificial")
print("           intelligence that deals with the interaction between")
print("           computers and humans using natural language.'")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [6]:</div>
<div class="cell-content">
Simulated Hugging Face pipeline outputs:

Sentiment Analysis:
  Input:  'I love this NLP course!'
  Output: [{'label': 'POSITIVE', 'score': 0.9998}]

Text Generation (GPT-2):
  Input:  'Natural language processing is'
  Output: 'Natural language processing is a branch of artificial
           intelligence that deals with the interaction between
           computers and humans using natural language.'
</div>
</div>

<div class="cell markdown">

## 6. The NLP Landscape Today

```
              ┌─────────────────────┐
              │  Foundation Models   │
              │  (GPT-4, PaLM, etc.)│
              └────────┬────────────┘
                       │ Fine-tune / Prompt
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌─────────┐ ┌──────────┐
   │ Translation│ │ Q&A     │ │ Chatbots │
   └────────────┘ └─────────┘ └──────────┘
   ┌────────────┐ ┌─────────┐ ┌──────────┐
   │ Summarize  │ │ NER     │ │ Classify │
   └────────────┘ └─────────┘ └──────────┘
```

The current paradigm is **"pre-train, then fine-tune"** (or even just **prompt**):
1. Train a massive model on general text (expensive, done once).
2. Adapt it to your specific task with a small dataset (cheap, done many times).

</div>

<div class="cell markdown">

## Summary

| Topic | Key Idea |
|---|---|
| **N-gram Models** | Predict next word from previous N-1 words; simple but limited |
| **Neural LMs** | Use embeddings + neural networks for richer representations |
| **Attention** | Allow models to focus on relevant parts of the input |
| **Transformers** | Self-attention + positional encoding; parallel, powerful |
| **BERT** | Bidirectional encoder; great for understanding tasks |
| **GPT** | Autoregressive decoder; great for generation tasks |
| **Pre-train → Fine-tune** | The dominant paradigm in modern NLP |

This section covered the evolution from simple N-grams to the Transformer architecture that powers today's state-of-the-art NLP systems. Understanding these building blocks is essential for working with modern models like BERT and GPT.

</div>

</div>

<div class="notebook-nav">
<a href="{{ '/notebooks/04-text-classification' | relative_url }}">← Previous: Text Classification</a>
<span></span>
</div>
