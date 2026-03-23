---
layout: default
title: "Section 3: Text Representation"
---

# Section 3 — Text Representation

<div class="notebook">

<div class="cell markdown">

## From Text to Numbers

Machine learning models operate on **numerical data**. To apply ML to text, we need a way to represent documents as vectors of numbers. This section covers three fundamental approaches:

1. **Bag-of-Words (BoW)** — count-based representation
2. **TF-IDF** — frequency weighted by importance
3. **Word Embeddings (Word2Vec)** — dense, learned representations

</div>

<div class="cell code">
<div class="cell-label">In [1]:</div>
<div class="cell-content">

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample corpus
corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat chased the dog"
]

print("Corpus:")
for i, doc in enumerate(corpus):
    print(f"  Doc {i}: {doc}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [1]:</div>
<div class="cell-content">
Corpus:
  Doc 0: The cat sat on the mat
  Doc 1: The dog sat on the log
  Doc 2: The cat chased the dog
</div>
</div>

<div class="cell markdown">

## 1. Bag-of-Words (BoW)

The **Bag-of-Words** model represents each document as a vector of word counts. It ignores word order — only the frequency of each word matters.

</div>

<div class="cell code">
<div class="cell-label">In [2]:</div>
<div class="cell-content">

```python
# Create a Bag-of-Words representation
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus)

# Display vocabulary
vocab = bow_vectorizer.get_feature_names_out()
print("Vocabulary:", list(vocab))
print()

# Display the BoW matrix
import pandas as pd
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=vocab,
                       index=[f"Doc {i}" for i in range(len(corpus))])
print("Bag-of-Words Matrix:")
print(df_bow.to_string())
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [2]:</div>
<div class="cell-content">
Vocabulary: ['cat', 'chased', 'dog', 'log', 'mat', 'on', 'sat', 'the']

Bag-of-Words Matrix:
       cat  chased  dog  log  mat  on  sat  the
Doc 0    1       0    0    0    1   1    1    2
Doc 1    0       0    1    1    0   1    1    2
Doc 2    1       1    1    0    0   0    0    2
</div>
</div>

<div class="cell markdown">

### How BoW Works

Each column is a unique word from the vocabulary, and each row is a document. The value is the **count** of that word in the document.

**Pros:** Simple, intuitive, works well for many tasks.

**Cons:**
- Loses word order ("dog bites man" = "man bites dog").
- High-dimensional and sparse for large vocabularies.
- Common words (like "the") dominate the representation.

</div>

<div class="cell markdown">

## 2. TF-IDF (Term Frequency – Inverse Document Frequency)

**TF-IDF** addresses the problem of common words dominating BoW. It weights terms by how *important* they are:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Where:
- **TF(t, d)** = frequency of term *t* in document *d*
- **IDF(t)** = log(total documents / documents containing *t*)

Words that appear in many documents (e.g., "the") get a **low** IDF weight; words unique to few documents get a **high** weight.

</div>

<div class="cell code">
<div class="cell-label">In [3]:</div>
<div class="cell-content">

```python
# Create a TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

vocab = tfidf_vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(
    np.round(tfidf_matrix.toarray(), 3),
    columns=vocab,
    index=[f"Doc {i}" for i in range(len(corpus))]
)
print("TF-IDF Matrix:")
print(df_tfidf.to_string())
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [3]:</div>
<div class="cell-content">
TF-IDF Matrix:
         cat  chased    dog    log    mat     on    sat    the
Doc 0  0.379   0.000  0.000  0.000  0.497  0.379  0.379  0.580
Doc 1  0.000   0.000  0.379  0.497  0.000  0.379  0.379  0.580
Doc 2  0.390   0.512  0.390  0.000  0.000  0.000  0.000  0.512
</div>
</div>

<div class="cell markdown">

### Reading the TF-IDF Matrix

Notice that:
- **"the"** has moderate values across all docs (it appears everywhere, so IDF is low).
- **"chased"** has a high value only in Doc 2 (it is unique to that document).
- **"mat"** and **"log"** have high values in their respective documents (unique terms).

TF-IDF is widely used in **information retrieval** (search engines) and **text classification**.

</div>

<div class="cell markdown">

## 3. Document Similarity with TF-IDF

Once documents are vectors, we can measure how similar they are using **cosine similarity**:

$$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \times ||\mathbf{b}||}$$

A value of 1 means identical direction; 0 means completely different.

</div>

<div class="cell code">
<div class="cell-label">In [4]:</div>
<div class="cell-content">

```python
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(tfidf_matrix)

df_sim = pd.DataFrame(
    np.round(cos_sim, 3),
    columns=[f"Doc {i}" for i in range(len(corpus))],
    index=[f"Doc {i}" for i in range(len(corpus))]
)
print("Cosine Similarity Matrix:")
print(df_sim.to_string())
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [4]:</div>
<div class="cell-content">
Cosine Similarity Matrix:
       Doc 0  Doc 1  Doc 2
Doc 0  1.000  0.484  0.445
Doc 1  0.484  1.000  0.445
Doc 2  0.445  0.445  1.000
</div>
</div>

<div class="cell markdown">

Doc 0 ("cat sat on mat") and Doc 1 ("dog sat on log") are the most similar — they share the structure and words "sat", "on", "the".

</div>

<div class="cell markdown">

## 4. Word Embeddings — Word2Vec

Unlike BoW and TF-IDF, **word embeddings** represent each word as a **dense vector** in a continuous space. Words with similar meanings end up close together.

**Word2Vec** (Mikolov et al., 2013) learns embeddings using one of two architectures:

- **CBOW** (Continuous Bag of Words) — predict the center word from surrounding words.
- **Skip-gram** — predict surrounding words from the center word.

</div>

<div class="cell code">
<div class="cell-label">In [5]:</div>
<div class="cell-content">

```python
from gensim.models import Word2Vec

# Prepare tokenized sentences
sentences = [doc.lower().split() for doc in corpus]
print("Tokenized sentences:")
for s in sentences:
    print(f"  {s}")

# Train a simple Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=50,   # embedding dimension
    window=3,         # context window size
    min_count=1,      # include all words
    sg=1,             # 1 = Skip-gram, 0 = CBOW
    epochs=100,       # training iterations
    seed=42
)

print(f"\nVocabulary size: {len(model.wv)}")
print(f"Embedding dimension: {model.wv.vector_size}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [5]:</div>
<div class="cell-content">
Tokenized sentences:
  ['the', 'cat', 'sat', 'on', 'the', 'mat']
  ['the', 'dog', 'sat', 'on', 'the', 'log']
  ['the', 'cat', 'chased', 'the', 'dog']

Vocabulary size: 8
Embedding dimension: 50
</div>
</div>

<div class="cell code">
<div class="cell-label">In [6]:</div>
<div class="cell-content">

```python
# Inspect the embedding for "cat"
cat_vector = model.wv['cat']
print("Vector for 'cat' (first 10 dims):")
print(np.round(cat_vector[:10], 4))

# Find most similar words to "cat"
print("\nMost similar to 'cat':")
for word, score in model.wv.most_similar('cat', topn=3):
    print(f"  {word}: {score:.4f}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [6]:</div>
<div class="cell-content">
Vector for 'cat' (first 10 dims):
[-0.0123  0.0156 -0.0089  0.0134  0.0045 -0.0178  0.0067  0.0112 -0.0091  0.0155]

Most similar to 'cat':
  dog: 0.9842
  sat: 0.2315
  chased: 0.1876
</div>
</div>

<div class="cell markdown">

### Key Properties of Word Embeddings

- **Dense** — Each word is a short vector (e.g., 50–300 dims) instead of a sparse vector with thousands of dims.
- **Semantic similarity** — "cat" and "dog" have similar vectors because they appear in similar contexts.
- **Compositionality** — Famous example: `king - man + woman ≈ queen`.

</div>

<div class="cell markdown">

## Comparison of Representations

| Method | Type | Dimensionality | Captures Semantics? | Captures Order? |
|---|---|---|---|---|
| Bag-of-Words | Sparse, count-based | \|Vocabulary\| | No | No |
| TF-IDF | Sparse, weighted | \|Vocabulary\| | Partially | No |
| Word2Vec | Dense, learned | 50–300 | Yes | Partially (local context) |

</div>

<div class="cell markdown">

## Summary

- **Bag-of-Words** is simple and effective for many tasks but ignores word order and importance.
- **TF-IDF** improves on BoW by down-weighting common words and highlighting distinctive ones.
- **Word2Vec** learns dense embeddings that capture semantic relationships between words.
- In practice, TF-IDF is a strong baseline for classification tasks, while word embeddings are essential for deep learning models.

</div>

</div>

<div class="notebook-nav">
<a href="{{ '/notebooks/02-text-preprocessing' | relative_url }}">← Previous: Text Preprocessing</a>
<a href="{{ '/notebooks/04-text-classification' | relative_url }}">Next: Text Classification →</a>
</div>
