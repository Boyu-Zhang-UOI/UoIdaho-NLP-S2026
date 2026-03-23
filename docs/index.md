---
layout: default
title: Home
---

# Natural Language Processing — Spring 2026

**University of Idaho · Department of Computer Science**

Welcome to the course materials for **CS 4/574: Natural Language Processing**. This site contains lecture notebooks covering the essential topics in NLP, from basic text processing to modern transformer architectures. Each section is organized in a Jupyter Notebook style with explanations and runnable Python code.

---

## Course Sections

<div class="section-cards">

<div class="section-card">
<h3>📖 Section 1</h3>
<p>Introduction to NLP — What NLP is, why it matters, core tasks, and setting up your Python environment.</p>
<a href="{{ '/notebooks/01-introduction' | relative_url }}">Open Notebook →</a>
</div>

<div class="section-card">
<h3>🔧 Section 2</h3>
<p>Text Preprocessing — Tokenization, stemming, lemmatization, stopword removal, and building a preprocessing pipeline.</p>
<a href="{{ '/notebooks/02-text-preprocessing' | relative_url }}">Open Notebook →</a>
</div>

<div class="section-card">
<h3>📊 Section 3</h3>
<p>Text Representation — Bag-of-Words, TF-IDF, and Word2Vec embeddings for converting text to numbers.</p>
<a href="{{ '/notebooks/03-text-representation' | relative_url }}">Open Notebook →</a>
</div>

<div class="section-card">
<h3>🏷️ Section 4</h3>
<p>Text Classification — Building classifiers with Naïve Bayes, logistic regression, and evaluation metrics.</p>
<a href="{{ '/notebooks/04-text-classification' | relative_url }}">Open Notebook →</a>
</div>

<div class="section-card">
<h3>🤖 Section 5</h3>
<p>Language Models &amp; Transformers — N-grams, neural language models, attention, and the Transformer architecture.</p>
<a href="{{ '/notebooks/05-language-models-transformers' | relative_url }}">Open Notebook →</a>
</div>

</div>

---

## How to Use These Materials

Each section is presented as a **notebook-style page** containing:

- **Markdown cells** (blue border) — explanations, theory, and context.
- **Code cells** (labeled `In [n]`) — Python code you can copy and run.
- **Output cells** (green border) — expected output from the code.

You can follow along by running the code in your own Jupyter Notebook or Google Colab environment.

## Prerequisites

- Python 3.9 or later
- Familiarity with basic Python programming
- Recommended libraries: `nltk`, `scikit-learn`, `numpy`, `gensim`
