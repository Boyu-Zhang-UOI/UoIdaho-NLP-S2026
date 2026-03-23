---
layout: default
title: "Section 1: Introduction to NLP"
---

# Section 1 — Introduction to Natural Language Processing

<div class="notebook">

<div class="cell markdown">

## What Is Natural Language Processing?

**Natural Language Processing (NLP)** is a field at the intersection of computer science, artificial intelligence, and linguistics. Its goal is to enable computers to understand, interpret, and generate human language.

NLP powers many everyday technologies:

| Application | Example |
|---|---|
| Machine Translation | Google Translate |
| Virtual Assistants | Siri, Alexa, ChatGPT |
| Spam Detection | Email spam filters |
| Sentiment Analysis | Product review analysis |
| Text Summarization | News article summaries |
| Autocomplete | Smartphone keyboards |

</div>

<div class="cell markdown">

## Why Is NLP Hard?

Human language is inherently **ambiguous** and **context-dependent**. Consider:

- *"I saw the man with the telescope."* — Who has the telescope?
- *"Bank"* can mean a financial institution or the side of a river.
- *"It's cold"* might be about weather or about someone's personality.

NLP systems must handle **lexical ambiguity** (word-level), **syntactic ambiguity** (structural), and **pragmatic ambiguity** (contextual meaning).

</div>

<div class="cell markdown">

## Core NLP Tasks

NLP encompasses many sub-tasks. Here are the most fundamental ones:

1. **Tokenization** — Splitting text into words or sub-words.
2. **Part-of-Speech (POS) Tagging** — Labeling each word with its grammatical role (noun, verb, etc.).
3. **Named Entity Recognition (NER)** — Identifying proper nouns like people, places, organizations.
4. **Parsing** — Analyzing sentence structure (syntax trees).
5. **Text Classification** — Assigning categories to documents.
6. **Machine Translation** — Translating text between languages.
7. **Question Answering** — Answering questions from text passages.

</div>

<div class="cell markdown">

## Setting Up the Python Environment

Let us start by installing and importing the libraries we will use throughout this course.

</div>

<div class="cell code">
<div class="cell-label">In [1]:</div>
<div class="cell-content">

```python
# Install core NLP libraries (run once)
# !pip install nltk scikit-learn numpy gensim

import nltk
import numpy as np
from pprint import pprint

# Download essential NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

print("Environment is ready!")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [1]:</div>
<div class="cell-content">
Environment is ready!
</div>
</div>

<div class="cell markdown">

## Your First NLP Pipeline: Tokenize → Tag → Recognize

Let us run a quick end-to-end example that demonstrates three core NLP operations in sequence.

</div>

<div class="cell code">
<div class="cell-label">In [2]:</div>
<div class="cell-content">

```python
from nltk import word_tokenize, pos_tag, ne_chunk

text = "Barack Obama served as the 44th President of the United States."

# Step 1: Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Step 2: POS Tagging
tagged = pos_tag(tokens)
print("\nPOS Tags:", tagged)

# Step 3: Named Entity Recognition
tree = ne_chunk(tagged)
print("\nNamed Entities:")
for subtree in tree:
    if hasattr(subtree, 'label'):
        entity = " ".join(word for word, tag in subtree)
        print(f"  {subtree.label()}: {entity}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [2]:</div>
<div class="cell-content">
Tokens: ['Barack', 'Obama', 'served', 'as', 'the', '44th', 'President', 'of', 'the', 'United', 'States', '.']

POS Tags: [('Barack', 'NNP'), ('Obama', 'NNP'), ('served', 'VBD'), ('as', 'IN'), ('the', 'DT'), ('44th', 'JJ'), ('President', 'NNP'), ('of', 'IN'), ('the', 'DT'), ('United', 'NNP'), ('States', 'NNPS'), ('.', '.')]

Named Entities:
  PERSON: Barack Obama
  GPE: United States
</div>
</div>

<div class="cell markdown">

## Understanding the Output

| Step | What It Does | Example |
|---|---|---|
| **Tokenization** | Splits text into individual words/punctuation | `"Barack Obama served..."` → `['Barack', 'Obama', 'served', ...]` |
| **POS Tagging** | Labels each token with its part of speech | `('Obama', 'NNP')` — NNP = proper noun, singular |
| **NER** | Groups tokens into named entities | `PERSON: Barack Obama`, `GPE: United States` |

Common POS tags: `NNP` = proper noun, `VBD` = past-tense verb, `DT` = determiner, `IN` = preposition, `JJ` = adjective.

</div>

<div class="cell markdown">

## Levels of Linguistic Analysis

NLP systems can analyze language at multiple levels:

```
Phonetics/Phonology   →  sounds
Morphology            →  word structure (prefixes, suffixes)
Syntax                →  sentence structure (grammar)
Semantics             →  meaning of words and sentences
Pragmatics            →  meaning in context
Discourse             →  meaning across sentences
```

In this course, we focus primarily on the **morphology → semantics** range, using computational tools.

</div>

<div class="cell markdown">

## Rule-Based vs. Statistical vs. Neural NLP

The field has evolved through three major paradigms:

| Era | Approach | Example |
|---|---|---|
| 1950s–1980s | **Rule-based** | Hand-written grammar rules |
| 1990s–2010s | **Statistical** | Hidden Markov Models, Naïve Bayes |
| 2010s–present | **Neural / Deep Learning** | RNNs, Transformers, BERT, GPT |

Modern NLP is dominated by **neural** approaches, but understanding statistical fundamentals is essential for building intuition.

</div>

<div class="cell markdown">

## Summary

In this section we covered:

- **What NLP is** and why it matters.
- **Why language is hard** for computers (ambiguity, context).
- **Core NLP tasks** (tokenization, POS tagging, NER, classification, etc.).
- A **hands-on pipeline** combining tokenization, POS tagging, and NER.
- The **evolution** from rule-based to neural NLP.

In the next section, we will dive deeper into **text preprocessing** — the essential first step in any NLP pipeline.

</div>

</div>

<div class="notebook-nav">
<span></span>
<a href="{{ '/notebooks/02-text-preprocessing' | relative_url }}">Next: Text Preprocessing →</a>
</div>
