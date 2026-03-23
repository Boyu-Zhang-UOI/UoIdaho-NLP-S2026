---
layout: default
title: "Section 2: Text Preprocessing"
---

# Section 2 — Text Preprocessing

<div class="notebook">

<div class="cell markdown">

## Why Preprocess Text?

Raw text is noisy — it contains punctuation, inconsistent casing, irrelevant words, and varying word forms. **Preprocessing** transforms raw text into a cleaner, more uniform representation that is easier for models to learn from.

A typical preprocessing pipeline looks like this:

```
Raw Text
  → Lowercasing
    → Tokenization
      → Stopword Removal
        → Stemming / Lemmatization
          → Clean Tokens
```

</div>

<div class="cell code">
<div class="cell-label">In [1]:</div>
<div class="cell-content">

```python
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

sample_text = """
Natural Language Processing (NLP) is a sub-field of Artificial Intelligence.
It focuses on the interaction between computers and human language!
NLP techniques are used in chatbots, search engines, and translation systems.
"""

print("Raw text:")
print(sample_text)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [1]:</div>
<div class="cell-content">
Raw text:

Natural Language Processing (NLP) is a sub-field of Artificial Intelligence.
It focuses on the interaction between computers and human language!
NLP techniques are used in chatbots, search engines, and translation systems.

</div>
</div>

<div class="cell markdown">

## Step 1: Lowercasing

Converting all text to lowercase ensures that "NLP", "nlp", and "Nlp" are treated as the same token.

</div>

<div class="cell code">
<div class="cell-label">In [2]:</div>
<div class="cell-content">

```python
text_lower = sample_text.lower()
print(text_lower)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [2]:</div>
<div class="cell-content">

natural language processing (nlp) is a sub-field of artificial intelligence.
it focuses on the interaction between computers and human language!
nlp techniques are used in chatbots, search engines, and translation systems.

</div>
</div>

<div class="cell markdown">

## Step 2: Tokenization

**Tokenization** splits text into individual units (tokens). There are two common levels:

- **Sentence tokenization** — split text into sentences.
- **Word tokenization** — split text into words.

</div>

<div class="cell code">
<div class="cell-label">In [3]:</div>
<div class="cell-content">

```python
# Sentence tokenization
sentences = sent_tokenize(text_lower)
print("Sentences:")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

# Word tokenization
tokens = word_tokenize(text_lower)
print(f"\nWord tokens ({len(tokens)} total):")
print(tokens)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [3]:</div>
<div class="cell-content">
Sentences:
  [0] natural language processing (nlp) is a sub-field of artificial intelligence.
  [1] it focuses on the interaction between computers and human language!
  [2] nlp techniques are used in chatbots, search engines, and translation systems.

Word tokens (38 total):
['natural', 'language', 'processing', '(', 'nlp', ')', 'is', 'a', 'sub-field', 'of', 'artificial', 'intelligence', '.', 'it', 'focuses', 'on', 'the', 'interaction', 'between', 'computers', 'and', 'human', 'language', '!', 'nlp', 'techniques', 'are', 'used', 'in', 'chatbots', ',', 'search', 'engines', ',', 'and', 'translation', 'systems', '.']
</div>
</div>

<div class="cell markdown">

## Step 3: Removing Punctuation and Special Characters

Punctuation tokens usually do not carry useful information for most NLP tasks.

</div>

<div class="cell code">
<div class="cell-label">In [4]:</div>
<div class="cell-content">

```python
# Keep only alphabetic tokens
clean_tokens = [token for token in tokens if token.isalpha()]

print(f"After removing punctuation ({len(clean_tokens)} tokens):")
print(clean_tokens)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [4]:</div>
<div class="cell-content">
After removing punctuation (30 tokens):
['natural', 'language', 'processing', 'nlp', 'is', 'a', 'of', 'artificial', 'intelligence', 'it', 'focuses', 'on', 'the', 'interaction', 'between', 'computers', 'and', 'human', 'language', 'nlp', 'techniques', 'are', 'used', 'in', 'chatbots', 'search', 'engines', 'and', 'translation', 'systems']
</div>
</div>

<div class="cell markdown">

**Note:** `isalpha()` also removes hyphenated words like "sub-field" because the hyphen is not an alphabetic character. For tasks where compound words matter, you may want a more nuanced filter (e.g., replacing hyphens before filtering).

## Step 4: Stopword Removal

**Stopwords** are high-frequency words (e.g., "the", "is", "and") that carry little semantic meaning. Removing them reduces noise and dimensionality.

</div>

<div class="cell code">
<div class="cell-label">In [5]:</div>
<div class="cell-content">

```python
stop_words = set(stopwords.words('english'))

print("Sample stopwords:", list(stop_words)[:10])
print()

filtered_tokens = [t for t in clean_tokens if t not in stop_words]

removed = [t for t in clean_tokens if t in stop_words]
print(f"Removed stopwords: {removed}")
print(f"\nFiltered tokens ({len(filtered_tokens)}):")
print(filtered_tokens)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [5]:</div>
<div class="cell-content">
Sample stopwords: ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during']

Removed stopwords: ['is', 'a', 'of', 'it', 'on', 'the', 'between', 'and', 'are', 'in', 'and']

Filtered tokens (19):
['natural', 'language', 'processing', 'nlp', 'artificial', 'intelligence', 'focuses', 'interaction', 'computers', 'human', 'language', 'nlp', 'techniques', 'used', 'chatbots', 'search', 'engines', 'translation', 'systems']
</div>
</div>

<div class="cell markdown">

## Step 5: Stemming

**Stemming** reduces words to their root form by chopping off suffixes. It is fast but can be aggressive — the result is not always a real word.

</div>

<div class="cell code">
<div class="cell-label">In [6]:</div>
<div class="cell-content">

```python
stemmer = PorterStemmer()

examples = ['running', 'runs', 'ran', 'easily', 'computers', 'processing']
print("Stemming examples:")
for word in examples:
    print(f"  {word:15s} → {stemmer.stem(word)}")

print("\nStemmed tokens:")
stemmed = [stemmer.stem(t) for t in filtered_tokens]
print(stemmed)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [6]:</div>
<div class="cell-content">
Stemming examples:
  running         → run
  runs            → run
  ran             → ran
  easily          → easili
  computers       → comput
  processing      → process

Stemmed tokens:
['natur', 'languag', 'process', 'nlp', 'artifici', 'intellig', 'focus', 'interact', 'comput', 'human', 'languag', 'nlp', 'techniqu', 'use', 'chatbot', 'search', 'engin', 'translat', 'system']
</div>
</div>

<div class="cell markdown">

## Step 6: Lemmatization

**Lemmatization** reduces words to their dictionary form (lemma). It is slower than stemming but produces actual words.

</div>

<div class="cell code">
<div class="cell-label">In [7]:</div>
<div class="cell-content">

```python
lemmatizer = WordNetLemmatizer()

examples = ['running', 'runs', 'ran', 'better', 'computers', 'processing']
print("Lemmatization examples (as verbs):")
for word in examples:
    print(f"  {word:15s} → {lemmatizer.lemmatize(word, pos='v')}")

print("\nLemmatized tokens:")
lemmatized = [lemmatizer.lemmatize(t) for t in filtered_tokens]
print(lemmatized)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [7]:</div>
<div class="cell-content">
Lemmatization examples (as verbs):
  running         → run
  runs            → run
  ran             → run
  better          → better
  computers       → computer
  processing      → process

Lemmatized tokens:
['natural', 'language', 'processing', 'nlp', 'artificial', 'intelligence', 'focus', 'interaction', 'computer', 'human', 'language', 'nlp', 'technique', 'used', 'chatbots', 'search', 'engine', 'translation', 'system']
</div>
</div>

<div class="cell markdown">

## Stemming vs. Lemmatization

| Feature | Stemming | Lemmatization |
|---|---|---|
| Speed | Fast | Slower |
| Output | May not be a real word ("easili") | Always a real word ("easily" → "easy") |
| Uses dictionary? | No | Yes (WordNet) |
| Use case | Search engines, IR | Text classification, chatbots |

**Rule of thumb:** Use lemmatization when you need interpretable output; use stemming when speed matters and exact forms do not.

</div>

<div class="cell markdown">

## Putting It All Together: A Complete Pipeline

</div>

<div class="cell code">
<div class="cell-label">In [8]:</div>
<div class="cell-content">

```python
def preprocess(text):
    """Full text preprocessing pipeline."""
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [t for t in tokens if t.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Test the pipeline
raw = "The quick brown foxes are jumping over the lazy dogs!"
result = preprocess(raw)
print(f"Input:  {raw}")
print(f"Output: {result}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [8]:</div>
<div class="cell-content">
Input:  The quick brown foxes are jumping over the lazy dogs!
Output: ['quick', 'brown', 'fox', 'jumping', 'lazy', 'dog']
</div>
</div>

<div class="cell markdown">

## Regular Expressions for Text Cleaning

**Regular expressions (regex)** are powerful for pattern-based text cleaning. Here are common patterns:

</div>

<div class="cell code">
<div class="cell-label">In [9]:</div>
<div class="cell-content">

```python
import re

messy = "Contact us at info@example.com or visit https://example.com! Call 555-1234."

# Remove emails
no_emails = re.sub(r'\S+@\S+', '<EMAIL>', messy)
print("Emails masked:", no_emails)

# Remove URLs
no_urls = re.sub(r'https?://\S+', '<URL>', no_emails)
print("URLs masked:  ", no_urls)

# Remove digits
no_digits = re.sub(r'\d+', '<NUM>', no_urls)
print("Digits masked:", no_digits)

# Keep only words
words_only = re.findall(r'[a-zA-Z]+', messy)
print("Words only:   ", words_only)
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [9]:</div>
<div class="cell-content">
Emails masked: Contact us at <EMAIL> or visit https://example.com! Call 555-1234.
URLs masked:   Contact us at <EMAIL> or visit <URL>! Call 555-1234.
Digits masked: Contact us at <EMAIL> or visit <URL>! Call <NUM>-<NUM>.
Words only:    ['Contact', 'us', 'at', 'info', 'example', 'com', 'or', 'visit', 'https', 'example', 'com', 'Call']
</div>
</div>

<div class="cell markdown">

## Summary

| Step | Purpose | Tool |
|---|---|---|
| Lowercasing | Normalize casing | `str.lower()` |
| Tokenization | Split into tokens | `nltk.word_tokenize()` |
| Punctuation removal | Remove non-informative symbols | `str.isalpha()` |
| Stopword removal | Remove common words | `nltk.corpus.stopwords` |
| Stemming | Reduce to root form (fast) | `PorterStemmer` |
| Lemmatization | Reduce to dictionary form | `WordNetLemmatizer` |
| Regex cleaning | Pattern-based substitution | `re.sub()` |

The preprocessing steps you choose depend on your task. For search engines, stemming may be enough. For chatbots, you might skip stopword removal to preserve natural phrasing.

</div>

</div>

<div class="notebook-nav">
<a href="{{ '/notebooks/01-introduction' | relative_url }}">← Previous: Introduction</a>
<a href="{{ '/notebooks/03-text-representation' | relative_url }}">Next: Text Representation →</a>
</div>
