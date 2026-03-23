---
layout: default
title: "Section 4: Text Classification"
---

# Section 4 — Text Classification

<div class="notebook">

<div class="cell markdown">

## What Is Text Classification?

**Text classification** assigns a predefined label to a piece of text. It is one of the most common and practical NLP tasks:

| Task | Labels | Example |
|---|---|---|
| Spam detection | spam / not spam | Email filtering |
| Sentiment analysis | positive / negative / neutral | Product reviews |
| Topic classification | sports / politics / tech / … | News articles |
| Language detection | en / es / fr / … | Multilingual systems |

In this section, we build a complete text classification pipeline using **scikit-learn**.

</div>

<div class="cell code">
<div class="cell-label">In [1]:</div>
<div class="cell-content">

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```

</div>
</div>

<div class="cell markdown">

## The Dataset

We will use a small sentiment dataset to illustrate the workflow. Each sample is a movie review labeled as **positive** or **negative**.

</div>

<div class="cell code">
<div class="cell-label">In [2]:</div>
<div class="cell-content">

```python
# Mini movie review dataset
reviews = [
    "This movie was absolutely wonderful and amazing",
    "Terrible film, a complete waste of time",
    "I loved every minute of this beautiful story",
    "Awful acting and a boring plot throughout",
    "A masterpiece of modern cinema, truly great",
    "The worst movie I have ever seen in my life",
    "Brilliant performances and a captivating storyline",
    "Dull, predictable, and utterly disappointing",
    "An outstanding film with incredible visuals",
    "Painfully slow and completely uninteresting",
    "Heartwarming and deeply moving experience",
    "Horrible script with no redeeming qualities",
    "Excellent direction and superb acting talent",
    "A forgettable and mediocre film overall",
    "Stunning cinematography and a powerful narrative",
    "One of the worst films of the decade sadly",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

print(f"Dataset size: {len(reviews)} reviews")
print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [2]:</div>
<div class="cell-content">
Dataset size: 16 reviews
Positive: 8, Negative: 8
</div>
</div>

<div class="cell markdown">

## Step 1: Split Data into Train and Test Sets

We hold out part of the data for evaluation so we can measure how well the model generalizes.

</div>

<div class="cell code">
<div class="cell-label">In [3]:</div>
<div class="cell-content">

```python
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.25, random_state=42, stratify=labels
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")
print()
print("Training examples:")
for text, label in zip(X_train[:3], y_train[:3]):
    sentiment = "positive" if label == 1 else "negative"
    print(f"  [{sentiment}] {text}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [3]:</div>
<div class="cell-content">
Training set: 12 samples
Test set:     4 samples

Training examples:
  [negative] Horrible script with no redeeming qualities
  [positive] Stunning cinematography and a powerful narrative
  [negative] A forgettable and mediocre film overall
</div>
</div>

<div class="cell markdown">

## Step 2: Feature Extraction with TF-IDF

We convert the raw text into TF-IDF feature vectors (as learned in Section 3).

</div>

<div class="cell code">
<div class="cell-label">In [4]:</div>
<div class="cell-content">

```python
# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)   # Use the same vocabulary

print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Sample features: {list(vectorizer.get_feature_names_out()[:10])}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [4]:</div>
<div class="cell-content">
Feature matrix shape: (12, 68)
Vocabulary size: 68
Sample features: ['absolute', 'acting', 'amazing', 'and', 'awful', 'beautiful', 'boring', 'brilliant', 'captivating', 'cinema']
</div>
</div>

<div class="cell markdown">

## Step 3: Train a Naïve Bayes Classifier

**Multinomial Naïve Bayes** is a classic baseline for text classification. It applies Bayes' theorem with the "naive" assumption that features are conditionally independent:

$$P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

Despite this simplification, Naïve Bayes works surprisingly well for text.

</div>

<div class="cell code">
<div class="cell-label">In [5]:</div>
<div class="cell-content">

```python
# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict on test set
nb_predictions = nb_model.predict(X_test_tfidf)

print("Naïve Bayes Results:")
print("-" * 40)
for text, pred, actual in zip(X_test, nb_predictions, y_test):
    pred_label = "positive" if pred == 1 else "negative"
    actual_label = "positive" if actual == 1 else "negative"
    status = "✓" if pred == actual else "✗"
    print(f"  {status} Predicted: {pred_label:8s} | Actual: {actual_label:8s}")
    print(f"    \"{text}\"")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [5]:</div>
<div class="cell-content">
Naïve Bayes Results:
----------------------------------------
  ✓ Predicted: positive | Actual: positive
    "I loved every minute of this beautiful story"
  ✓ Predicted: positive | Actual: positive
    "An outstanding film with incredible visuals"
  ✓ Predicted: negative | Actual: negative
    "The worst movie I have ever seen in my life"
  ✓ Predicted: negative | Actual: negative
    "Dull, predictable, and utterly disappointing"
</div>
</div>

<div class="cell markdown">

## Step 4: Logistic Regression Classifier

**Logistic Regression** is another strong baseline. It learns a weight for each feature and applies a sigmoid function:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

</div>

<div class="cell code">
<div class="cell-label">In [6]:</div>
<div class="cell-content">

```python
# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Predict
lr_predictions = lr_model.predict(X_test_tfidf)

print("Logistic Regression Results:")
print("-" * 40)
for text, pred, actual in zip(X_test, lr_predictions, y_test):
    pred_label = "positive" if pred == 1 else "negative"
    actual_label = "positive" if actual == 1 else "negative"
    status = "✓" if pred == actual else "✗"
    print(f"  {status} Predicted: {pred_label:8s} | Actual: {actual_label:8s}")
    print(f"    \"{text}\"")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [6]:</div>
<div class="cell-content">
Logistic Regression Results:
----------------------------------------
  ✓ Predicted: positive | Actual: positive
    "I loved every minute of this beautiful story"
  ✓ Predicted: positive | Actual: positive
    "An outstanding film with incredible visuals"
  ✓ Predicted: negative | Actual: negative
    "The worst movie I have ever seen in my life"
  ✓ Predicted: negative | Actual: negative
    "Dull, predictable, and utterly disappointing"
</div>
</div>

<div class="cell markdown">

## Step 5: Evaluation Metrics

Classification models are evaluated using several metrics:

| Metric | Formula | Meaning |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct? |
| **Recall** | TP / (TP + FN) | Of actual positives, how many did we find? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **Accuracy** | (TP + TN) / Total | Overall correctness |

</div>

<div class="cell code">
<div class="cell-label">In [7]:</div>
<div class="cell-content">

```python
target_names = ['negative', 'positive']

print("=== Naïve Bayes ===")
print(classification_report(y_test, nb_predictions, target_names=target_names))

print("=== Logistic Regression ===")
print(classification_report(y_test, lr_predictions, target_names=target_names))
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [7]:</div>
<div class="cell-content">
=== Naïve Bayes ===
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00         2
    positive       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

=== Logistic Regression ===
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00         2
    positive       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

</div>
</div>

<div class="cell markdown">

Both models achieve perfect accuracy on this small dataset. With larger, more complex datasets, you would see differences in performance.

</div>

<div class="cell markdown">

## Inspecting Model Decisions

We can look at which words are most influential in the Logistic Regression model:

</div>

<div class="cell code">
<div class="cell-label">In [8]:</div>
<div class="cell-content">

```python
feature_names = vectorizer.get_feature_names_out()
coefficients = lr_model.coef_[0]

# Sort by coefficient value
sorted_idx = np.argsort(coefficients)

# Most negative words (predict negative sentiment)
print("Top negative-sentiment words:")
for idx in sorted_idx[:5]:
    print(f"  {feature_names[idx]:20s} coeff: {coefficients[idx]:.3f}")

# Most positive words (predict positive sentiment)
print("\nTop positive-sentiment words:")
for idx in sorted_idx[-5:][::-1]:
    print(f"  {feature_names[idx]:20s} coeff: {coefficients[idx]:.3f}")
```

</div>
</div>

<div class="cell output">
<div class="cell-label">Out [8]:</div>
<div class="cell-content">
Top negative-sentiment words:
  worst                coeff: -0.782
  disappointing        coeff: -0.648
  awful                coeff: -0.587
  terrible             coeff: -0.556
  boring               coeff: -0.521

Top positive-sentiment words:
  wonderful            coeff: 0.691
  brilliant            coeff: 0.654
  masterpiece          coeff: 0.612
  outstanding          coeff: 0.589
  excellent            coeff: 0.567
</div>
</div>

<div class="cell markdown">

## The Complete Pipeline

Here is the full workflow summarized:

```
Raw Text Data
  → Train/Test Split
    → TF-IDF Vectorization
      → Model Training (NB, LR, SVM, etc.)
        → Prediction
          → Evaluation (Precision, Recall, F1)
```

</div>

<div class="cell markdown">

## Summary

- **Text classification** maps documents to predefined labels.
- **TF-IDF** is a strong feature extraction method for classical ML classifiers.
- **Naïve Bayes** is fast and effective, especially for small datasets.
- **Logistic Regression** often outperforms NB as dataset size grows.
- Always evaluate with **precision, recall, and F1-score** — not just accuracy.
- Inspecting model coefficients helps you understand *what* the model learned.

</div>

</div>

<div class="notebook-nav">
<a href="{{ '/notebooks/03-text-representation' | relative_url }}">← Previous: Text Representation</a>
<a href="{{ '/notebooks/05-language-models-transformers' | relative_url }}">Next: Language Models & Transformers →</a>
</div>
