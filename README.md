# Multi-Class Fake News Detection

## Overview

This project classifies news articles into:

- True
- Fake
- Bias
- Satire

using Natural Language Processing (NLP) and Machine Learning.

---

## Dataset

Total Samples: 73,725+

Classes:

- True
- Fake
- Bias
- Satire

---

## NLP Techniques

- TF-IDF Vectorization
- Unigrams + Bigrams
- Text Feature Engineering

---

## Machine Learning

Model:
- Logistic Regression

Hyperparameter Tuning:
- GridSearchCV

Best Parameters:

- C = 10
- solver = saga
- class_weight = balanced

---

## Performance

Accuracy: 85.58%

Macro F1 Score: ~0.85

---

## Deployment

Built using:

- Streamlit

Features:

- Real-time prediction
- Confidence score
- Multi-class classification
- Probability distribution display

---

## Project Structure

MULTI CLASS FAKE NEWS DETECTOR PROJECT

├── app.py

├── requirements.txt

├── data/

├── models/

├── notebooks/

└── README.md

---

Future Improvements

- RoBERTa Fine-Tuning
- Explainable AI
- News Source Analysis
- API Deployment

## Author

Nishant Saubhari

Aspiring Machine Learning & NLP Engineer
