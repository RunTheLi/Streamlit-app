# 🧠 Sentiment Analysis App with Explainable AI (LIME + SHAP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Explainability](https://img.shields.io/badge/XAI-LIME%20%26%20SHAP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview

This project is an **interactive Machine Learning web application** built with **Streamlit** that performs sentiment analysis on text input and explains predictions using **LIME** and **SHAP**.

It transforms a traditional ML notebook into a **real-world AI product demo** with a modern dashboard-style UI.

---

## Features

- 🧠 Sentiment classification (Positive / Negative)
- 📊 Model explainability using:
  - LIME (Local Interpretability)
  - SHAP (Feature importance analysis)
- 📈 Confidence score display
- 🖥️ Dashboard-style UI with sidebar navigation
- ⚡ Fast predictions using Logistic Regression + TF-IDF
- 📉 Interactive visualizations (bar charts)

---

## Tech Stack

- Python 
- Scikit-learn 
- Streamlit 
- TF-IDF Vectorizer 
- Logistic Regression 
- LIME 
- SHAP 
- Matplotlib 

---

## Model Pipeline
Raw Text
↓
TF-IDF Vectorization
↓
Logistic Regression Model
↓
Prediction (Positive / Negative)
↓
Explainability Layer
├── LIME (local explanation)
└── SHAP (feature contribution)


---

## Screenshots

### Dashboard View
> Sentiment prediction with confidence score

![Dashboard](images/dashboard.png)

---

### LIME Explanation
> Shows word-level importance for prediction

![LIME](images/lime.png)

---

### SHAP Explanation
> Feature impact visualization

![SHAP](images/shap.png)

---

### Sidebar Navigation (SaaS-style UI)
> Multi-page dashboard structure

![Sidebar](images/sidebar.png)

---

## Installation

```bash
git clone https://github.com/your-username/sentiment-xai-app.git
cd sentiment-xai-app

pip install -r requirements.txt


