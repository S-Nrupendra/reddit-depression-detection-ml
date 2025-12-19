# Reddit Depression Detection using Machine Learning (NLP)

## 📌 Project Overview

This project focuses on detecting signs of depression from Reddit posts using **classical Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.

The goal is to build a **scalable, interpretable, and efficient** text classification pipeline without using deep learning or transformer-based models.

---

## 📂 Dataset

* Source: Reddit posts related to mental health
* Fields used:

  * `title`
  * `body`
  * `label` (0 → Normal, 1 → Depressed)

### Text Selection Strategy

* If `body` is present → use `body`
* If `body` is missing → use `title`
* If both are missing → drop the row

This ensures meaningful text while avoiding redundancy.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

1. Removing empty, duplicate, and very short posts
2. Lowercasing text
3. Removing URLs and special characters
4. Removing stopwords
5. Lemmatization
6. Dropping rows that become empty after cleaning

The cleaned dataset is saved separately to avoid repeated preprocessing.

---

Nice — this section is already good. I’ll **rewrite it cleanly and professionally**, while **implicitly addressing all your doubts**:

* why unigrams still work
* why bigrams help only a little
* why more features don’t help much
* why recall-focused stability matters
* why `min_df` and `sublinear_tf` are important

I’ll keep it **README-ready**, **interview-safe**, and **technically correct**.

---

## ✨ Feature Engineering (TF-IDF)

We use **TF-IDF vectorization** to convert Reddit posts into numerical features that can be consumed by machine learning models.

The goal is to capture **both individual depressive words and meaningful short phrases**, while keeping the representation **memory-efficient and generalizable**.

---

### 🔬 Experiments Conducted

| Configuration             | Features | Observation          |
| ------------------------- | -------- | -------------------- |
| TF-IDF (Unigram)          | 10,000   | Strong baseline      |
| TF-IDF (Unigram + Bigram) | 15,000   | Slight improvement   |
| TF-IDF (Unigram + Bigram) | 20,000   | Not used (see below) |

---

### ⚙️ Final TF-IDF Configuration

* **Unigrams + Bigrams** (`ngram_range = (1,2)`)
* **Sublinear TF scaling** (`sublinear_tf = True`)
* **Minimum document frequency** (`min_df = 5`)
* **Maximum features**: **15,000**

---

### 🧠 Why Unigrams Still Work Well

Depressive language often contains **strong individual words** such as:
> *empty, hopeless, lonely, worthless*
These words alone carry high predictive power, which is why a unigram-only model already performs well.

---

### 🧠 Why Bigrams Help (but only slightly)

Bigrams capture **contextual phrases** such as:
> *feel empty*, *not happy*, *no motivation*
Adding bigrams improves the model’s understanding of **negation and emotional context**, but:

* Most core depressive signals are already captured by unigrams
* Therefore, the improvement is **incremental rather than dramatic**

---

### 🚫 Why We Stopped at 15,000 Features

Increasing features beyond 15k showed **diminishing returns**:

* Most new features are:

  * Rare words
  * User-specific language
  * Noise
* Memory usage increases significantly
* Training time increases
* Risk of overfitting increases
* Model performance **plateaus** after top features are captured

Additionally:

* Words appearing in **fewer than 5 documents** are ignored (`min_df = 5`)
* This removes typos, usernames, and extremely rare expressions
* Improves generalization and stability

---

### 🔍 Why `sublinear_tf` Matters

Instead of treating repeated words as linearly more important, we apply **logarithmic scaling**:
> Repeating an emotional word many times does not proportionally increase its meaning.
This prevents long, repetitive posts from dominating the model and improves robustness.
---

## 🤖 Models Used

### 1. Logistic Regression (Primary Model)

* Chosen for:

  * Interpretability
  * Strong performance on sparse TF-IDF features
  * Fast training on large datasets

### 2. Multinomial Naive Bayes (Baseline)

Used for comparison with Logistic Regression.

---

## 📊 Model Evaluation

### Logistic Regression (Final Model)

```
Accuracy: 92%

Class 0 (Normal):
Precision: 0.97
Recall:    0.93
F1-score:  0.95

Class 1 (Depressed):
Precision: 0.76
Recall:    0.90
F1-score:  0.83
```

### Why Recall Matters More Here
Recall : TP / (TP + FN)
Out of all the actually depressed posts, how many did the model successfully catch?

For binary classification :
|                          | Predicted Depressed (1) | Predicted Normal (0)    |
| ------------------------ | ----------------------- | ----------------------- |
| **Actual Depressed (1)** | **TP** (True Positive)  | **FN** (False Negative) |
| **Actual Normal (0)**    | **FP** (False Positive) | **TN** (True Negative)  |

* TP → Depressed post correctly detected
* FN → Depressed post predicted as normal ❌ (dangerous)
* FP → Normal post predicted as depressed
* TN → Normal post correctly detected

In depression detection:

* **False negatives are more harmful than false positives**
* High recall ensures most depressed posts are correctly identified

Hence, Logistic Regression was selected as the final model.

---

## 🧠 Final Pipeline

1. Raw Reddit data
2. Text selection (body → title fallback)
3. Cleaning & normalization
4. TF-IDF vectorization (15k features)
5. Logistic Regression training
6. Model evaluation
7. Model & vectorizer saved using `.pkl`
8. Lightweight inference using `app.py`

---

## 🚀 How to Run

### Training

Run notebooks in order:

```
01_data_loading.ipynb
02_preprocessing.ipynb
03_feature_engineering.ipynb
04_model_training.ipynb
```

### Inference

```bash
python app/app.py
```

---

## 📦 Project Structure

```
reddit-depression-detection-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│
├── models/
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│
├── app/
│   └── app.py
│
└── README.md
```

---

## 🔮 Future Improvements

* Add bigrams/trigrams selectively
* Hyperparameter tuning (`C`, `class_weight`)
* Threshold tuning for recall–precision balance
* Transformer-based models (BERT, RoBERTa)
* Deployment using FastAPI or Streamlit

---

## ✅ Conclusion

This project demonstrates that **classical NLP + ML**, when done correctly, can achieve **strong, reliable results** on large real-world datasets.

Rather than blindly increasing complexity, we stopped at the point of **diminishing returns**, prioritizing:

* Performance
* Scalability
* Interpretability

This reflects **real-world ML engineering practice**.