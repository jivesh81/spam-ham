# spam-ham
# Spam Message Classifier using Neural Network

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using a Multi-Layer Perceptron (MLP) neural network with TF-IDF text features.

---

## Overview

This project trains a neural network on the SMS Spam Collection dataset to detect spam messages. It includes text preprocessing, model training, evaluation, and an interactive prediction interface where you can type your own messages and get instant results.

---

## Dataset

- **File:** `spam.csv`
- **Source:** SMS Spam Collection Dataset
- **Columns used:** `v1` (label: ham/spam), `v2` (message text)
- **Size:** ~5,572 messages

---

## Tech Stack

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading and manipulation |
| `nltk` | Stopword removal and stemming |
| `scikit-learn` | TF-IDF vectorization, MLP model, evaluation |
| `matplotlib` / `seaborn` | Visualizations |
| `joblib` | Saving and loading the model |

---

## How It Works

1. **Preprocessing** — Text is lowercased, punctuation removed, stop words filtered, and words stemmed using Porter Stemmer
2. **Vectorization** — TF-IDF converts text into numerical feature vectors (top 5000 features)
3. **Model** — MLPClassifier with two hidden layers (128 → 64 neurons), ReLU activation, Adam optimizer
4. **Evaluation** — Accuracy score, classification report, and confusion matrix heatmap
5. **Prediction** — Interactive loop lets you test any message in real time

---

## Model Architecture

```
Input (TF-IDF 5000 features)
        ↓
Hidden Layer 1 — 128 neurons, ReLU
        ↓
Hidden Layer 2 — 64 neurons, ReLU
        ↓
Output — Ham / Spam
```

---

## How to Run

### 1. Upload dataset to Google Drive
Place `spam.csv` inside `MyDrive/scam data/` folder.

### 2. Open in Google Colab
Upload or open the notebook in [Google Colab](https://colab.research.google.com).

### 3. Run all cells
The notebook will:
- Mount your Google Drive
- Load and preprocess the data
- Train the model
- Show evaluation results
- Save the model to Drive
- Start an interactive prediction loop

### 4. Test your own messages
After training, type any message when prompted:
```
Enter message: You won a free iPhone! Click now
→ SPAM  (Spam: 97.3% | Ham: 2.7%)

Enter message: Can we meet at 5pm tomorrow?
→ HAM   (Spam: 1.2% | Ham: 98.8%)
```
Type `quit` to exit.

---

## Output Files

After training, these files are saved to your Google Drive:

| File | Description |
|------|-------------|
| `spam_nn_model.pkl` | Trained MLP classifier |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer |

---

## Sample Results

```
Accuracy: ~97-98%

              precision    recall  f1-score
ham              0.98       0.99      0.99
spam             0.97       0.92      0.94
```

---

## Project Structure

```
├── spam_classifier.ipynb     # Main Colab notebook
├── README.md                 # This file
└── (Google Drive)
    ├── scam data/spam.csv    # Dataset
    ├── spam_nn_model.pkl     # Saved model
    └── vectorizer.pkl        # Saved vectorizer
```
