# 📧 Email Spam Detection using NLP and LSTM

This project uses deep learning and Natural Language Processing (NLP) techniques to classify SMS messages as either **spam** or **ham (not spam)**. The core model is built using an LSTM (Long Short-Term Memory) network, delivering high accuracy through sequential learning of message patterns.

---

## 🚀 Overview

- **Objective**: Automatically detect spam messages in text using machine learning.
- **Model**: LSTM-based neural network for sequential pattern recognition.
- **Dataset**: SMS Spam Collection Dataset with 5,572 labeled messages.

---

## 📂 Dataset Information

- **Source**: `spam.csv` from the UCI Machine Learning Repository
- **Features**:
  - `Category`: Message label (ham/spam)
  - `Message`: Text content of the message

- **Class Distribution**:
  - Ham: 4,825
  - Spam: 747

To handle class imbalance, the dataset was **oversampled** using duplication techniques.

---

## 🧠 Model & Methods

### 📌 Preprocessing
- Lowercased all text
- Removed stopwords and punctuation
- Tokenized and lemmatized using NLTK
- Engineered features:
  - Message length
  - Sentiment polarity (TextBlob)

### 🔢 Tokenization
- Tokenized words using `Tokenizer` from Keras
- Padded sequences to ensure consistent input size

### 🔮 Model Architecture
- **Embedding Layer**
- **SpatialDropout1D**
- **LSTM Layer**
- **Dense Layer with Sigmoid Activation**

### ⚙️ Training Details
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Epochs**: 15
- **Batch Size**: 32
- **Validation Split**: 20%

---

## 📈 Evaluation

| Metric     | Ham     | Spam    |
|------------|---------|---------|
| Precision  | 1.00    | 0.99    |
| Recall     | 0.99    | 1.00    |
| F1-score   | 1.00    | 1.00    |

- **Overall Accuracy**: 99.51%
- **Loss**: ~0.03

LSTM model shows strong generalization and **no signs of overfitting**.

---

## 🤖 Other Models Compared

| Model           | Accuracy |
|----------------|----------|
| LSTM (Deep Learning) | **99.51%** |
| Random Forest        | 97.75%     |
| Naive Bayes          | 79.38%     |

---

## 📊 Visualizations

- Class distribution (before & after balancing)
- Message length histograms by category
- Sentiment polarity distributions
- Accuracy/Loss curves per epoch
