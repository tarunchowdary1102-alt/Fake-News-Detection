# 📰 Fake News Detection using Machine Learning

A machine learning project that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP). The project compares **Logistic Regression** and **Decision Tree Classifier** models trained on TF-IDF features extracted from news articles.

---

## Features

- Text preprocessing using NLP
- Stopword removal
- Stemming using Porter Stemmer
- TF-IDF feature extraction
- Logistic Regression model
- Decision Tree Classifier
- Model comparison
- Performance evaluation
- Interactive Flask web application

---

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- NLTK
- Flask
- Joblib

---

## Folder Structure

```
Fake-News-Detection/
│
├── dataset/
│   └── news.csv
│
├── models/
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   └── tfidf.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

Place the dataset inside:

```
dataset/news.csv
```

Required columns:

| Column | Description |
|---------|-------------|
| text | News article |
| label | REAL or FAKE |

Example:

| text | label |
|------|-------|
| Government launches new policy | REAL |
| Aliens elected as world leaders | FAKE |

---

## Installation

Clone the repository.

```bash
git clone https://github.com/your-username/Fake-News-Detection.git
```

Move into the project directory.

```bash
cd Fake-News-Detection
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Train the Models

```bash
python src/train.py
```

This generates:

```
models/
│
├── logistic.pkl
├── decision_tree.pkl
└── tfidf.pkl
```

---

## Predict from Terminal

```bash
python src/predict.py
```

Example output:

```
Choose Model

1. Logistic Regression
2. Decision Tree

Enter News

NASA discovers new planet.

Prediction

REAL NEWS
```

---

## Run Flask Web Application

```bash
python app.py
```

Open your browser.

```
http://127.0.0.1:5000
```

Paste a news article, choose a model, and click **Predict**.

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Future Improvements

- Random Forest Classifier
- Naive Bayes
- SVM
- LSTM
- BERT Transformer
- News API Integration
- User Authentication
- Docker Deployment
- Streamlit Interface

---

## Author

**M. Tarun Chowdary**

B.Tech – Computer Science Engineering

Machine Learning | Python | NLP | Scikit-Learn

---

## License

This project is intended for educational and learning purposes.
