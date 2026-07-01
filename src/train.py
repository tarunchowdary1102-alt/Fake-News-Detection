import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocess import (
    preprocess_texts,
    create_tfidf,
    save_vectorizer,
)

# -----------------------------
# Load Dataset
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Keep only required columns
df = df[["text", "label"]]

# Remove missing values
df.dropna(inplace=True)

# Convert labels to binary
# FAKE -> 0
# REAL -> 1
df["label"] = df["label"].map({
    "FAKE": 0,
    "REAL": 1
})

print("Dataset Shape:", df.shape)

# -----------------------------
# Text Preprocessing
# -----------------------------

print("Cleaning text...")

df["text"] = preprocess_texts(df["text"])

# -----------------------------
# TF-IDF
# -----------------------------

print("Creating TF-IDF features...")

X, vectorizer = create_tfidf(df["text"])

save_vectorizer(vectorizer, os.path.join(MODEL_DIR, "tfidf.pkl"))

y = df["label"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# -----------------------------
# Logistic Regression
# -----------------------------

print("\nTraining Logistic Regression...")

logistic = LogisticRegression(max_iter=1000)

logistic.fit(X_train, y_train)

log_pred = logistic.predict(X_test)

print("\n===== Logistic Regression =====")

print("Accuracy :", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall   :", recall_score(y_test, log_pred))
print("F1 Score :", f1_score(y_test, log_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, log_pred))

print("\nClassification Report")
print(classification_report(y_test, log_pred))

joblib.dump(
    logistic,
    os.path.join(MODEL_DIR, "logistic.pkl")
)

# -----------------------------
# Decision Tree
# -----------------------------

print("\nTraining Decision Tree...")

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=25
)

decision_tree.fit(X_train, y_train)

tree_pred = decision_tree.predict(X_test)

print("\n===== Decision Tree =====")

print("Accuracy :", accuracy_score(y_test, tree_pred))
print("Precision:", precision_score(y_test, tree_pred))
print("Recall   :", recall_score(y_test, tree_pred))
print("F1 Score :", f1_score(y_test, tree_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, tree_pred))

print("\nClassification Report")
print(classification_report(y_test, tree_pred))

joblib.dump(
    decision_tree,
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

print("\n====================================")
print("Models saved successfully!")
print("Location:", MODEL_DIR)
print("====================================")
