import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# -----------------------------------------------------
# Dataset Functions
# -----------------------------------------------------

def load_dataset(csv_path):
    """
    Load dataset from CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    df = df[["text", "label"]]
    df.dropna(inplace=True)

    return df


# -----------------------------------------------------
# Model Functions
# -----------------------------------------------------

def save_model(model, path):
    """
    Save trained model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at: {path}")


def load_model(path):
    """
    Load trained model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))


# -----------------------------------------------------
# Single Prediction
# -----------------------------------------------------

def predict_single(model, vectorizer, clean_function, text):
    """
    Predict one news article.
    """
    cleaned_text = clean_function(text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "REAL NEWS"

    return "FAKE NEWS"


# -----------------------------------------------------
# Batch Prediction
# -----------------------------------------------------

def predict_batch(model, vectorizer, clean_function, texts):
    """
    Predict multiple news articles.
    """
    cleaned = [clean_function(t) for t in texts]

    vectors = vectorizer.transform(cleaned)

    predictions = model.predict(vectors)

    results = []

    for value in predictions:
        if value == 1:
            results.append("REAL NEWS")
        else:
            results.append("FAKE NEWS")

    return results


# -----------------------------------------------------
# Pretty Printer
# -----------------------------------------------------

def print_header(title):
    """
    Print formatted heading.
    """
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
