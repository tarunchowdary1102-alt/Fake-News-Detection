import os
import joblib

from preprocess import clean_text

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

LOGISTIC_MODEL = os.path.join(MODEL_DIR, "logistic.pkl")
DECISION_TREE_MODEL = os.path.join(MODEL_DIR, "decision_tree.pkl")
TFIDF_MODEL = os.path.join(MODEL_DIR, "tfidf.pkl")


# --------------------------------------------------
# Load Models
# --------------------------------------------------

print("Loading models...")

logistic = joblib.load(LOGISTIC_MODEL)
decision_tree = joblib.load(DECISION_TREE_MODEL)
vectorizer = joblib.load(TFIDF_MODEL)


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_news(news_text, model_name="logistic"):

    cleaned = clean_text(news_text)

    vector = vectorizer.transform([cleaned])

    if model_name.lower() == "decision_tree":
        prediction = decision_tree.predict(vector)[0]
    else:
        prediction = logistic.predict(vector)[0]

    if prediction == 1:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"


# --------------------------------------------------
# User Input
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" Fake News Detection ")
    print("=" * 60)

    print("\nChoose Model")
    print("1. Logistic Regression")
    print("2. Decision Tree")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        model = "decision_tree"
    else:
        model = "logistic"

    print("\nEnter News Article\n")

    article = input("> ")

    result = predict_news(article, model)

    print("\nPrediction")
    print("-" * 40)
    print(result)
