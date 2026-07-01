import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords (only needed the first time)
nltk.download("stopwords")

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean and preprocess text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = text.strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and apply stemming
    processed_words = []

    for word in words:
        if word not in stop_words:
            processed_words.append(stemmer.stem(word))

    return " ".join(processed_words)


def preprocess_texts(texts):
    """
    Apply preprocessing to a list of texts.
    """
    return [clean_text(text) for text in texts]


def create_tfidf(train_texts):
    """
    Train TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(train_texts)

    return X, vectorizer


def transform_texts(vectorizer, texts):
    """
    Transform new texts using an existing vectorizer.
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path="../models/tfidf.pkl"):
    """
    Save TF-IDF vectorizer.
    """
    joblib.dump(vectorizer, path)


def load_vectorizer(path="../models/tfidf.pkl"):
    """
    Load TF-IDF vectorizer.
    """
    return joblib.load(path)
