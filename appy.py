import os
import joblib

from flask import Flask, request, render_template_string
from src.preprocess import clean_text

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LOGISTIC_MODEL = os.path.join(MODEL_DIR, "logistic.pkl")
DECISION_TREE_MODEL = os.path.join(MODEL_DIR, "decision_tree.pkl")
TFIDF_MODEL = os.path.join(MODEL_DIR, "tfidf.pkl")

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------

logistic = joblib.load(LOGISTIC_MODEL)
decision_tree = joblib.load(DECISION_TREE_MODEL)
vectorizer = joblib.load(TFIDF_MODEL)

# ----------------------------------------------------
# Flask App
# ----------------------------------------------------

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection</title>

    <style>

        body{
            font-family:Arial;
            background:#f5f5f5;
            margin:40px;
        }

        .container{
            width:700px;
            margin:auto;
            background:white;
            padding:30px;
            border-radius:10px;
            box-shadow:0 0 10px gray;
        }

        textarea{
            width:100%;
            height:220px;
            font-size:16px;
            padding:10px;
        }

        select{
            width:100%;
            padding:10px;
            margin-top:15px;
            font-size:16px;
        }

        button{
            margin-top:20px;
            width:100%;
            padding:12px;
            background:#2196F3;
            color:white;
            border:none;
            font-size:18px;
            cursor:pointer;
        }

        button:hover{
            background:#0b7dda;
        }

        h1{
            text-align:center;
        }

        h2{
            text-align:center;
            color:green;
        }

    </style>

</head>

<body>

<div class="container">

<h1>Fake News Detection</h1>

<form method="POST">

<textarea
name="news"
placeholder="Paste the news article here..."
required>{{news}}</textarea>

<br>

<select name="model">

<option value="logistic">Logistic Regression</option>

<option value="decision_tree">Decision Tree</option>

</select>

<button type="submit">
Predict
</button>

</form>

{% if result %}

<h2>
Prediction : {{result}}
</h2>

{% endif %}

</div>

</body>

</html>

"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    article = ""

    if request.method == "POST":

        article = request.form["news"]

        model = request.form["model"]

        cleaned = clean_text(article)

        vector = vectorizer.transform([cleaned])

        if model == "decision_tree":
            pred = decision_tree.predict(vector)[0]
        else:
            pred = logistic.predict(vector)[0]

        if pred == 1:
            prediction = "REAL NEWS"
        else:
            prediction = "FAKE NEWS"

    return render_template_string(
        HTML,
        result=prediction,
        news=article
    )


if __name__ == "__main__":
    app.run(debug=True)
