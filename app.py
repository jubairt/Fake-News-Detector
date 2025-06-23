import spacy
import numpy as np
import pickle
from flask import Flask, request, render_template
import os

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Text to vector
def text_to_vector(text):
    doc = nlp(text)
    # Average the vectors of all non-stop, non-punct tokens
    vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct]
    if not vectors:
        return np.zeros((300,))
    return np.mean(vectors, axis=0)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text")
    if not text:
        return render_template("index.html", prediction="No input provided.")
    
    vec = text_to_vector(text).reshape(1, -1)
    result = model.predict(vec)[0]
    prediction = "Real" if result == 1 else "Fake"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
