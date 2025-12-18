import sys
import os
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.preprocess import clean_text

model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

def predict_depression(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Depressed" if prediction == 1 else "Normal"

if __name__ == "__main__":
    sample = "I feel empty and hopeless every day"
    print(predict_depression(sample))