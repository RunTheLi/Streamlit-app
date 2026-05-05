import pickle
import os

def load_model():
    model_path = "artifacts/model.pkl"
    vectorizer_path = "artifacts/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model files not found. Run train.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict(text):
    model, vectorizer = load_model()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return "Positive" if pred == 1 else "Negative"