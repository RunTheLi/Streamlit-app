import pickle

def load_model():
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('artifacts/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict(text):
    model, vectorizer = load_model()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return "Positive" if pred == 1 else "Negative"