import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # save artifacts
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('artifacts/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)