#train.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2)  
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('artifacts/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # TEMP SAMPLE DATA (replace with your real dataset later)
    X = [
        "good product", "very good product", "excellent product", "amazing product", "great product",
        "bad product", "very bad product", "terrible product", "worst product",

        "good", "very good", "excellent", "amazing", "great",
        "bad", "very bad", "terrible", "worst",

        "very satisfied", "not satisfied", "satisfied", "unsatisfied"
    ]

    y = [
        1,1,1,1,1,
        0,0,0,0,

        1,1,1,1,1,
        0,0,0,0,

        1,0,1,0
    ]

    train_model(X, y)