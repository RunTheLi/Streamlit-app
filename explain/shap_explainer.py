import shap
import numpy as np

def explain_shap(text, model, vectorizer):
    X = vectorizer.transform([text])

    background = vectorizer.transform([
    "good product",
    "bad product",
    "excellent quality",
    "worst experience",
    "average item",
    "not good",
    "very bad product",
    "really good quality"
])

    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer(X)

    feature_names = vectorizer.get_feature_names_out()
    values = shap_values.values[0]

    # 🔥 IMPORTANT: keep only words present in input
    X_dense = X.toarray()[0]

    feature_importance = []
    for i, val in enumerate(values):
        if X_dense[i] > 0:   # only words that appear in text
            feature_importance.append((feature_names[i], val))

    # sort
    feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

    return feature_importance[:10]