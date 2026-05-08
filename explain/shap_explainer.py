import shap
import numpy as np

def explain_shap(text, model, vectorizer):
    X = vectorizer.transform([text])

    # 🔥 bigger + more realistic background
    background = vectorizer.transform([
        "good product",
        "bad product",
        "excellent quality",
        "worst experience",
        "very good item",
        "not satisfied",
        "amazing product",
        "terrible product"
    ])

    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer(X)

    feature_names = vectorizer.get_feature_names_out()
    values = shap_values.values[0]

    X_dense = X.toarray()[0]

    feature_importance = []

    for i in range(len(values)):
        # 🔥 only words actually present in input
        if X_dense[i] > 0 and abs(values[i]) > 0:
            feature_importance.append((feature_names[i], values[i]))

    # fallback safety
    if len(feature_importance) == 0:
        return [("no_signal", 0.0)]

    feature_importance = sorted(
        feature_importance,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return feature_importance[:10]