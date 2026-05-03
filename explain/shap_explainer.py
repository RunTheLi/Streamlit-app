import shap

def explain_shap(text, model, vectorizer):
    explainer = shap.Explainer(model, vectorizer.transform)
    shap_values = explainer([text])
    return shap_values