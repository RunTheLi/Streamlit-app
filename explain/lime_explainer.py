from lime.lime_text import LimeTextExplainer

class_names = ["Negative", "Positive"]

def explain_lime(text, model, vectorizer):
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(texts))

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        num_samples=2000  # 🔥 stability fix
    )

    result = exp.as_list()

    # 🔥 safety fallback
    if result is None or len(result) == 0:
        return [("no_signal", 0.0)]

    return result