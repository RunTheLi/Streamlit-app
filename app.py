import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model.predict import load_model, predict
from explain.lime_explainer import explain_lime
from explain.shap_explainer import explain_shap


st.title("🧠 Sentiment Analysis App with Explainability")

text = st.text_area("Enter your review:")


# ------------------ SHAP PLOT ------------------
def plot_shap(feature_importance):

    if len(feature_importance) == 0:
        st.warning("No important features found for this input.")
        return

    words = [x[0] for x in feature_importance]
    values = [x[1] for x in feature_importance]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words, values)

    ax.set_title("SHAP Explanation")
    ax.set_xlabel("Impact")
    ax.axvline(0)

    st.pyplot(fig)


# ------------------ LIME PLOT ------------------
def plot_lime(words, values):
    sorted_idx = np.argsort(values)

    words = np.array(words)[sorted_idx]
    values = np.array(values)[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words, values)

    ax.set_title("LIME Explanation")
    ax.set_xlabel("Importance")
    ax.axvline(0, color='black', linewidth=1)

    st.pyplot(fig)


# ------------------ MAIN APP ------------------
if st.button("Predict"):

    if not text.strip():
        st.warning("Please enter some text.")
    else:
        model, vectorizer = load_model()

        # normalize input
        text_clean = text.lower().strip()

        # Prediction
        result = predict(text_clean)
        st.subheader(f"Prediction: {result}")

        proba = model.predict_proba(vectorizer.transform([text_clean]))[0]
        confidence = max(proba)

        st.metric("Confidence", f"{confidence*100:.2f}%")

        # ------------------ LIME ------------------
        st.subheader("📊 LIME Explanation")

        lime_exp = explain_lime(text_clean, model, vectorizer)
        lime_list = lime_exp.as_list()

        if len(lime_list) == 0:
            st.warning("LIME could not generate explanation.")
        else:
            words = [x[0] for x in lime_list]
            values = [x[1] for x in lime_list]

            plot_lime(words, values)

            with st.expander("Raw LIME values"):
                st.write(lime_list)

        # ------------------ SHAP ------------------
        st.subheader("📈 SHAP Explanation")

        try:
            shap_data = explain_shap(text_clean, model, vectorizer)

            if len(shap_data) == 0:
                st.warning("SHAP could not generate explanation for this input.")
            else:
                plot_shap(shap_data)

                with st.expander("Raw SHAP values"):
                    st.write(shap_data)

        except Exception as e:
            st.warning(f"SHAP failed: {e}")