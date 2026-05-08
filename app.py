import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model.predict import load_model, predict
from explain.lime_explainer import explain_lime
from explain.shap_explainer import explain_shap


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sentiment AI Dashboard",
    page_icon="",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("AI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Explainability", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit + LIME + SHAP")

# ------------------ LOAD MODEL (cache later optional) ------------------
model, vectorizer = load_model()


# ================== DASHBOARD PAGE ==================
if page == "🏠 Dashboard":

    st.title("Sentiment Analysis Dashboard")

    text = st.text_area("Enter your review:", height=150)

    if st.button("Analyze"):

        if not text.strip():
            st.warning("Please enter text.")
        else:
            text_clean = text.lower().strip()

            result = predict(text_clean)
            proba = model.predict_proba(vectorizer.transform([text_clean]))[0]
            confidence = max(proba)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", result)

            with col2:
                st.metric("Confidence", f"{confidence*100:.2f}%")

            with col3:
                st.metric("Model", "Logistic Regression")

            st.success(f"Processed: {text_clean}")


# ================== EXPLAINABILITY PAGE ==================
elif page == "📊 Explainability":

    st.title("Model Explainability (LIME + SHAP)")

    text = st.text_area("Enter text for explanation:", height=120)

    if st.button("Explain"):

        if not text.strip():
            st.warning("Enter text first.")
        else:
            text_clean = text.lower().strip()

            col1, col2 = st.columns(2)

            # ------------------ LIME ------------------
            with col1:
                st.subheader("LIME")

                lime_list = explain_lime(text_clean, model, vectorizer)

                if len(lime_list) == 0:
                    st.warning("No LIME explanation.")
                else:
                    words = [x[0] for x in lime_list]
                    values = [x[1] for x in lime_list]

                    fig, ax = plt.subplots(figsize=(5, 4))
                    sorted_idx = np.argsort(values)

                    words = np.array(words)[sorted_idx]
                    values = np.array(values)[sorted_idx]

                    ax.barh(words, values)
                    ax.set_title("LIME Explanation")

                    st.pyplot(fig)

                    with st.expander("Raw LIME"):
                        st.write(lime_list)

            # ------------------ SHAP ------------------
            with col2:
                st.subheader("SHAP")

                try:
                    shap_data = explain_shap(text_clean, model, vectorizer)

                    if len(shap_data) == 0:
                        st.warning("No SHAP explanation.")
                    else:
                        words = [x[0] for x in shap_data]
                        values = [x[1] for x in shap_data]

                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.barh(words, values)
                        ax.set_title("SHAP Explanation")

                        st.pyplot(fig)

                        with st.expander("Raw SHAP"):
                            st.write(shap_data)

                except Exception as e:
                    st.error(f"SHAP error: {e}")


# ================== ABOUT PAGE ==================
elif page == "ℹ️ About":

    st.title("About This Project")

    st.markdown("""
    ### Sentiment Analysis AI App

    This project demonstrates:
    - Machine Learning (Logistic Regression)
    - TF-IDF Text Vectorization
    - Model Explainability (LIME + SHAP)
    - Streamlit Web App UI

    ---

    ### Tech Stack
    - Python
    - Scikit-learn
    - Streamlit
    - LIME
    - SHAP

    ---

    ### Features
    - Sentiment Prediction (Positive / Negative)
    - Confidence Score
    - LIME Explanation
    - SHAP Explanation
    - Interactive Dashboard UI

    ---

    ### Goal
    Turn ML notebook → real-world AI product demo
    """)