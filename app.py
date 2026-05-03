import streamlit as st
from model.predict import load_model, predict
from explain.lime_explainer import explain_lime
from explain.shap_explainer import explain_shap

st.title("🧠 Sentiment Analysis App with Explainability")

text = st.text_area("Enter your review:")

if st.button("Predict"):
    model, vectorizer = load_model()

    result = predict(text)
    st.subheader(f"Prediction: {result}")

    # LIME
    st.subheader("LIME Explanation")
    lime_exp = explain_lime(text, model, vectorizer)
    st.write(lime_exp.as_list())

    # SHAP
    st.subheader("SHAP Explanation")
    shap_values = explain_shap(text, model, vectorizer)
    st.write(shap_values)