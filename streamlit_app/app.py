"""Streamlit app for the sentiment classifier (Task 4).

The models are NOT stored in this GitHub repo. Both are hosted on Hugging Face
and downloaded when the app starts (then cached by Streamlit):
  * RoBERTa: pretrained by cardiffnlp, used as-is (zero-shot)
  * TF-IDF v2 + Logistic Regression: trained by us in notebook 02

Run locally:   streamlit run streamlit_app/app.py
Deploy:        Streamlit Community Cloud -> main file path: streamlit_app/app.py
"""
import os
import sys

import joblib
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import pipeline

# Hugging Face model repositories
ROBERTA_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # pretrained, used as-is
TFIDF_REPO = "hamza-shabbir94/amazon-review-sentiment-analysis"      # our own trained model

MODELS = {
    "RoBERTa (pretrained)": ROBERTA_ID,
    "TF-IDF v2 + LogReg (our model)": TFIDF_REPO,
}

LABELS = {
    "negative": ("😞", "Negative", "#FF7A7A"),
    "neutral": ("😐", "Neutral", "#9AA6FF"),
    "positive": ("😊", "Positive", "#3DDC97"),
}

EXAMPLES = [
    "Great tablet for the price. My kids love it and the battery lasts all day.",
    "It works, but the screen is dimmer than I expected for the price.",
    "Worst customer service I have ever experienced.",
]


@st.cache_resource(show_spinner="Loading RoBERTa from Hugging Face (first run only)...")
def load_roberta():
    """Download the transformer once and keep it in memory for every visitor."""
    return pipeline("text-classification", model=ROBERTA_ID, top_k=None)


@st.cache_resource(show_spinner="Loading our TF-IDF model from Hugging Face (first run only)...")
def load_tfidf():
    """Download our sklearn Pipeline and its cleaning code from Hugging Face."""
    model_path = hf_hub_download(TFIDF_REPO, "sentiment_model.joblib")
    clean_path = hf_hub_download(TFIDF_REPO, "text_cleaning.py")
    sys.path.insert(0, os.path.dirname(clean_path))   # the Pipeline imports text_cleaning
    return joblib.load(model_path)


def predict(model_name, review):
    """Return {label: probability}, highest first."""
    if model_name.startswith("RoBERTa"):
        out = load_roberta()(review, truncation=True, max_length=512)
        out = out[0] if isinstance(out[0], list) else out
        scores = {d["label"].lower(): float(d["score"]) for d in out}
    else:
        model = load_tfidf()
        probs = model.predict_proba([review])[0]
        scores = {label: float(p) for label, p in zip(model.classes_, probs)}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Amazon Review Sentiment", page_icon="⭐", layout="centered")
st.title("⭐ Amazon Review Sentiment")
st.caption("Paste a product review and get negative / neutral / positive. Both models are hosted on Hugging Face.")

model_name = st.radio("Model", list(MODELS), horizontal=True)
st.caption(f"Hugging Face repo: `{MODELS[model_name]}`")

example = st.selectbox("Try an example (optional)", ["—"] + EXAMPLES)
review = st.text_area(
    "Customer review",
    value="" if example == "—" else example,
    height=150,
    placeholder="Paste a product review here...",
)

if st.button("Analyse sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        scores = predict(model_name, review.strip())
        top = next(iter(scores))
        emoji, name, color = LABELS.get(top, ("", top, "#999999"))

        st.markdown(
            f"<h2 style='color:{color};margin-bottom:0'>{emoji} {name}</h2>"
            f"<p style='margin-top:0'>confidence {scores[top]:.0%} · {model_name}</p>",
            unsafe_allow_html=True,
        )
        for label, p in scores.items():
            e, n, _ = LABELS.get(label, ("", label, ""))
            st.progress(p, text=f"{e} {n}: {p:.0%}")

st.divider()
st.caption("Ironhack AI Engineering · Project 4 · Hamza Shabbir & Yasaman Najafi Jozani")
