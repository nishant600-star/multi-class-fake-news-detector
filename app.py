import streamlit as st
import joblib
import pandas as pd

# =========================
# Load Saved Files
# =========================

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# =========================
# Prediction Function
# =========================

def predict_news(text):

    vector = vectorizer.transform([text])

    probs = model.predict_proba(vector)[0]

    pred_idx = probs.argmax()

    confidence = probs.max()

    label = encoder.inverse_transform([pred_idx])[0]

    return label, confidence, probs


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Multi-Class Fake News Detector")

st.markdown(
    """
Predict whether a news article belongs to:

- ✅ True
- ❌ Fake
- ⚠️ Bias
- 🎭 Satire
"""
)

user_input = st.text_area(
    "Enter News Article",
    height=200
)

# =========================
# Prediction Button
# =========================

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        label, confidence, probs = predict_news(user_input)

        st.subheader("Prediction Result")

        # Confidence Threshold

        if confidence < 0.60:

            st.warning(
                f"⚠️ Uncertain Prediction ({confidence*100:.2f}%)"
            )

        else:

            if label == "True":
                st.success(f"✅ {label}")

            elif label == "Fake":
                st.error(f"❌ {label}")

            elif label == "Bias":
                st.warning(f"⚠️ {label}")

            else:
                st.info(f"🎭 {label}")

        # Confidence

        st.write(
            f"### Confidence : {confidence*100:.2f}%"
        )

        st.progress(float(confidence))

        # Probability Table

        st.write("## Class Probabilities")

        prob_df = pd.DataFrame({
            "Class": encoder.classes_,
            "Probability (%)": probs * 100
        })

        prob_df = prob_df.sort_values(
            by="Probability (%)",
            ascending=False
        )

        st.dataframe(
            prob_df,
            use_container_width=True
        )