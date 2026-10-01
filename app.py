import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Twitter Sentiment Analysis - BERT
# Based on the uploaded New_Twitter_data.ipynb
# ============================================================

# Keep this folder next to app.py:
# Twitter_App/
# ├── app.py
# └── bert_model/
#     ├── config.json
#     ├── model.safetensors
#     ├── tokenizer.json
#     └── tokenizer_config.json

MODEL_PATH = "./bert_model"

# The notebook uses sklearn LabelEncoder.
# With the four labels in the dataset, LabelEncoder orders them
# alphabetically as follows:
# 0 = Irrelevant
# 1 = Negative
# 2 = Neutral
# 3 = Positive
LABELS = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
}


# ------------------------------------------------------------
# Same preprocessing used in the training notebook
# ------------------------------------------------------------
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# Load the trained model
# @st.cache_resource prevents loading BERT again on every click
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------
def predict_sentiment(tweet, tokenizer, model, device):

    # Apply the same preprocessing used during training
    cleaned_tweet = preprocess_text(tweet)

    # Same tokenization settings used in the notebook
    inputs = tokenizer(
        cleaned_tweet,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    # Move tensors to CPU/GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits into probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    # Find class with highest probability
    predicted_class = torch.argmax(probabilities).item()

    sentiment = LABELS.get(
        predicted_class,
        f"Class {predicted_class}"
    )

    confidence = probabilities[predicted_class].item()

    return sentiment, confidence, probabilities, cleaned_tweet


# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered",
)

# ============================================================
# Header
# ============================================================
st.title("🐦 Twitter Sentiment Analysis")
st.write(
    "Enter a tweet and the fine-tuned BERT model will predict "
    "its sentiment."
)

st.divider()

# ============================================================
# Load model
# ============================================================
try:
    tokenizer, model, device = load_model()

    st.success("BERT model loaded successfully.")

    tweet = st.text_area(
        "Enter your tweet",
        height=150,
        placeholder="Example: I absolutely love this game!",
    )

    # ========================================================
    # Analyze button
    # ========================================================
    if st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    ):

        if not tweet.strip():
            st.warning("Please enter a tweet.")

        else:
            with st.spinner("Analyzing tweet..."):

                sentiment, confidence, probabilities, cleaned_tweet = (
                    predict_sentiment(
                        tweet,
                        tokenizer,
                        model,
                        device,
                    )
                )

            st.divider()

            # ------------------------------------------------
            # Main prediction
            # ------------------------------------------------
            st.subheader("Prediction")

            if sentiment == "Positive":
                st.success(f"😊 {sentiment}")

            elif sentiment == "Negative":
                st.error(f"😞 {sentiment}")

            elif sentiment == "Neutral":
                st.info(f"😐 {sentiment}")

            else:
                st.warning(f"❓ {sentiment}")

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

            # ------------------------------------------------
            # Probability for every class
            # ------------------------------------------------
            st.subheader("Class Probabilities")

            for class_id, label in LABELS.items():

                probability = probabilities[class_id].item()

                st.write(
                    f"**{label}:** {probability:.2%}"
                )

                st.progress(float(probability))

            # ------------------------------------------------
            # Processed tweet
            # ------------------------------------------------
            with st.expander("View processed tweet"):

                st.write(cleaned_tweet)

            # ------------------------------------------------
            # Model information
            # ------------------------------------------------
            with st.expander("Model information"):

                st.write("Model: BERT (bert-base-uncased)")
                st.write("Number of classes: 4")
                st.write("Maximum sequence length: 128")
                st.write(f"Running on: {device}")

except Exception as e:

    st.error("Unable to load the trained BERT model.")

    st.write(
        "Check that the 'bert_model' folder is in the same "
        "directory as app.py."
    )

    st.code(
        """Twitter_App/
├── app.py
└── bert_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json"""
    )

    st.exception(e)
