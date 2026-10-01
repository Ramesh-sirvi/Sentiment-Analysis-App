"""
FastAPI backend for the Twitter sentiment BERT model.

Endpoints:
    GET  /health          -> simple liveness/readiness check
    POST /predict         -> predict sentiment for one piece of text
    POST /predict/batch   -> predict sentiment for a list of texts

Run with:
    uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import ensure_nltk_resources, preprocess_text

MODEL_PATH = "./bert_model"
MAX_LENGTH = 128
MAX_BATCH_SIZE = 256  # guardrail so one request can't hang the server

# Same alphabetical LabelEncoder ordering used during training.
LABELS = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
}

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once, keep it in memory for the life of the process.
    ensure_nltk_resources()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device

    yield

    _state.clear()


app = FastAPI(
    title="Twitter Sentiment API",
    description="Serves the fine-tuned BERT model from Twitter_sentiment_analysis.ipynb",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Tweet text to classify")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in _state else "loading",
        "device": str(_state.get("device", "unknown")),
    }


def _run_prediction(text: str) -> PredictResponse:
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]

    cleaned = preprocess_text(text)

    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_class = torch.argmax(probs).item()

    return PredictResponse(
        sentiment=LABELS.get(predicted_class, f"Class {predicted_class}"),
        confidence=probs[predicted_class].item(),
        probabilities={LABELS[i]: probs[i].item() for i in range(len(LABELS))},
        cleaned_text=cleaned,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return _run_prediction(request.text)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return [_run_prediction(text) for text in request.texts]
