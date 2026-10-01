<<<<<<< HEAD
# Twitter Sentiment Analysis — Full-Stack (FastAPI + Streamlit)

A two-tier version of the Twitter sentiment project: a **FastAPI
backend** that loads the fine-tuned BERT model once and serves
predictions over HTTP, and a **Streamlit frontend** that calls it. This
is the same model and preprocessing as the single-app version, split
so the model-serving layer and the UI can be deployed, scaled, and
updated independently — and so other clients (a mobile app, another
service, curl) can hit the same API.

```
twitter_sentiment_fullstack/
├── backend/
│   ├── main.py              # FastAPI app: /health, /predict, /predict/batch
│   ├── preprocessing.py     # Same training-matched cleaning pipeline
│   ├── test_preprocessing.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── bert_model/          # <- copy your unzipped model here
├── frontend/
│   ├── app.py                # Streamlit UI, calls the backend over HTTP
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Option A — Run locally without Docker

**1. Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
unzip /path/to/bert_model.zip -d bert_model          # if not already there
uvicorn main:app --reload --port 8000
```

Check it's up: open `http://localhost:8000/health` — should show
`{"status": "ok", "device": "cpu"}` (or `"cuda"` if you have a GPU).
Interactive API docs are auto-generated at `http://localhost:8000/docs`.

**2. Frontend** (in a second terminal)

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

By default the frontend looks for the backend at `http://localhost:8000`.
Override with an environment variable if it's running elsewhere:

```bash
export SENTIMENT_API_URL="http://localhost:8000"   # set SENTIMENT_API_URL=... on Windows
```

## Option B — Run with Docker Compose

```bash
# put your unzipped model at backend/bert_model/ first
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

Compose wires the frontend to reach the backend at `http://backend:8000`
automatically (Docker's internal DNS) — no manual `SENTIMENT_API_URL`
needed in this mode.

## API reference

| Method | Path             | Body                              | Returns                                                   |
|--------|------------------|------------------------------------|-------------------------------------------------------------|
| GET    | `/health`        | –                                  | `{status, device}`                                          |
| POST   | `/predict`       | `{"text": "..."}`                 | `{sentiment, confidence, probabilities, cleaned_text}`       |
| POST   | `/predict/batch` | `{"texts": ["...", "..."]}` (≤256) | list of the above, one per input text                       |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this game!"}'
```

## Why split it this way

- **Independent scaling** — the model is the expensive part (memory,
  GPU); you can run several backend replicas behind a load balancer
  without duplicating the UI, or scale the frontend independently
  since it's lightweight.
- **Reusable API** — any client (a different frontend, a notebook, a
  script, another team's service) can call `/predict` directly instead
  of going through Streamlit.
- **Cleaner deploys** — the backend and frontend can live in separate
  containers/services (e.g. backend on a GPU instance, frontend on a
  small always-on web dyno) and be redeployed independently.

## Notes

- Preprocessing, label ordering, and the "why the original app.py's
  cleaning didn't match training" fix are unchanged from the single-app
  version — see `backend/preprocessing.py`. Run `python
  test_preprocessing.py` inside `backend/` to sanity-check it.
- `bert_model/` (~420MB) is deliberately not shipped in the Docker build
  context by default in this README — for a real deployment, pull it
  from object storage (S3/GCS) or Git LFS at build/start time rather
  than baking it into the image, unless your registry/host is fine
  with a large image.
- The batch endpoint currently runs texts through the model one at a
  time in a Python loop rather than as a single padded batch — simpler
  and safe for typical CSV sizes; if you're pushing thousands of rows
  regularly, batch the tokenizer/model calls together for more speed.
=======
# Twitter Sentiment Analysis using BERT

A Natural Language Processing (NLP) project that performs **multi-class sentiment classification** on Twitter text using **BERT (bert-base-uncased)** and the Hugging Face Transformers library. The model classifies tweets into four sentiment categories: **Positive**, **Negative**, **Neutral**, and **Irrelevant**.

---

## 📌 Project Overview

This project demonstrates an end-to-end NLP pipeline for Twitter sentiment analysis, including:

- Data preprocessing and cleaning
- Text tokenization using BERT tokenizer
- Fine-tuning a pre-trained BERT model
- Model evaluation using standard classification metrics
- Sentiment prediction on unseen tweets

The project is implemented in **Python** using **Google Colab**, **PyTorch**, and the **Hugging Face Transformers** library.

---

## 🎯 Objectives

- Clean and preprocess Twitter text
- Remove unwanted characters and missing values
- Tokenize text using the BERT tokenizer
- Fine-tune a pre-trained BERT model
- Evaluate model performance
- Predict sentiments of new tweets

---

## 📂 Dataset

The project uses the **Twitter Sentiment Analysis** dataset.

### Dataset Columns

| Column | Description |
|---------|-------------|
| tweet_id | Unique Tweet ID |
| entity | Related company/person/topic |
| sentiment | Target label |
| tweet_content | Tweet text |

### Target Classes

- Positive
- Negative
- Neutral
- Irrelevant

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- PyTorch
- Hugging Face Transformers
- Datasets Library
- Google Colab

---

## 📦 Required Libraries

```bash
pip install transformers
pip install datasets
pip install torch
pip install nltk
pip install scikit-learn
pip install pandas
pip install numpy
```

---

## 📁 Project Structure

```
Twitter-Sentiment-Analysis/
│
├── Twitter_sentiment_analysis.ipynb
├── twitter_training.csv
├── twitter_validation.csv
├── README.md
└── requirements.txt
```

---

## 🔄 Workflow

### 1. Import Libraries

- Pandas
- NumPy
- NLTK
- Transformers
- Datasets
- Scikit-learn
- PyTorch

---

### 2. Load Dataset

```python
df = pd.read_csv("twitter_training.csv")
df2 = pd.read_csv("twitter_validation.csv")
```

---

### 3. Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate records
- Removing missing values
- Dropping unnecessary columns
- Cleaning tweet text
- Lowercasing text
- Removing HTML tags
- Removing punctuation
- Removing stopwords

---

### 4. Tokenization

Tweets are tokenized using

```
bert-base-uncased
```

Example:

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

---

### 5. Model

Pre-trained model:

```
BERT Base Uncased
```

```python
AutoModelForSequenceClassification
```

Number of classes:

```
4
```

---

### 6. Training

The model is fine-tuned using the Hugging Face `Trainer` API.

Typical training parameters include:

- Learning Rate
- Batch Size
- Epochs
- Weight Decay
- Evaluation Strategy

---

### 7. Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📊 Results

The notebook evaluates the fine-tuned BERT model on the validation dataset and reports standard classification metrics to measure performance.

> **Note:** Replace this section with your actual results after training.

Example:

| Metric | Score |
|---------|--------|
| Accuracy | 95.97% |
| Precision | 95.99 |
| Recall | 95.97 |
| F1 Score | 95.97 |

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/Ramesh-sirvi/Twitter-Sentiment-Analysis.git
```

### Open Project

```bash
cd Twitter-Sentiment-Analysis
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Launch Notebook

```bash
jupyter notebook
```

or open the notebook directly in **Google Colab**.

---

## 📈 Future Improvements

- Hyperparameter tuning
- Use RoBERTa or DeBERTa models
- Deploy using Streamlit or Flask
- Real-time Twitter sentiment analysis
- Add model explainability using SHAP or LIME
- Experiment with larger transformer models

---

## 📚 Learning Outcomes

This project demonstrates practical experience with:

- Natural Language Processing (NLP)
- Text preprocessing
- BERT architecture
- Hugging Face Transformers
- Transfer Learning
- Multi-class Text Classification
- Model Evaluation
- Deep Learning using PyTorch

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 👨‍💻 Author

**Ramesh Choudhary**

If you found this project useful, consider giving it a ⭐ on GitHub!

---

## 📄 License

This project is intended for educational and research purposes.
>>>>>>> 9a61f79326e472e173ee578089fb05d97ba1e534
