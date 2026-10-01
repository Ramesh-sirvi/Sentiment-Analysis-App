# Sentiment Analysis App

A full-stack **Sentiment Analysis App** application using a fine-tuned **BERT model**. It classifies tweets into **Positive, Negative, Neutral, and Irrelevant** categories.

## 🚀 Features

* BERT-based sentiment classification
* FastAPI backend
* Streamlit frontend
* Text preprocessing
* Single and batch predictions
* Docker support

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* FastAPI
* Streamlit
* Docker

## 📁 Project Structure

```text
Sentiment-Analysis-App/
├── backend/
│   ├── main.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── bert_model/
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── docker-compose.yml
└── twitter-sentiment-analysis.ipynb
```

## ▶️ Run with Docker

Download the bert_model from running the twitter-sentiment-analysis.ipynb

Place the trained model inside:

```text
backend/bert_model/
```

Then run:

```bash
docker compose up --build
```

Open:

```text
Frontend: http://localhost:8501
Backend:  http://localhost:8000
API Docs: http://localhost:8000/docs
```

## 🧠 Model

**BERT (`bert-base-uncased`)** fine-tuned for 4-class Twitter sentiment classification.

🎓 Learning Outcomes

This project demonstrates practical experience with:

Natural Language Processing
BERT and Transformer models
Text classification
Model deployment
REST APIs
FastAPI
Streamlit
Docker
Git and GitHub

## 👨‍💻 Author

**Ramesh Choudhary**

GitHub: `https://github.com/Ramesh-sirvi`

⭐ If you like this project, give it a star!

📄 License

This project is created for educational and learning purposes.
