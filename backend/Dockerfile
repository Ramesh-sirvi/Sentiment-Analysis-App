FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessing.py main.py ./
# bert_model/ is large (400MB+) and not committed to source control —
# mount it as a volume or copy it in at build/deploy time. See README.
COPY bert_model ./bert_model

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
