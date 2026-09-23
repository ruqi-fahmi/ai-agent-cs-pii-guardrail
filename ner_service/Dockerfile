# NER Service — build dari root repo:
#   docker build -t ner-service:1.0.0 ner_service
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# requirements dulu -> layer dependency di-cache selama requirements tidak berubah
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model hasil `python ner_service/train.py` ikut di-bake ke image:
# image = kode + model dengan versi yang terkunci bersama.
COPY app.py postprocess.py ./
COPY model/ ./model/

# Jangan jalan sebagai root
RUN useradd --create-home --uid 10001 app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
