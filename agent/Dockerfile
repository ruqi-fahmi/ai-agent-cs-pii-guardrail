# AI Agent — build dari root repo:
#   docker build -t cs-agent:1.0.0 agent
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# server.py = halaman demo + API chat yang menjalankan agent ADK (App + plugin +
# callback). Endpoint-nya TANPA autentikasi, jadi di k8s sengaja tidak dibuka ke
# internet (Service ClusterIP + port-forward).
COPY cs_agent/ ./cs_agent/
COPY server.py .
COPY web/ ./web/

RUN useradd --create-home --uid 10001 app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
