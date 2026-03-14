FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/services/indexer

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/services/indexer

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
