FROM python:3.11-slim

WORKDIR /workspace/services/agentic-rag

COPY requirements.txt /tmp/agentic-rag-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agentic-rag-requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
