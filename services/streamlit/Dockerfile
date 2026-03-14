FROM python:3.11-slim

WORKDIR /workspace/services/streamlit

COPY requirements.txt /tmp/streamlit-requirements.txt
RUN pip install --no-cache-dir -r /tmp/streamlit-requirements.txt

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
