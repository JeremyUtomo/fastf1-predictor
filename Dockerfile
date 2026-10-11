FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTF1_CACHE=/data

WORKDIR /app
COPY requirements.lock .
RUN pip install -r requirements.lock

RUN useradd --uid 1000 --create-home f1 && mkdir -p /data && chown f1 /data
COPY --chown=f1 . .
USER f1

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", "--server.runOnSave=false"]
