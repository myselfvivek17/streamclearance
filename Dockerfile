FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app \
 && mkdir /app && chown app:app /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# uploads/, jobs/ and dry-run rule cards are written under /app at runtime.
COPY --chown=app:app . .
USER app

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
