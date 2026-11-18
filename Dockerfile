FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# CPU-only PyTorch from the official PyTorch index keeps the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fixed-weight TorchScript model and sample images are shipped in the repo.
COPY app ./app
COPY samples ./samples

# Runtime configuration (all optional, defaults shown):
#   MAX_CAPACITY=16        max images queued or in-flight
#   BATCH_SIZE=4           max images per batch
#   BATCH_WAIT_MS=50       idle model waits at most this long to fill a batch
#   INFERENCE_DELAY_MS=0   artificial delay to demo slow inference
ENV MAX_CAPACITY=16 \
    BATCH_SIZE=4 \
    BATCH_WAIT_MS=50 \
    INFERENCE_DELAY_MS=0

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
