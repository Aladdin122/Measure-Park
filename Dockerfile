FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SAM 3 model and processor into the image at build time.
# At runtime HF_HUB_OFFLINE=1 prevents any network calls to HuggingFace.
# HF_TOKEN is passed as a build-arg (never stored in the final image env).
ARG HF_TOKEN
RUN HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} python -c "from transformers import Sam3Model, Sam3Processor; Sam3Model.from_pretrained('facebook/sam3'); Sam3Processor.from_pretrained('facebook/sam3')"

COPY stage1.py .
COPY stage2.py .
COPY handler.py .

CMD ["python", "-u", "handler.py"]
