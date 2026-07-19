FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    U2NET_HOME=/home/user/.u2net \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        torch==2.5.1+cpu torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user && mkdir -p /home/user/.cache/huggingface /home/user/.u2net && \
    chown -R user:user /home/user /app
COPY --chown=user:user . /app

USER user
EXPOSE 7860
CMD ["python", "app.py"]
