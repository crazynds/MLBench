FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /bench


COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN find /usr -type d -name "cv2" -exec rm -rf {} + 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless==4.9.0.80 \
    && pip install --no-cache-dir -r requirements.txt
