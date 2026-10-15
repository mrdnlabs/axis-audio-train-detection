# Training / export environment for the train-detection model (CPU only).
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        tensorflow-cpu==2.17.1 \
        tf_keras==2.17.0 \
        numpy==1.26.4 \
        scipy==1.13.1 \
        soundfile==0.12.1 \
        scikit-learn==1.5.2 \
        requests==2.32.3

WORKDIR /work
