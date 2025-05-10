FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

    RUN pip install --no-cache-dir \
    prefect \
    mlflow==2.22.0 \
    fastapi \
    uvicorn \
    cloudpickle==3.1.1 \
    defusedxml==0.7.1 \
    matplotlib==3.10.1 \
    numpy==2.2.5 \
    pandas==2.2.3 \
    psutil==7.0.0 \
    pytorch-lightning==2.5.1.post0 \
    scikit-learn==1.6.1 \
    scipy==1.15.2 \
    torch==2.7.0


# Copy application files
COPY model_train_with_mlflow.py /app/model_train_with_mlflow.py
COPY model.pth /app/model.pth
COPY flow.py /app/flow.py

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
ENTRYPOINT ["python", "flow.py"]
