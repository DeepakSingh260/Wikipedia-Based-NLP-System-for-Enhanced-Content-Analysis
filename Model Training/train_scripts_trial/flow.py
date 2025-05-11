import os
import time
import torch
import mlflow
import asyncio
from fastapi import FastAPI, HTTPException
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient
from typing import Optional
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader

# Import your existing components
from model_train_with_mlflow import (
    setup_dataset,
    load_behaviors,
    load_news,
    process_impressions,
    extract_news_features,
    SimpleTokenizer,
    NewsDataset,
    NewsRecommendationModel,
    parse_args_or_use_defaults
)

MODEL_NAME = "NewsRecommendationModel"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
pipeline_lock = asyncio.Lock()

@task
def prepare_datasets():
    """Task to prepare datasets and features"""
    logger = get_run_logger()
    logger.info("Preparing datasets...")
    
    args = parse_args_or_use_defaults()
    data_dirs = setup_dataset(args)
    
    # Load and process data
    train_behaviors = load_behaviors(os.path.join(data_dirs['train_dir'], 'behaviors.tsv'))
    train_news = load_news(os.path.join(data_dirs['train_dir'], 'news.tsv'))
    dev_behaviors = load_behaviors(os.path.join(data_dirs['dev_dir'], 'behaviors.tsv'))
    
    # Process interactions
    train_interactions = process_impressions(train_behaviors)
    dev_interactions = process_impressions(dev_behaviors)
    
    # Extract and combine features
    train_news_features = extract_news_features(train_news)
    all_news_features = {**train_news_features}
    
    # Build tokenizer
    all_titles = [news['title'] for news in all_news_features.values() if news['title']]
    tokenizer = SimpleTokenizer(max_vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.fit(all_titles)
    
    return {
        'train_interactions': train_interactions,
        'dev_interactions': dev_interactions,
        'news_features': all_news_features,
        'tokenizer': tokenizer,
        'args': args
    }

@task
def train_model(data_package):
    """Task to train the news recommendation model"""
    logger = get_run_logger()
    logger.info("Starting model training...")
    
    args = data_package['args']
    tokenizer = data_package['tokenizer']
    
    # Create datasets
    train_dataset = NewsDataset(
        data_package['train_interactions'],
        data_package['news_features'],
        tokenizer,
        max_history=args.max_history
    )
    
    val_dataset = NewsDataset(
        data_package['dev_interactions'],
        data_package['news_features'],
        tokenizer,
        max_history=args.max_history
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = NewsRecommendationModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    ).to(DEVICE)
    
    # Setup MLflow logging
    mlflow.pytorch.autolog()
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator=DEVICE,
        log_every_n_steps=50,
        logger=False  # We're using MLflow instead of Lightning's logger
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    return model

@task
def evaluate_model(model, data_package):
    """Task to evaluate model performance"""
    logger = get_run_logger()
    logger.info("Evaluating model...")
    
    args = data_package['args']
    tokenizer = data_package['tokenizer']
    
    # Create validation dataset
    val_dataset = NewsDataset(
        data_package['dev_interactions'],
        data_package['news_features'],
        tokenizer,
        max_history=args.max_history
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Run evaluation
    trainer = pl.Trainer(
        devices=1,
        accelerator=DEVICE,
        logger=False
    )
    
    results = trainer.test(model, val_loader)
    val_auc = results[0]['test_auc']
    
    mlflow.log_metric("val_auc", val_auc)
    return val_auc >= 0.75  # Example threshold

@task
def register_model(model, passed: bool):
    """Task to register model if evaluation passed"""
    logger = get_run_logger()
    
    if not passed:
        logger.info("Model did not meet validation criteria")
        return None

    logger.info("Registering model in MLflow Model Registry...")
    
    # Log model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="news_recommender",
        registered_model_name=MODEL_NAME
    )
    
    # Set alias
    client = MlflowClient()
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="staging",
        version=latest_version
    )
    
    return latest_version

@flow(name="news_recommendation_flow")
def recommendation_pipeline():
    """Main pipeline flow"""
    with mlflow.start_run():
        # Prepare data
        data_package = prepare_datasets()
        
        # Train model
        model = train_model(data_package)
        
        # Evaluate
        evaluation_passed = evaluate_model(model, data_package)
        
        # Conditionally register
        model_version = register_model(model, evaluation_passed)
        
        return model_version

@app.post("/trigger-training")
async def trigger_training():
    """API endpoint to trigger pipeline"""
    if pipeline_lock.locked():
        raise HTTPException(status_code=423, detail="Training already in progress")

    async with pipeline_lock:
        loop = asyncio.get_event_loop()
        version = await loop.run_in_executor(None, recommendation_pipeline)
        return {"status": "completed", "model_version": version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
