import os
import argparse
import logging
import subprocess
import mlflow
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    # Set MLflow tracking URI if needed
    if 'MLFLOW_TRACKING_URI' not in os.environ:
        os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:8000"
    
    logger.info(f"Using MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    
    # Paths for feedback data and model artifacts
    feedback_dir = os.environ.get('FEEDBACK_DIR', '/mnt/block/feedback_data')
    model_dir = os.environ.get('MODEL_DIR', '/mnt/block/model_artifacts')
    
    # Create directories if they don't exist
    os.makedirs(feedback_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return feedback_dir, model_dir

def get_latest_model_checkpoint():
    client = mlflow.tracking.MlflowClient()
    
    # Get the experiment ID for our news recommendation experiment
    experiment = client.get_experiment_by_name("news_recommendation_mind_large_temporal")
    if experiment is None:
        logger.warning("Experiment not found, searching for any news recommendation experiment")
        for exp in client.search_experiments():
            if "news_recommendation" in exp.name:
                experiment = exp
                break
    
    if experiment is None:
        logger.error("No news recommendation experiment found")
        return None
    
    # Get the best run based on validation AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_auc DESC"]
    )
    
    if not runs:
        logger.error("No runs found in the experiment")
        return None
    
    best_run = runs[0]
    logger.info(f"Found best run: {best_run.info.run_id} with val_auc: {best_run.data.metrics.get('val_auc', 'unknown')}")
    
    # Get the artifacts for the best run
    artifacts = client.list_artifacts(best_run.info.run_id)
    checkpoint_artifacts = [art for art in artifacts if art.path.endswith('.ckpt') or art.path.endswith('.pth')]
    
    if not checkpoint_artifacts:
        logger.error("No checkpoint artifacts found in the best run")
        return None
    
    # Download the best checkpoint
    local_model_path = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    checkpoint_path = checkpoint_artifacts[0].path
    
    logger.info(f"Downloading checkpoint from run {best_run.info.run_id}: {checkpoint_path}")
    local_path = client.download_artifacts(best_run.info.run_id, checkpoint_path, ".")
    
    logger.info(f"Downloaded checkpoint to {local_path}")
    return local_path

def run_retraining(feedback_path, initial_weights, output_dir):
    """Run the retraining process with the provided feedback and weights"""
    cmd = [
        "python", "train_withtimestamps.py",
        "--retrain",
        f"--feedback_path={feedback_path}",
        f"--initial_weights={initial_weights}",
        "--batch_size=64",
        "--epochs=3",  # Use fewer epochs for retraining
        "--learning_rate=0.0005"  # Lower learning rate for fine-tuning
    ]
    
    # Additional args for GPU environment if available
    if torch.cuda.is_available():
        cmd.extend([
            "--device=cuda",
            "--use_fp16",
            f"--num_gpus={torch.cuda.device_count()}"
        ])
    
    logger.info(f"Running retraining command: {' '.join(cmd)}")
    
    # Run the training script
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output to logger
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    return process.returncode == 0

def main():
    """Main function for retraining workflow"""
    parser = argparse.ArgumentParser(description='Retrain news recommendation model with feedback data')
    parser.add_argument('--feedback_path', type=str, help='Path to processed feedback data (defaults to most recent)')
    parser.add_argument('--model_checkpoint', type=str, help='Path to model checkpoint (defaults to best from MLflow)')
    args = parser.parse_args()
    
    feedback_dir, model_dir = setup_environment()
    
    # Find the latest feedback data if not provided
    feedback_path = args.feedback_path
    if not feedback_path:
        feedback_files = [os.path.join(feedback_dir, f) for f in os.listdir(feedback_dir) 
                         if f.endswith('.tsv') and 'processed' in f]
        if feedback_files:
            feedback_path = sorted(feedback_files)[-1]  # Get the most recent feedback file
            logger.info(f"Using latest feedback data: {feedback_path}")
        else:
            logger.error("No processed feedback data found")
            return 1
    
    # Find the best model checkpoint if not provided
    model_checkpoint = args.model_checkpoint
    if not model_checkpoint:
        model_checkpoint = get_latest_model_checkpoint()
        if not model_checkpoint:
            logger.error("Failed to find latest model checkpoint")
            return 1
    
    # Run the retraining
    success = run_retraining(feedback_path, model_checkpoint, model_dir)
    
    if success:
        logger.info("Retraining completed successfully")
        return 0
    else:
        logger.error("Retraining failed")
        return 1

if __name__ == "__main__":
    import torch  # Import here for cleaner error handling
    exit(main())