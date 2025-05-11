import os
import requests
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Fix MLflow connection issue
try:
    # This works inside AWS/EC2 instances
    host_ip = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4', timeout=2).text
except:
    # Fallback to localhost
    host_ip = "localhost"

# Set the MLflow tracking URI with the correct IP
os.environ['MLFLOW_TRACKING_URI'] = f"http://{host_ip}:8000"

print(f"Using MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")

# Verify MIND dataset access
mind_data_dir = os.environ.get('MIND_DATA_DIR', '/mnt/MIND')
print(f"MIND Data Directory: {mind_data_dir}")
print(f"Contents: {os.listdir(mind_data_dir)}")

# Example of loading MIND data
def load_behaviors(file_path):
    """Load the behaviors.tsv file as a DataFrame."""
    return pd.read_csv(
        file_path, 
        sep="\t", 
        names=['impression_id', 'user_id', 'time', 'history', 'impressions']
    )

def load_news(file_path):
    """Load the news.tsv file as a DataFrame."""
    return pd.read_csv(
        file_path,
        sep="\t",
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    )

# Start MLflow experiment
experiment_name = "MIND-Dataset-Analysis-2"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="initial-data-exploration"):
    # Log parameters
    mlflow.log_param("dataset", "MIND")
    mlflow.log_param("dataset_path", mind_data_dir)
    
    # Example analysis
    try:
        # Try to find and load behaviors file
        behaviors_path = None
        for root, dirs, files in os.walk(mind_data_dir):
            if 'behaviors.tsv' in files:
                behaviors_path = os.path.join(root, 'behaviors.tsv')
                break
        
        if behaviors_path:
            behaviors = load_behaviors(behaviors_path)
            mlflow.log_param("behaviors_file", behaviors_path)
            mlflow.log_metric("num_user_behaviors", len(behaviors))
            
            # Log a simple plot
            plt.figure(figsize=(10, 6))
            behaviors['hour'] = pd.to_datetime(behaviors['time']).dt.hour
            behaviors['hour'].value_counts().sort_index().plot(kind='bar')
            plt.title('User Activity by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('user_activity.png')
            mlflow.log_artifact('user_activity.png')
            
            print(f"Successfully logged metrics and artifacts for {len(behaviors)} behaviors")
        else:
            print("Could not find behaviors.tsv file in the dataset")
    except Exception as e:
        print(f"Error in data analysis: {e}")
        mlflow.log_param("error", str(e))