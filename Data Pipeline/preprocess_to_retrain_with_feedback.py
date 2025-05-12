import os
import shutil
import argparse
import subprocess
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feedback_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='News Recommendation Feedback Pipeline')
    parser.add_argument('--kvm_host', type=str, default=None, help='KVM@TACC hostname for SSH transfer')
    parser.add_argument('--feedback_source', type=str, default='Serving/feedback_data/feedback.csv',
                        help='Path to feedback data on KVM@TACC')
    parser.add_argument('--feedback_dir', type=str, default='/mnt/block/feedback_data',
                        help='Local directory for feedback data')
    parser.add_argument('--object_container', type=str, default='object-persist-project-50',
                        help='Object storage container name')
    parser.add_argument('--skip_transfer', action='store_true', help='Skip data transfer step')
    parser.add_argument('--skip_transform', action='store_true', help='Skip data transformation step')
    parser.add_argument('--skip_retrain', action='store_true', help='Skip model retraining step')
    return parser.parse_args()

def transfer_data(args):
    """Transfer feedback data from KVM@TACC to local storage"""
    logger.info("Starting feedback data transfer...")

    # Ensure the feedback directory exists
    os.makedirs(args.feedback_dir, exist_ok=True)

    # Method 1: Direct SSH transfer if hostname provided
    if args.kvm_host:
        logger.info(f"Using direct SSH transfer from {args.kvm_host}")

        scp_cmd = [
            "scp",
            f"cc@{args.kvm_host}:{args.feedback_source}",
            f"{args.feedback_dir}/feedback_raw.csv"
        ]

        logger.info(f"Running command: {' '.join(scp_cmd)}")
        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("SCP transfer successful")
            return os.path.join(args.feedback_dir, "feedback_raw.csv")
        else:
            logger.error(f"SCP transfer failed: {result.stderr}")

            # Fallback to rclone
            logger.info("Falling back to rclone transfer")

    logger.info(f"Using rclone transfer from container {args.object_container}")

    check_cmd = [
        "rclone", "lsf",
        f"chi_tacc:{args.object_container}/feedback_transfer/feedback.csv"
    ]

    check_result = subprocess.run(check_cmd, capture_output=True, text=True)

    if check_result.returncode != 0 or not check_result.stdout.strip():
        logger.error("Feedback file not found in object storage")
        return None

    # File exists, copy it locally
    copy_cmd = [
        "rclone", "copy",
        f"chi_tacc:{args.object_container}/feedback_transfer/feedback.csv",
        args.feedback_dir
    ]

    logger.info(f"Running command: {' '.join(copy_cmd)}")
    copy_result = subprocess.run(copy_cmd, capture_output=True, text=True)

    if copy_result.returncode == 0:
        logger.info("Rclone transfer successful")

        # Rename the file to include the timestamp
        source_file = os.path.join(args.feedback_dir, "feedback.csv")
        dest_file = os.path.join(args.feedback_dir, "feedback_raw.csv")

        try:
            shutil.copy2(source_file, dest_file)
            logger.info(f"Copied feedback.csv to {dest_file}")
            return dest_file
        except Exception as e:
            logger.error(f"Error copying feedback file: {e}")
            return source_file
    else:
        logger.error(f"Rclone transfer failed: {copy_result.stderr}")
        return None

def transform_feedback(raw_feedback_path, output_dir):
    """Transform raw feedback data into format needed for model retraining"""
    logger.info(f"Transforming feedback data from {raw_feedback_path}")

    if not os.path.exists(raw_feedback_path):
        logger.error(f"Raw feedback file not found at {raw_feedback_path}")
        return None

    try:
        # Read the raw feedback CSV
        feedback_df = pd.read_csv(raw_feedback_path)
        logger.info(f"Loaded {len(feedback_df)} feedback entries")

        # Create a timestamped output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        processed_path = os.path.join(output_dir, f"processed_feedback_{timestamp}.tsv")

        # Transform the feedback data into the format expected for training
        # Group by user_id to consolidate feedback
        user_feedback = {}
        for _, row in feedback_df.iterrows():
            user_id = row['user_id']
            news_id = row['news_id']
            feedback = row['feedback']

            if user_id not in user_feedback:
                user_feedback[user_id] = []

            # Add the user interaction with label
            user_feedback[user_id].append((news_id, int(feedback)))

        # Convert to the format used in behaviors.tsv
        processed_rows = []
        current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S %p")

        for i, (user_id, interactions) in enumerate(user_feedback.items()):
            impression_id = f"FDB-{i}-{timestamp}"

            history = " ".join([news_id for news_id, feedback in interactions if int(feedback) == 1])

            # For impressions, we'll include all interactions with their feedback
            impressions = " ".join([f"{news_id}-{feedback}" for news_id, feedback in interactions])

            processed_rows.append({
                'impression_id': impression_id,
                'user_id': user_id,
                'time': current_time,
                'history': history,
                'impressions': impressions
            })

        # Convert to DataFrame and save as TSV
        processed_df = pd.DataFrame(processed_rows)
        processed_df.to_csv(processed_path, sep='\t', index=False, header=False)

        logger.info(f"Processed {len(feedback_df)} feedback entries into {len(processed_df)} user records")
        logger.info(f"Saved to {processed_path}")

        # Create a symlink to the latest processed file for easier reference
        latest_link = os.path.join(output_dir, "latest_processed_feedback.tsv")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(processed_path, latest_link)
        logger.info(f"Created symlink {latest_link} -> {processed_path}")

        # Make a backup of the raw feedback
        backup_dir = os.path.join(output_dir, "history")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"feedback_raw_{timestamp}.csv")
        shutil.copy2(raw_feedback_path, backup_path)
        logger.info(f"Backed up raw feedback to {backup_path}")

        return processed_path
    except Exception as e:
        logger.error(f"Error transforming feedback data: {e}")
        return None

def trigger_retraining(processed_feedback_path):
    """Trigger model retraining using the transformed feedback data"""
    logger.info(f"Triggering model retraining with feedback: {processed_feedback_path}")

    if not os.path.exists(processed_feedback_path):
        logger.error(f"Processed feedback file not found at {processed_feedback_path}")
        return False

    # Use the retrain_with_feedback.py script
    retrain_cmd = [
        "python", "retrain_with_feedback.py",
        f"--feedback_path={processed_feedback_path}"
    ]

    logger.info(f"Running command: {' '.join(retrain_cmd)}")

    try:
        # Run the retraining script
        process = subprocess.Popen(
            retrain_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Stream output to logger
        for line in process.stdout:
            logger.info(line.strip())

        process.wait()

        if process.returncode == 0:
            logger.info("Retraining completed successfully")
            return True
        else:
            logger.error(f"Retraining failed with exit code {process.returncode}")
            return False
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        return False

def main():
    """Main function to run the feedback pipeline"""
    args = parse_arguments()

    logger.info("Starting News Recommendation Feedback Pipeline")

    # Step 1: Transfer feedback data
    raw_feedback_path = None
    if not args.skip_transfer:
        raw_feedback_path = transfer_data(args)
        if not raw_feedback_path:
            logger.error("Data transfer failed, exiting pipeline")
            return 1
    else:
        logger.info("Skipping data transfer step")
        # Use the most recent raw feedback file
        raw_files = [f for f in os.listdir(args.feedback_dir) if f.startswith("feedback_raw") or f == "feedback.csv"]
        if raw_files:
            raw_feedback_path = os.path.join(args.feedback_dir, sorted(raw_files)[-1])
            logger.info(f"Using existing feedback file: {raw_feedback_path}")
        else:
            logger.error("No raw feedback file found, cannot continue")
            return 1

    # Step 2: Transform feedback data
    processed_feedback_path = None
    if not args.skip_transform:
        processed_feedback_path = transform_feedback(raw_feedback_path, args.feedback_dir)
        if not processed_feedback_path:
            logger.error("Data transformation failed, exiting pipeline")
            return 1
    else:
        logger.info("Skipping data transformation step")
        # Use the most recent processed feedback file
        link_path = os.path.join(args.feedback_dir, "latest_processed_feedback.tsv")
        if os.path.exists(link_path):
            processed_feedback_path = link_path
            logger.info(f"Using existing processed feedback file: {processed_feedback_path}")
        else:
            # Look for any processed feedback files
            processed_files = [f for f in os.listdir(args.feedback_dir) if f.startswith("processed_feedback") and f.endswith(".tsv")]
            if processed_files:
                processed_feedback_path = os.path.join(args.feedback_dir, sorted(processed_files)[-1])
                logger.info(f"Using existing processed feedback file: {processed_feedback_path}")
            else:
                logger.error("No processed feedback file found, cannot continue")
                return 1

    # Step 3: Trigger model retraining
    if not args.skip_retrain:
        success = trigger_retraining(processed_feedback_path)
        if not success:
            logger.error("Model retraining failed")
            return 1
        logger.info("Model retraining completed successfully")
    else:
        logger.info("Skipping model retraining step")

    logger.info("Feedback pipeline completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())