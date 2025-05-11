# Object Storage Setup Guide for MIND Dataset Project

This document outlines the complete process for setting up object storage and block storage for the MIND dataset project in a cloud environment.

## 1. Overview

The setup involves:

- Setting up object storage for the MIND dataset
- Downloading and extracting the MIND dataset
- Loading the dataset into object storage
- Creating and attaching block storage
- Setting up a monitoring and experiment tracking infrastructure
- Mounting the object storage for use in ML workflows

## 2. Object Storage Setup with rclone

### 2.1 Install rclone

```bash
# If rclone is not already installed
curl https://rclone.org/install.sh | sudo bash
```

### 2.2 Configure rclone

```bash
# Create config directory
mkdir -p ~/.config/rclone

# Create or edit rclone.conf with your credentials
nano ~/.config/rclone/rclone.conf
```

Add the following configuration (adjust with your credentials):

```
[chi_tacc]
type = s3
provider = Other
env_auth = false
access_key_id = your_access_key
secret_access_key = your_secret_key
endpoint = your_endpoint_url
```

## 3. MIND Dataset ETL Pipeline

### 3.1 Create Docker Compose File for ETL

Create a file named `docker-compose-mind-etl.yaml` that includes services for:

- **download-data**: Downloads MIND dataset zip files from NYU Box storage
- **transform-data**: Extracts files into training, validation, and testing directories
- **load-data**: Uses rclone to upload data to object storage

The complete file can be found in the project repository.

### 3.2 Run the ETL Pipeline

```bash
# Run the download stage
docker compose -f docker-compose-mind-etl.yaml run download-data

# Run the transform stage
docker compose -f docker-compose-mind-etl.yaml run transform-data

# Run the load stage (set your container name)
export RCLONE_CONTAINER=object-persist-project-50
docker compose -f docker-compose-mind-etl.yaml run load-data
```

### 3.3 Verify the Object Storage

```bash
# Verify the object storage contents
rclone ls chi_tacc:object-persist-project-50
```

## 4. Block Storage Setup

### 4.1 Create a Block Storage Volume

1. From the Chameleon dashboard, navigate to:

   - Experiment > KVM@TACC
   - Volumes > Volumes > Create Volume

2. Configure the volume:
   - Name: `block-persist-project-50` (or your preferred name)
   - Size: 2 GiB
   - Leave other settings at defaults
   - Click "Create Volume"

### 4.2 Attach the Volume to Your Instance

1. Next to your volume, click the dropdown (▼) and choose "Manage Attachments"
2. Select your compute instance
3. Click "Attach Volume"

### 4.3 Format and Mount the Volume

```bash
# Verify the block device exists
lsblk

# Create a partition
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%

# Verify the partition was created
lsblk

# Format the partition
sudo mkfs.ext4 /dev/vdb1

# Create a mount point and mount the partition
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block

# Set permissions
sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block

# Verify it's mounted
df -h
```

## 5. Mounting Object Storage to Local File System

### 5.1 Create a Mount Point

```bash
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object
```

### 5.2 Mount the Object Store

```bash
rclone mount chi_tacc:object-persist-project-50 /mnt/object --read-only --allow-other --daemon
```

### 5.3 Verify the Mount

```bash
ls /mnt/object
```

You should see the MIND dataset directories (training, validation, testing).

## 6. Setting Up MLflow and Monitoring Infrastructure

### 6.1 Create Directories for Persistence

```bash
sudo mkdir -p /mnt/block/minio_data
sudo mkdir -p /mnt/block/postgres_data
sudo mkdir -p /mnt/block/prometheus_data
sudo mkdir -p /mnt/block/grafana_data

sudo chown -R cc:cc /mnt/block
sudo chmod -R 777 /mnt/block/grafana_data
```

### 6.2 Create Prometheus Configuration

Create a basic `prometheus.yml` configuration file that includes scrape configuration for Prometheus self-monitoring. More advanced configurations can be added as needed.

### 6.3 Create Docker Compose File for the Infrastructure

Create a file named `docker-compose-mind-complete.yaml` that includes the following services:

- **minio**: MinIO object storage for MLflow artifacts
- **minio-create-bucket**: Creates required buckets in MinIO
- **postgres**: PostgreSQL database for MLflow metadata
- **mlflow**: MLflow tracking server
- **jupyter**: Jupyter notebook server with access to the MIND dataset
- **prometheus**: Prometheus for metrics collection
- **grafana**: Grafana for metrics visualization

The complete file can be found in the project repository.

### 6.4 Start the Infrastructure

```bash
# Get host IP
export HOST_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "Using HOST_IP: $HOST_IP"

# Start services
HOST_IP=$HOST_IP docker compose -f docker-compose-mind-complete.yaml up -d
```

### 6.5 Access the Services

- **Jupyter**: http://YOUR_IP:8888
- **MLflow**: http://YOUR_IP:8000
- **MinIO**: http://YOUR_IP:9001 (login with minio/minio123)
- **Prometheus**: http://YOUR_IP:9090
- **Grafana**: http://YOUR_IP:3000 (login with admin/admin)

## 7. Testing the Setup

### 7.1 Verify Object Storage Mount

```python
# In a Jupyter notebook
import os

# Check MIND dataset access
mind_data_dir = os.environ.get('MIND_DATA_DIR', '/mnt/MIND')
print(f"MIND Data Directory: {mind_data_dir}")
print(f"Contents: {os.listdir(mind_data_dir)}")
```

### 7.2 Test MLflow Connection

```python
# In a Jupyter notebook
import os
import mlflow

# Print the tracking URI
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# Create a test experiment
mlflow.set_experiment("test-connection")
with mlflow.start_run(run_name="connection-test"):
    mlflow.log_param("test_param", "test_value")
    mlflow.log_metric("test_metric", 1.0)
    print("Successfully logged to MLflow!")
```

## 8. Troubleshooting

### 8.1 MLflow Connection Issues

If you see `InvalidUrlException: Invalid url: http://:8000/api/2.0/mlflow/...`, the HOST_IP is not properly set. Fix it with:

```python
import os
import requests

# Get the instance IP
host_ip = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4', timeout=2).text
os.environ['MLFLOW_TRACKING_URI'] = f"http://{host_ip}:8000"
```

### 8.2 Object Storage Mount Issues

If the object storage mount is not accessible, try:

```bash
# Unmount and remount
fusermount -u /mnt/object
rclone mount chi_tacc:object-persist-project-50 /mnt/object --read-only --allow-other --daemon
```

### 8.3 Container Issues

Check container logs for problems:

```bash
docker logs mlflow
docker logs prometheus
docker logs grafana
```

## 9. Cleaning Up

When you're done with your work:

```bash
# Unmount object storage
fusermount -u /mnt/object

# Stop containers
docker compose -f docker-compose-mind-complete.yaml down

# Unmount block storage
sudo umount /mnt/block
```

## 10. References

- [Chameleon Cloud Documentation](https://chameleoncloud.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)
- [MIND Dataset Documentation](https://msnews.github.io/)
- [rclone Documentation](https://rclone.org/docs/)
