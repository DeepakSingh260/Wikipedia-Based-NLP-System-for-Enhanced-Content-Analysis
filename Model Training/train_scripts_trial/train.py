import os
import zipfile
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import mlflow

def setup_dataset(args):
    """
    Setup the MIND dataset files
    """
    # Create directories if they don't exist
    os.makedirs('MIND_large/train', exist_ok=True)
    os.makedirs('MIND_large/dev', exist_ok=True)
    os.makedirs('MIND_large/test', exist_ok=True)

    # Extract training set if needed
    if not os.path.exists('MIND_large/train/behaviors.tsv'):
        print("Extracting MIND large training set...")
        try:
            with zipfile.ZipFile('Dataset/MINDlarge_train.zip') as zip_ref:
                zip_ref.extractall('MIND_large/train')
            print("MIND large training set extracted.")
        except zipfile.BadZipFile as e:
            print(f"Error: The training set file is not a valid ZIP: {e}")
            return None

    # Extract validation set if needed
    if not os.path.exists('MIND_large/dev/behaviors.tsv'):
        print("Extracting MIND large validation set...")
        try:
            with zipfile.ZipFile('Dataset/MINDlarge_dev.zip') as zip_ref:
                zip_ref.extractall('MIND_large/dev')
            print("MIND large validation set extracted.")
        except zipfile.BadZipFile as e:
            print(f"Error: The validation set file is not a valid ZIP: {e}")
            return None

    # Extract test set if needed
    if not os.path.exists('MIND_large/test/behaviors.tsv'):
        print("Extracting MIND large test set...")
        try:
            with zipfile.ZipFile('Dataset/MINDlarge_test.zip') as zip_ref:
                zip_ref.extractall('MIND_large/test')
            print("MIND large test set extracted.")
        except zipfile.BadZipFile as e:
            print(f"Error: The test set file is not a valid ZIP: {e}")
            return None

    return {
        'train_dir': 'MIND_large/train',
        'dev_dir': 'MIND_large/dev',
        'test_dir': 'MIND_large/test'
    }


def load_behaviors(file_path):
    """Load user behavior data"""
    columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    behaviors = pd.read_csv(file_path, sep='\t', names=columns)
    return behaviors


def load_news(file_path):
    """Load news data"""
    columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    news = pd.read_csv(file_path, sep='\t', names=columns)
    return news



def process_impressions(behaviors_df):
    """Convert impressions to user-item interactions with labels"""
    user_item_pairs = []

    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']
        history = row['history'].split() if isinstance(row['history'], str) and pd.notna(row['history']) else []

        if isinstance(row['impressions'], str):
            for impression in row['impressions'].split():
                parts = impression.split('-')
                if len(parts) == 2:
                    news_id, label = parts
                    user_item_pairs.append({
                        'user_id': user_id,
                        'news_id': news_id,
                        'label': int(label),
                        'history': history
                    })

    return pd.DataFrame(user_item_pairs)


def extract_news_features(news_df):
    """Extract text features from news articles"""
    news_features = {}

    for _, row in news_df.iterrows():
        news_id = row['news_id']

        category = row['category'] if pd.notna(row['category']) else ""
        subcategory = row['subcategory'] if pd.notna(row['subcategory']) else ""
        title = row['title'] if pd.notna(row['title']) else ""
        abstract = row['abstract'] if pd.notna(row['abstract']) else ""

        news_features[news_id] = {
            'category': category,
            'subcategory': subcategory,
            'title': title,
            'abstract': abstract
        }

    return news_features


class SimpleTokenizer:
    def __init__(self, max_vocab_size=50000, min_freq=5):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
        self.vocab_size = 2  # PAD and UNK

    def fit(self, texts):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            for word in text.lower().split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Filter by frequency and sort
        valid_words = [(word, freq) for word, freq in self.word_freq.items()
                       if freq >= self.min_freq]
        valid_words.sort(key=lambda x: x[1], reverse=True)

        # Build vocabulary limited by max size
        for word, _ in valid_words[:self.max_vocab_size - 2]:  # -2 for PAD and UNK
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

        print(f"Vocabulary size: {self.vocab_size}")

    def tokenize(self, text, max_length=30):
        """Convert text to token ids with padding/truncation"""
        if not text:
            return torch.tensor([0] * max_length)

        words = text.lower().split()
        tokens = [self.word2idx.get(word, 1) for word in words]  # 1 is UNK

        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))  # 0 is PAD

        return torch.tensor(tokens)


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=100, hidden_dim=128):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x is the token ids
        embedded = self.embedding(x).mean(dim=1)  # Simple mean pooling
        h1 = self.relu(self.fc1(embedded))
        out = self.fc2(h1)
        return out


class UserEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UserEncoder, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, news_embeddings):
        # news_embeddings: batch_size x history_length x hidden_dim
        attention_weights = self.attention(news_embeddings)  # batch_size x history_length x 1
        weighted_embeddings = news_embeddings * attention_weights  # Element-wise multiplication
        user_embedding = torch.sum(weighted_embeddings, dim=1)  # Sum along history dimension
        return user_embedding



class NewsDataset(Dataset):
    def __init__(self, interactions_df, news_features, tokenizer, max_history=20, max_title_length=30):
        self.interactions = interactions_df
        self.news_features = news_features
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_title_length = max_title_length

        # Create user to history mapping
        self.user_history = {}
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            if user_id not in self.user_history:
                self.user_history[user_id] = row['history']

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        news_id = row['news_id']
        label = row['label']

        # Get history
        history = self.user_history.get(user_id, [])[:self.max_history]

        # Pad history if needed
        if len(history) < self.max_history:
            history = history + ['PAD'] * (self.max_history - len(history))

        # Process candidate news
        candidate_title = self.news_features.get(news_id, {}).get('title', '')
        candidate_tokens = self.tokenizer.tokenize(candidate_title, self.max_title_length)

        # Process history news
        history_tokens_list = []
        for h_news_id in history:
            if h_news_id == 'PAD':
                history_title = ''
            else:
                history_title = self.news_features.get(h_news_id, {}).get('title', '')

            history_tokens = self.tokenizer.tokenize(history_title, self.max_title_length)
            history_tokens_list.append(history_tokens)

        # Stack history tokens
        history_tokens = torch.stack(history_tokens_list)

        return {
            'history_tokens': history_tokens,
            'candidate_tokens': candidate_tokens,
            'label': torch.tensor(label, dtype=torch.float)
        }


class NewsRecommendationModel(pl.LightningModule):
    def __init__(self, vocab_size=50000, embedding_dim=100, hidden_dim=128, learning_rate=0.001):
        super(NewsRecommendationModel, self).__init__()
        self.save_hyperparameters()

        # Define model components
        self.news_encoder = NewsEncoder(vocab_size, embedding_dim, hidden_dim)
        self.user_encoder = UserEncoder(hidden_dim)
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate

    def forward(self, history_news, candidate_news):
        # Encode history news
        batch_size = history_news.size(0)
        history_length = history_news.size(1)

        # Reshape for processing
        h_news = history_news.view(batch_size * history_length, -1)

        # Encode all history news at once
        h_news_embeddings = self.news_encoder(h_news)

        # Reshape back
        history_news_embeddings = h_news_embeddings.view(batch_size, history_length, -1)

        # Encode user
        user_embedding = self.user_encoder(history_news_embeddings)

        # Encode candidate news
        candidate_embedding = self.news_encoder(candidate_news)

        # Calculate score (dot product)
        score = torch.sum(user_embedding * candidate_embedding, dim=1)

        return torch.sigmoid(score)

    def training_step(self, batch, batch_idx):
        history_tokens = batch['history_tokens']
        candidate_tokens = batch['candidate_tokens']
        labels = batch['label']

        scores = self(history_tokens, candidate_tokens)
        loss = self.criterion(scores, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
      history_tokens = batch['history_tokens']
      candidate_tokens = batch['candidate_tokens']
      labels = batch['label']

      scores = self(history_tokens, candidate_tokens)
      loss = self.criterion(scores, labels)

      self.log('val_loss', loss, prog_bar=True)

      # Store predictions for AUC calculation
      if not hasattr(self, 'val_preds'):
          self.val_preds = []
          self.val_labels = []

      self.val_preds.append(scores.detach())
      self.val_labels.append(labels.detach())

      return {'loss': loss}

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_preds') and len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds).cpu().numpy()
            all_labels = torch.cat(self.val_labels).cpu().numpy()

            auc = roc_auc_score(all_labels, all_preds)
            self.log('val_auc', auc, prog_bar=True)

            # Reset for next epoch
            self.val_preds = []
            self.val_labels = []

    def test_step(self, batch, batch_idx):
        history_tokens = batch['history_tokens']
        candidate_tokens = batch['candidate_tokens']
        labels = batch['label']

        scores = self(history_tokens, candidate_tokens)
        loss = self.criterion(scores, labels)

        self.log('test_loss', loss)

        # Store predictions for AUC calculation
        if not hasattr(self, 'test_preds'):
            self.test_preds = []
            self.test_labels = []

        self.test_preds.append(scores.detach())
        self.test_labels.append(labels.detach())

        return {'loss': loss}

    def on_test_epoch_end(self):
        if hasattr(self, 'test_preds') and len(self.test_preds) > 0:
            all_preds = torch.cat(self.test_preds).cpu().numpy()
            all_labels = torch.cat(self.test_labels).cpu().numpy()

            auc = roc_auc_score(all_labels, all_preds)
            self.log('test_auc', auc)

            # Reset for next test
            self.test_preds = []
            self.test_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def recommend_news(model, user_id, user_history, candidate_news_ids, news_features, tokenizer, top_k=5, device='cpu'):
    model.eval()
    model.to(device)

    max_history = 20

    # Process user history
    history = user_history[:max_history]
    if len(history) < max_history:
        history += ['PAD'] * (max_history - len(history))

    history_tokens_list = []
    for h_news_id in history:
        title = news_features.get(h_news_id, {}).get('title', '') if h_news_id != 'PAD' else ''
        tokens = tokenizer.tokenize(title)
        history_tokens_list.append(tokens)

    history_tokens = torch.stack(history_tokens_list).unsqueeze(0).repeat(len(candidate_news_ids), 1, 1).to(device)

    candidate_tokens_list = []
    for news_id in candidate_news_ids:
        title = news_features.get(news_id, {}).get('title', '')
        candidate_tokens = tokenizer.tokenize(title)
        candidate_tokens_list.append(candidate_tokens)

    candidate_tokens = torch.stack(candidate_tokens_list).to(device)

    with torch.no_grad():
        scores = model(history_tokens, candidate_tokens).cpu().numpy()

    candidate_scores = list(zip(candidate_news_ids, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_news = [news_id for news_id, _ in candidate_scores[:top_k]]

    return recommended_news


def main(args):
    # Setup dataset
    mlflow.pytorch.autolog()

    mlflow.set_experiment("news_recommendation_mind_large")
    data_dirs = setup_dataset(args)
    if not data_dirs:
        return

    # Load data
    print("Loading and processing data...")

    # Load training data
    train_behaviors = load_behaviors(os.path.join(data_dirs['train_dir'], 'behaviors.tsv'))
    train_news = load_news(os.path.join(data_dirs['train_dir'], 'news.tsv'))

    # Load validation data
    dev_behaviors = load_behaviors(os.path.join(data_dirs['dev_dir'], 'behaviors.tsv'))
    dev_news = load_news(os.path.join(data_dirs['dev_dir'], 'news.tsv'))

    # Load test data
    test_behaviors = load_behaviors(os.path.join(data_dirs['test_dir'], 'behaviors.tsv'))
    test_news = load_news(os.path.join(data_dirs['test_dir'], 'news.tsv'))

    # Process impressions - using a subset for large dataset to save memory
    print("Processing train impressions...")
    if args.debug:
        # Use only a small subset of data for debugging
        train_behaviors = train_behaviors.head(10000)
        dev_behaviors = dev_behaviors.head(1000)
        test_behaviors = test_behaviors.head(1000)

    train_interactions = process_impressions(train_behaviors)
    dev_interactions = process_impressions(dev_behaviors)
    test_interactions = process_impressions(test_behaviors)

    # Extract news features
    print("Extracting news features...")
    train_news_features = extract_news_features(train_news)
    dev_news_features = extract_news_features(dev_news)
    test_news_features = extract_news_features(test_news)

    # Combine news features
    all_news_features = {**train_news_features, **dev_news_features, **test_news_features}

    # Get all news titles for tokenizer training
    all_titles = [news['title'] for news in all_news_features.values() if news['title']]

    # Initialize tokenizer
    print("Building vocabulary...")
    tokenizer = SimpleTokenizer(max_vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.fit(all_titles)

    # Create datasets
    print("Creating datasets...")
    train_dataset = NewsDataset(
        train_interactions,
        all_news_features,
        tokenizer,
        max_history=args.max_history
    )

    val_dataset = NewsDataset(
        dev_interactions,
        all_news_features,
        tokenizer,
        max_history=args.max_history
    )

    test_dataset = NewsDataset(
        test_interactions,
        all_news_features,
        tokenizer,
        max_history=args.max_history
    )

    # Create dataloaders with PyTorch Lightning
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize PyTorch Lightning model
    model = NewsRecommendationModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        dirpath='checkpoints/',
        filename='news-recommender-{epoch:02d}-{val_auc:.4f}',
        save_top_k=3,
        mode='max'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_auc',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='max'
    )

    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="news_recommendation")

    if not args.eval_only:
        # Initialize trainer with distributed training support

        trainer = pl.Trainer(
          max_epochs=args.epochs,
          devices=args.num_gpus if torch.cuda.is_available() else 1,
          accelerator='gpu' if args.device == 'cuda' else 'cpu',
          strategy='ddp' if args.num_gpus > 1 else 'auto',  # Use 'auto' instead of None
          callbacks=[checkpoint_callback, early_stop_callback],
          logger=logger,
          log_every_n_steps=50,
          accumulate_grad_batches=args.accumulate_grad_batches,
          precision=16 if args.use_fp16 else 32  # Enable mixed precision training
      )

        # Train model
        print("Training model...")
        trainer.fit(model, train_loader, val_loader)

        # Test model
        trainer.test(model, test_loader)
    else:
        # Load best model for evaluation
        checkpoint_path = 'checkpoints/best_model.ckpt'
        if os.path.exists(checkpoint_path):
            model = NewsRecommendationModel.load_from_checkpoint(checkpoint_path)

            # Initialize trainer for testing only
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                devices=args.num_gpus if torch.cuda.is_available() else 1,
                accelerator='gpu' if args.device == 'cuda' else 'cpu',  # Changed None to 'auto'
                strategy='ddp' if args.num_gpus > 1 else 'auto',
                precision=16 if args.use_fp16 else 32
            )
            trainer.test(model, test_loader)

 # Example recommendation
    print("\nGenerating example recommendations...")

    # Load the best model for recommendation
    if not args.eval_only and checkpoint_callback.best_model_path:
        model = NewsRecommendationModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    else:
        # Use the current model if no best model path is available
        model = model.to(args.device)

    # Debug information
    print(f"Train interactions shape: {train_interactions.shape}")
    print(f"Train interactions columns: {train_interactions.columns.tolist()}")
    print(f"Dev interactions shape: {dev_interactions.shape}")
    print(f"Dev interactions columns: {dev_interactions.columns.tolist()}")
    print(f"Test interactions shape: {test_interactions.shape}")
    print(f"Test interactions columns: {test_interactions.columns.tolist()}")

    # Try to use dev_interactions for recommendations since test might be empty
    recommendation_data = dev_interactions if test_interactions.empty or 'user_id' not in test_interactions.columns else test_interactions
    print(f"Using dataset with {len(recommendation_data)} rows for recommendations")

    if recommendation_data.empty:
        print("No interactions available for generating recommendations.")
    else:
        # Get sample users - ensure we have some
        if 'user_id' in recommendation_data.columns:
            sample_users = recommendation_data['user_id'].unique()
            print(f"Found {len(sample_users)} unique users")

            if len(sample_users) > 0:
                # Limit to first 5 users
                sample_users = sample_users[:5]

                for user_id in sample_users:
                    user_data = recommendation_data[recommendation_data['user_id'] == user_id]

                    # Debug
                    print(f"\nUser {user_id} has {len(user_data)} interactions")
                    print(f"User data columns: {user_data.columns.tolist()}")

                    # Skip if no data for this user
                    if user_data.empty:
                        print(f"No data for user {user_id}")
                        continue

                    # Ensure history exists and is properly processed
                    if 'history' in user_data.columns:
                        sample_history = user_data.iloc[0]['history']
                        print(f"User history type: {type(sample_history)}")
                        print(f"User history length: {len(sample_history) if isinstance(sample_history, list) else 'not a list'}")
                    else:
                        print("No history column found")
                        sample_history = []

                    # Get all candidate news for this user
                    if 'news_id' in user_data.columns:
                        candidate_news_ids = list(set(user_data['news_id']))
                        print(f"Found {len(candidate_news_ids)} candidate news items")
                    else:
                        print("No news_id column found")
                        continue

                    if not candidate_news_ids:
                        print(f"No candidate news items for user {user_id}")
                        continue

                    try:
                        # Get recommendations
                        recommendations = recommend_news(
                            model,
                            user_id,
                            sample_history,
                            candidate_news_ids,
                            all_news_features,
                            tokenizer,
                            top_k=5,
                            device=args.device
                        )

                        # Print recommendations
                        print(f"\nRecommended news for user {user_id}:")
                        for i, news_id in enumerate(recommendations):
                            title = all_news_features.get(news_id, {}).get('title', 'Unknown')
                            category = all_news_features.get(news_id, {}).get('category', 'Unknown')
                            print(f"{i+1}. [{category}] {title}")

                            # Print if it was clicked by user
                            if 'label' in user_data.columns:
                                user_labels = user_data[user_data['news_id'] == news_id]['label']
                                if not user_labels.empty:
                                    label = user_labels.iloc[0]
                                    print(f"   User clicked: {'Yes' if label == 1 else 'No'}")
                    except Exception as e:
                        print(f"Error generating recommendations for user {user_id}: {str(e)}")
        else:
            print("No user_id column in the interaction data")


def parse_args_or_use_defaults():
    """Parse command line args or use defaults when in notebook"""
    try:
        # Check if we're running in IPython/Jupyter
        import sys
        is_notebook = 'ipykernel' in sys.modules
    except:
        is_notebook = False

    if is_notebook:
        # Use default args for notebook environment
        class Args:
            force_download = False
            debug = True  # Set debug to True for notebook to use smaller datasets
            vocab_size = 50000
            embedding_dim = 100
            hidden_dim = 128
            max_history = 20
            min_freq = 3
            batch_size = 64
            epochs = 2
            learning_rate = 0.001
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            num_gpus = 1
            use_fp16 = False
            accumulate_grad_batches = 1
            eval_only = False
            num_workers = 2  # Lower for notebooks to prevent memory issues

        return Args()
    else:
        # Normal argparse for command line usage
        parser = argparse.ArgumentParser(description='News Recommendation System with PyTorch Lightning')

        # Data parameters
        parser.add_argument('--force_download', action='store_true', help='Force download dataset even if already exists')
        parser.add_argument('--debug', action='store_true', help='Run in debug mode with smaller dataset')

        # Model parameters
        parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary')
        parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
        parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
        parser.add_argument('--max_history', type=int, default=20, help='Maximum history length')
        parser.add_argument('--min_freq', type=int, default=3, help='Minimum word frequency')

        # Training parameters
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
        parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for training')
        parser.add_argument('--use_fp16', action='store_true', help='Use mixed precision training')
        parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate gradients over n batches')
        parser.add_argument('--eval_only', action='store_true', help='Only evaluate model')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

        return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_or_use_defaults()
    main(args)