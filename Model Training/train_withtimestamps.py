import os
import zipfile
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
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dataset(args):
    """
    Setup the MIND dataset files
    """
    os.makedirs('MIND_large/train', exist_ok=True)
    os.makedirs('MIND_large/dev', exist_ok=True)
    os.makedirs('MIND_large/test', exist_ok=True)

    if not os.path.exists('MIND_large/train/behaviors.tsv'):
        logger.info("Extracting MIND large training set...")
        try:
            with zipfile.ZipFile('/mnt/MINDlarge_train.zip') as zip_ref:
                zip_ref.extractall('MIND_large/train')
            logger.info("MIND large training set extracted.")
        except zipfile.BadZipFile as e:
            logger.error(f"Error: The training set file is not a valid ZIP: {e}")
            return None

    if not os.path.exists('MIND_large/dev/behaviors.tsv'):
        logger.info("Extracting MIND large validation set...")
        try:
            with zipfile.ZipFile('/mnt/MINDlarge_dev.zip') as zip_ref:
                zip_ref.extractall('MIND_large/dev')
            logger.info("MIND large validation set extracted.")
        except zipfile.BadZipFile as e:
            logger.error(f"Error: The validation set file is not a valid ZIP: {e}")
            return None

    if not os.path.exists('MIND_large/test/behaviors.tsv'):
        logger.info("Extracting MIND large test set...")
        try:
            with zipfile.ZipFile('/mnt/MINDlarge_test.zip') as zip_ref:
                zip_ref.extractall('MIND_large/test')
            logger.info("MIND large test set extracted.")
        except zipfile.BadZipFile as e:
            logger.error(f"Error: The test set file is not a valid ZIP: {e}")
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

def parse_time(time_str):
    """Parse time string into datetime object"""
    try:
        return datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S %p')
    except ValueError:
        try:
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.warning(f"Could not parse time: {time_str}, using default")
            return datetime(2000, 1, 1)

def process_impressions_with_time(behaviors_df):
    """
    Convert impressions to user-item interactions with labels,
    preserving temporal information
    """
    user_item_pairs = []

    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']
        time_str = row['time']
        timestamp = parse_time(time_str)

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
                        'history': history,
                        'time_str': time_str,
                        'timestamp': timestamp
                    })

    # Convert to DataFrame and sort by timestamp
    interactions_df = pd.DataFrame(user_item_pairs)
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values('timestamp')

    return interactions_df

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

class TemporalTokenizer:
    def __init__(self, max_vocab_size=50000, min_freq=5):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
        self.vocab_size = 2

    def fit(self, texts):
        """Build vocabulary from texts"""
        # Only fit on training data to avoid leakage
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

        logger.info(f"Vocabulary size: {self.vocab_size}")

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

class TemporalNewsDataset(Dataset):
    def __init__(self, interactions_df, news_features, tokenizer, cutoff_time=None, max_history=20, max_title_length=30):
        """
        Dataset that respects temporal ordering

        Args:
            interactions_df: DataFrame with user-item interactions
            news_features: Dictionary of news features
            tokenizer: Tokenizer object
            cutoff_time: Datetime to filter interactions (include only before this time)
            max_history: Maximum number of history items to include
            max_title_length: Maximum length of title tokens
        """
        # Filter by time if cutoff_time is provided
        if cutoff_time is not None and not interactions_df.empty:
            self.interactions = interactions_df[interactions_df['timestamp'] <= cutoff_time].copy()
        else:
            self.interactions = interactions_df.copy()

        self.news_features = news_features
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_title_length = max_title_length

        # Create mapping from user to their history, respecting temporal order
        self.user_history_map = {}

        if not self.interactions.empty:
            # Sort by timestamp to respect temporal ordering
            sorted_df = self.interactions.sort_values('timestamp')

            for _, row in sorted_df.iterrows():
                user_id = row['user_id']
                news_id = row['news_id']
                timestamp = row['timestamp']

                if user_id not in self.user_history_map:
                    self.user_history_map[user_id] = []

                if row['label'] == 1:
                    self.user_history_map[user_id].append((news_id, timestamp))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        news_id = row['news_id']
        label = row['label']
        current_time = row['timestamp']

        # Get user history up to the impression time
        # Filter history to include only items before the current interaction
        history = [h_news_id for h_news_id, h_time in self.user_history_map.get(user_id, [])
                  if h_time < current_time]

        # Take most recent history items
        history = history[-self.max_history:] if len(history) > self.max_history else history

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

        # Convert timestamp to float to avoid collate issues with DataLoader
        timestamp_float = current_time.timestamp()

        return {
            'history_tokens': history_tokens,
            'candidate_tokens': candidate_tokens,
            'label': torch.tensor(label, dtype=torch.float),
            'timestamp_float': timestamp_float  # Use float instead of Pandas Timestamp
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

            # Calculate AUC score
            auc = roc_auc_score(all_labels, all_preds)
            self.log('test_auc', auc)

            # Calculate accuracy
            preds_binary = (all_preds >= 0.5).astype(int)
            accuracy = (preds_binary == all_labels).mean()
            self.log('test_accuracy', accuracy)

            # Print test metrics directly
            print(f"\nTest Results:")
            print(f"Test AUC: {auc:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")

            # Reset for next test
            self.test_preds = []
            self.test_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def temporal_evaluation(model, test_interactions, news_features, tokenizer, device='cpu'):
    """
    Perform temporal evaluation by days

    For each day, predict impressions on that day using only the history before that day
    """
    model.eval()
    model.to(device)

    # Group interactions by day
    test_interactions['date'] = test_interactions['timestamp'].dt.date
    days = sorted(test_interactions['date'].unique())

    results = []
    all_preds = []
    all_labels = []

    for day in tqdm(days, desc="Evaluating by days"):
        # Get interactions for this day
        day_interactions = test_interactions[test_interactions['date'] == day]

        # Create dataset with history cutoff at beginning of day
        cutoff_time = datetime.combine(day, datetime.min.time())
        day_dataset = TemporalNewsDataset(
            test_interactions,
            news_features,
            tokenizer,
            cutoff_time=cutoff_time
        )

        # Create dataloader
        day_loader = DataLoader(
            day_dataset,
            batch_size=64,
            shuffle=False
        )

        # Collect predictions
        day_preds = []
        day_labels = []

        with torch.no_grad():
            for batch in day_loader:
                history_tokens = batch['history_tokens'].to(device)
                candidate_tokens = batch['candidate_tokens'].to(device)
                labels = batch['label']

                scores = model(history_tokens, candidate_tokens)

                day_preds.append(scores.cpu())
                day_labels.append(labels)

        # Calculate AUC for this day
        if day_preds:
            day_preds_concat = torch.cat(day_preds).numpy()
            day_labels_concat = torch.cat(day_labels).numpy()

            try:
                day_auc = roc_auc_score(day_labels_concat, day_preds_concat)
                results.append({
                    'day': day,
                    'auc': day_auc,
                    'num_impressions': len(day_interactions),
                    'preds': day_preds_concat,
                    'labels': day_labels_concat
                })
                logger.info(f"Day {day}: AUC = {day_auc:.4f} (n={len(day_interactions)})")

                # Add to overall collection
                all_preds.append(torch.cat(day_preds))
                all_labels.append(torch.cat(day_labels))

            except ValueError:
                # This can happen if all labels are the same
                logger.warning(f"Could not calculate AUC for day {day}")

    # Overall stats
    if all_preds and all_labels:
        try:
            overall_auc = roc_auc_score(
                torch.cat(all_labels).numpy(),
                torch.cat(all_preds).numpy()
            )
            logger.info(f"Overall temporal evaluation AUC: {overall_auc:.4f}")
            return results, overall_auc
        except ValueError:
            logger.warning("Could not calculate overall AUC")
            return results, None
    else:
        logger.warning("No predictions collected during temporal evaluation")
        return results, None

def recommend_news_temporal(model, user_id, history, candidate_news_ids, news_features, tokenizer,
                          current_time, top_k=5, device='cpu'):
    """
    Recommend news articles to a user at a specific time point,
    only considering the history before that time
    """
    model.eval()
    model.to(device)

    max_history = 20

    # Filter history to include only items before the current time
    filtered_history = [h_id for h_id, h_time in history if h_time < current_time]

    if not filtered_history:
        print(f"WARNING: No filtered history for user {user_id} at time {current_time}")
        # If no history, return random recommendations from candidates
        import random
        if len(candidate_news_ids) <= top_k:
            return candidate_news_ids
        else:
            return random.sample(candidate_news_ids, top_k)

    # Take most recent history items
    filtered_history = filtered_history[-max_history:] if len(filtered_history) > max_history else filtered_history

    # Pad history if needed
    if len(filtered_history) < max_history:
        filtered_history += ['PAD'] * (max_history - len(filtered_history))

    # Debug history titles
    print("User's recent history:")
    for i, h_news_id in enumerate(filtered_history[:5]):  # Show top 5 history items
        if h_news_id != 'PAD':
            title = news_features.get(h_news_id, {}).get('title', 'Unknown')
            category = news_features.get(h_news_id, {}).get('category', 'Unknown')
            print(f"  History item {i+1}: [{category}] {title}")

    history_tokens_list = []
    for h_news_id in filtered_history:
        title = news_features.get(h_news_id, {}).get('title', '') if h_news_id != 'PAD' else ''
        tokens = tokenizer.tokenize(title)
        history_tokens_list.append(tokens)

    # Process candidates in batches to avoid memory issues
    batch_size = 100
    all_scores = []
    all_news_ids = []

    for i in range(0, len(candidate_news_ids), batch_size):
        batch_news_ids = candidate_news_ids[i:i+batch_size]

        history_tokens = torch.stack(history_tokens_list).unsqueeze(0).repeat(len(batch_news_ids), 1, 1).to(device)

        candidate_tokens_list = []
        for news_id in batch_news_ids:
            title = news_features.get(news_id, {}).get('title', '')
            candidate_tokens = tokenizer.tokenize(title)
            candidate_tokens_list.append(candidate_tokens)

        candidate_tokens = torch.stack(candidate_tokens_list).to(device)

        with torch.no_grad():
            scores = model(history_tokens, candidate_tokens).cpu().numpy()

        all_scores.extend(scores)
        all_news_ids.extend(batch_news_ids)

    # Sort by score and return top-k
    candidate_scores = list(zip(all_news_ids, all_scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_news = [news_id for news_id, score in candidate_scores[:top_k]]

    return recommended_news

def main(args):
    # Setup dataset
    mlflow.pytorch.autolog()
    mlflow.set_experiment("news_recommendation_mind_large_temporal")

    data_dirs = setup_dataset(args)
    if not data_dirs:
        return

    # Load data
    logger.info("Loading and processing data...")

    # Load training data
    train_behaviors = load_behaviors(os.path.join(data_dirs['train_dir'], 'behaviors.tsv'))
    train_news = load_news(os.path.join(data_dirs['train_dir'], 'news.tsv'))

    # Load validation data
    dev_behaviors = load_behaviors(os.path.join(data_dirs['dev_dir'], 'behaviors.tsv'))
    dev_news = load_news(os.path.join(data_dirs['dev_dir'], 'news.tsv'))

    # Load test data
    test_behaviors = load_behaviors(os.path.join(data_dirs['test_dir'], 'behaviors.tsv'))
    test_news = load_news(os.path.join(data_dirs['test_dir'], 'news.tsv'))

    # Use a subset of data for debugging if specified
    if args.debug:
        logger.info("Running in debug mode with reduced dataset")
        train_behaviors = train_behaviors.head(10000)
        dev_behaviors = dev_behaviors.head(1000)
        test_behaviors = test_behaviors.head(1000)

    # Process impressions with temporal information
    logger.info("Processing train impressions...")
    train_interactions = process_impressions_with_time(train_behaviors)

    logger.info("Processing dev impressions...")
    dev_interactions = process_impressions_with_time(dev_behaviors)

    logger.info("Processing test impressions...")
    test_interactions = process_impressions_with_time(test_behaviors)

    # Extract news features (keeping separate for each dataset to avoid leakage)
    logger.info("Extracting news features...")
    train_news_features = extract_news_features(train_news)
    dev_news_features = extract_news_features(dev_news)
    test_news_features = extract_news_features(test_news)

    # Initialize tokenizer only on training data to avoid leakage
    logger.info("Building vocabulary from training data only...")
    train_titles = [news['title'] for news in train_news_features.values() if news['title']]
    tokenizer = TemporalTokenizer(max_vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.fit(train_titles)

    # For evaluation, we need to merge news features (but keep in mind when items were published)
    # In a real system, we would track news publication dates
    all_news_features = {**train_news_features, **dev_news_features, **test_news_features}

    # Create temporal datasets
    logger.info("Creating temporal datasets...")
    # Sort by time to ensure proper splitting
    all_interactions = pd.concat([train_interactions, dev_interactions], ignore_index=True)
    if not all_interactions.empty:
        all_interactions = all_interactions.sort_values('timestamp')

        # For validation, use a time-based split
        # Determine time threshold (e.g., last 20% of data)
        time_sorted = all_interactions['timestamp'].sort_values().reset_index(drop=True)
        split_idx = int(len(time_sorted) * 0.8)
        split_time = time_sorted.iloc[split_idx]

        logger.info(f"Time-based split threshold: {split_time}")

        # Split interactions based on time
        train_temporal = all_interactions[all_interactions['timestamp'] < split_time]
        val_temporal = all_interactions[all_interactions['timestamp'] >= split_time]

        logger.info(f"Training set: {len(train_temporal)} interactions")
        logger.info(f"Validation set: {len(val_temporal)} interactions")

        # Create the datasets
        train_dataset = TemporalNewsDataset(
            train_temporal,
            all_news_features,  # Use all news, but the dataset will filter by time
            tokenizer,
            max_history=args.max_history
        )

        val_dataset = TemporalNewsDataset(
            val_temporal,
            all_news_features,
            tokenizer,
            max_history=args.max_history
        )

        if not test_interactions.empty:
            test_dataset = TemporalNewsDataset(
                test_interactions,
                all_news_features,
                tokenizer,
                max_history=args.max_history
            )
        else:
            test_dataset = None

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            test_loader = None

        # Initialize model
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
            min_delta=0.001,
            patience=3,
            verbose=True,
            mode='max'
        )

        # Setup logger
        tb_logger = TensorBoardLogger("lightning_logs", name="news_recommendation_temporal")

        if not args.eval_only:
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                devices=args.num_gpus if torch.cuda.is_available() else None,
                accelerator='gpu' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu',
                strategy='ddp' if args.num_gpus > 1 else 'auto',
                callbacks=[checkpoint_callback, early_stop_callback],
                logger=tb_logger,
                log_every_n_steps=50,
                accumulate_grad_batches=args.accumulate_grad_batches,
                precision=16 if args.use_fp16 and torch.cuda.is_available() else 32
            )

            # Train model
            logger.info("Training model...")
            trainer.fit(model, train_loader, val_loader)

            # Test model if test data is available
            if test_loader:
                test_results = trainer.test(model, test_loader)
                if test_results and len(test_results) > 0:
                    print("\nTEST ACCURACY RESULTS:")
                    for metric_name, metric_value in test_results[0].items():
                        print(f"{metric_name}: {metric_value:.4f}")
        else:
            # Load best model for evaluation
            checkpoint_path = args.checkpoint or 'checkpoints/best_model.ckpt'
            if os.path.exists(checkpoint_path):
                model = NewsRecommendationModel.load_from_checkpoint(checkpoint_path)

                # Initialize trainer for testing only
                trainer = pl.Trainer(
                    devices=args.num_gpus if torch.cuda.is_available() else None,
                    accelerator='gpu' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu',
                    strategy='ddp' if args.num_gpus > 1 else 'auto',
                    precision=16 if args.use_fp16 and torch.cuda.is_available() else 32
                )

                if test_loader:
                    test_results = trainer.test(model, test_loader)
                    if test_results and len(test_results) > 0:
                        print("\nTEST ACCURACY RESULTS:")
                        for metric_name, metric_value in test_results[0].items():
                            print(f"{metric_name}: {metric_value:.4f}")
            else:
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return

        # Perform temporal evaluation by days
        if not val_temporal.empty:
            logger.info("Performing temporal evaluation...")
            eval_results, overall_auc = temporal_evaluation(
                model,
                val_temporal,
                all_news_features,
                tokenizer,
                device=args.device
            )

            if overall_auc:
                logger.info(f"Overall temporal AUC: {overall_auc:.4f}")

                # Plot AUC over time
                if eval_results:
                    days = [r['day'] for r in eval_results]
                    aucs = [r['auc'] for r in eval_results]
                    counts = [r['num_impressions'] for r in eval_results]

                    plt.figure(figsize=(12, 6))
                    plt.subplot(2, 1, 1)
                    plt.plot(days, aucs, 'o-', label='Daily AUC')
                    plt.axhline(y=overall_auc, color='r', linestyle='--', label=f'Overall AUC: {overall_auc:.4f}')
                    plt.title('AUC Score Over Time')
                    plt.ylabel('AUC')
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(2, 1, 2)
                    plt.bar(days, counts)
                    plt.title('Number of Impressions by Day')
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.savefig('temporal_evaluation.png')

        # Example recommendation
        logger.info("\nGenerating example recommendations...")
        print("\nGENERATING EXAMPLE RECOMMENDATIONS:")

        # Use the final model for recommendations
        if not args.eval_only and 'checkpoint_callback' in locals() and checkpoint_callback.best_model_path:
            model = NewsRecommendationModel.load_from_checkpoint(checkpoint_callback.best_model_path)

        model = model.to(args.device)

        # Select a few users from validation set for demonstration
        if not val_temporal.empty:
            # Find users who have both training history and validation data with clicks
            print("Looking for users with history in training and clicks in validation...")

            # First find users with clicks in validation
            val_clicks = val_temporal[val_temporal['label'] == 1]
            val_clickers = val_clicks['user_id'].unique()
            print(f"Found {len(val_clickers)} users with clicks in validation")

            # For each user with clicks in validation, check for history in training
            user_stats = []
            for user_id in val_clickers:
                # Count clicks in training
                train_clicks = train_temporal[
                    (train_temporal['user_id'] == user_id) &
                    (train_temporal['label'] == 1)
                ].shape[0]

                # Count clicks in validation
                val_user_clicks = val_clicks[val_clicks['user_id'] == user_id].shape[0]

                if train_clicks > 0:
                    user_stats.append({
                        'user_id': user_id,
                        'train_clicks': train_clicks,
                        'val_clicks': val_user_clicks
                    })

            print(f"Found {len(user_stats)} users with clicks in both training and validation")

            # Sort by total clicks (training + validation)
            user_stats.sort(key=lambda x: x['train_clicks'] + x['val_clicks'], reverse=True)

            # Take top users
            sample_users = [u['user_id'] for u in user_stats[:10]]

            if not sample_users:
                # Fallback to users with most validation interactions
                val_user_counts = val_temporal.groupby('user_id').size().reset_index(name='count')
                val_user_counts = val_user_counts.sort_values('count', ascending=False)
                sample_users = val_user_counts.head(5)['user_id'].tolist()
                print("Using users with most validation interactions but no training history")

            print(f"Selected {len(sample_users)} users for recommendation examples")

            # For each sample user, generate recommendations with a focus on clicked items
            for i, user_id in enumerate(sample_users):
                if i >= 5:  # Limit to 5 users
                    break

                # Get user's interactions from each set
                user_train = train_temporal[train_temporal['user_id'] == user_id]
                user_val = val_temporal[val_temporal['user_id'] == user_id]

                # Get user's clicked items in validation
                user_val_clicks = user_val[user_val['label'] == 1]

                print(f"\nUser {user_id} has {len(user_train)} interactions in training ({user_train[user_train['label']==1].shape[0]} clicks)")
                print(f"User {user_id} has {len(user_val)} interactions in validation ({user_val_clicks.shape[0]} clicks)")

                # Skip users with no validation clicks
                if user_val_clicks.empty:
                    print(f"No validation clicks for user {user_id}")
                    continue

                # For each click in validation
                success = False
                for click_idx in range(min(3, len(user_val_clicks))):
                    click_row = user_val_clicks.iloc[click_idx]
                    click_time = click_row['timestamp']
                    clicked_news_id = click_row['news_id']

                    import datetime
                    recommend_time = click_time - datetime.timedelta(seconds=1)

                    # Get user history up to this time
                    user_history = []
                    for _, row in user_train[
                        (user_train['label'] == 1) &
                        (user_train['timestamp'] < recommend_time)
                    ].iterrows():
                        user_history.append((row['news_id'], row['timestamp']))

                    # Sort history by time
                    user_history.sort(key=lambda x: x[1])
                    print(f"User history length at {recommend_time}: {len(user_history)}")

                    if len(user_history) == 0:
                        print(f"No historical clicks found for user {user_id} before {recommend_time}")
                        continue

                    # Include news the user will see in validation
                    candidate_news = [clicked_news_id]

                    # Add more validation news
                    other_val_news = user_val[
                        (user_val['timestamp'] > recommend_time) &
                        (user_val['news_id'] != clicked_news_id)
                    ]['news_id'].unique().tolist()
                    candidate_news.extend(other_val_news)

                    # Add popular news
                    popular_news = train_temporal[
                        (train_temporal['timestamp'] < recommend_time) &
                        (train_temporal['label'] == 1)
                    ]['news_id'].value_counts().head(50).index.tolist()
                    candidate_news.extend(popular_news)

                    # Add random news
                    all_news_ids = list(all_news_features.keys())
                    import random
                    random_news = random.sample(all_news_ids, min(50, len(all_news_ids)))
                    candidate_news.extend(random_news)

                    # Remove duplicates
                    candidate_news = list(set(candidate_news))

                    # Generate recommendations
                    print(f"Generating recommendations for user {user_id} just before their click at {click_time}")
                    print(f"Candidate pool size: {len(candidate_news)}")
                    print(f"Looking for clicked article: {clicked_news_id}")
                    print(f"Clicked article title: {all_news_features.get(clicked_news_id, {}).get('title', 'Unknown')}")

                    recommendations = recommend_news_temporal(
                        model,
                        user_id,
                        user_history,
                        candidate_news,
                        all_news_features,
                        tokenizer,
                        recommend_time,
                        top_k=10,
                        device=args.device
                    )

                    # Check if we recommended the clicked item
                    if clicked_news_id in recommendations:
                        success = True
                        rank = recommendations.index(clicked_news_id) + 1
                        print(f"SUCCESS! Recommended the article user clicked at rank {rank}/10")
                    else:
                        print(f"Did not recommend the article user actually clicked")

                    # Print recommendations
                    print(f"Recommended news for user {user_id} at {recommend_time}:")
                    for i, news_id in enumerate(recommendations[:5]):  # Show top 5
                        title = all_news_features.get(news_id, {}).get('title', 'Unknown')
                        category = all_news_features.get(news_id, {}).get('category', 'Unknown')
                        print(f"{i+1}. [{category}] {title}")

                        # Show if this is the clicked item
                        if news_id == clicked_news_id:
                            print(f"   ✓✓✓ THIS IS THE ARTICLE USER CLICKED! ✓✓✓")
                        else:
                            # Check if it was seen but not clicked
                            user_future_impressions = user_val[
                                (user_val['news_id'] == news_id) &
                                (user_val['timestamp'] > recommend_time) &
                                (user_val['label'] == 0)
                            ]

                            if not user_future_impressions.empty:
                                print(f"   User later saw but did not click")
                            else:
                                print(f"   No future interaction with this article")

                    # If we already found a success, move to next user
                    if success:
                        break

                if not success:
                    print(f"Could not find a recommendation scenario where user clicked our recommendation")

            # If no users, inform
            if len(sample_users) == 0:
                print("No users found for example recommendations")
        else:
            print("No validation data available for generating recommendations")
    else:
        logger.warning("No interactions data available. Please check dataset loading.")

def parse_args_or_use_defaults():
    """Parse command line args or use defaults when in notebook"""
    try:

        import sys
        is_notebook = 'ipykernel' in sys.modules
    except:
        is_notebook = False

    if is_notebook:
        # Use default args for notebook environment
        class Args:
            force_download = False
            debug = True
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
            num_workers = 2
            checkpoint = None

        return Args()
    else:
        # Normal argparse for command line usage
        parser = argparse.ArgumentParser(description='Temporal News Recommendation System')

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
        parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for evaluation')

        return parser.parse_args()

if __name__ == "__main__":
    args = parse_args_or_use_defaults()
    main(args)
