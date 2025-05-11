from flask import Flask, jsonify
from flask import request
import model_train_with_mlflow
from model_train_with_mlflow import load_behaviors
from model_train_with_mlflow import load_news
from model_train_with_mlflow import extract_news_features
from model_train_with_mlflow import SimpleTokenizer
from model_train_with_mlflow import NewsRecommendationModel
from model_train_with_mlflow import recommend_news
import pandas as pd
import os
import json     
import torch
from flask_cors import CORS
app = Flask(__name__)
# Enable CORS for the Flask app
CORS(app)
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Real-Time News Recommendation System API!"})


@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Parse the user ID from the request
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        
        # Call the test_model_with_pth_weights function
        recommendations = test_model_with_pth_weights(
            model, tokenizer, behaviors_data, news_data, news_features, user_id=user_id
        )
        
        if recommendations is None:
            return jsonify({"error": f"No recommendations found for user {user_id}"}), 404
        
        return jsonify({"user_id": user_id, "recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
        # Parse the user ID and news ID from the request
        # Ensure the feedback directory and file exist
        feedback_dir = "feedback_data"
        feedback_file = os.path.join(feedback_dir, "feedback.csv")

        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)

        if not os.path.exists(feedback_file):
            # Create the CSV file with headers if it doesn't exist
            with open(feedback_file, mode='w') as file:
                file.write("user_id,news_id,feedback\n")

        # Append the feedback to the CSV file
        data = request.get_json()
        user_id = data.get('user_id')
        news_id = data.get('news_id')
        feedback = data.get('feedback')
        print("data",user_id, news_id, feedback)
        with open(feedback_file, mode='a') as file:
            file.write(f"{user_id},{news_id},{feedback}\n")
        if not user_id or not news_id:
            return jsonify({"error": "User ID and News ID are required"}), 400
        
        # Log or process the "not like" feedback
        # For example, save it to a database or file for further analysis
        feedback = {"user_id": user_id, "news_id": news_id, "feedback": feedback}
        print(f"Received feedback: {feedback}")
        
        # Optionally, you can implement logic to adjust recommendations based on feedback
        
        return jsonify({"message": "Feedback received successfully"}), 200
    

def load_data():
    """
    Load behaviors and news data at the start of the application.
    This ensures the data is available for inference without reloading.
    """
    global behaviors_data, news_data, news_features, tokenizer

    try:
        # Load behaviors and news data
        print("Loading behaviors and news data...")
        behaviors_data = {
            "dev": load_behaviors('MINDlarge_dev/behaviors.tsv'),
            "test": load_behaviors('MINDlarge_test/behaviors.tsv')
        }
        
        print(behaviors_data["dev"].info())
        news_data = {
            "train": load_news('MINDlarge_train/news.tsv'),
            "dev": load_news('MINDlarge_dev/news.tsv'),
            "test": load_news('MINDlarge_test/news.tsv')
        }

        # Extract news features
        print("Extracting news features...")
        train_features = extract_news_features(news_data["train"])
        dev_features = extract_news_features(news_data["dev"])
        test_features = extract_news_features(news_data["test"])
        news_features = {**train_features, **dev_features, **test_features}

        # Initialize tokenizer
        print("Initializing tokenizer...")
        all_titles = [news['title'] for news in news_features.values() if news['title']]
        tokenizer = SimpleTokenizer(max_vocab_size=50000, min_freq=3)
        tokenizer.fit(all_titles)

        print("Data loading completed successfully.")
        
        # Load the model
        print("Loading the model...")
        model = NewsRecommendationModel.load_from_checkpoint('latest_checkpoint.pth')
        model.eval()
        print("Model loaded successfully.")
        return model, behaviors_data, news_data, news_features, tokenizer
    except Exception as e:
        print(f"Error during data loading: {e}")
        
    
    
def test_model_with_pth_weights(model, tokenizer, behaviors_data, news_data, news_features, user_id=None, history=None, num_recommendations=5, device='cpu'):
    """
    Test the model with .pth weights on a specific user or with custom history
    
    Args:
        model: Loaded NewsRecommendationModel
        tokenizer: Initialized tokenizer
        behaviors_data: Loaded behaviors data
        news_data: Loaded news data
        news_features: Extracted news features
        user_id: User ID from the dataset (if None, will use custom history)
        history: List of news IDs for custom history (required if user_id is None)
        num_recommendations: Number of recommendations to generate
        device: Device to run model on ('cpu' or 'cuda')
    
    Returns:
        List of recommended news items
    """
    # If user_id is provided, get their history from the dataset
    if user_id is not None:
        print(f"Looking up user {user_id}...")
        dev_behaviors = behaviors_data["dev"]
        test_behaviors = behaviors_data["test"]
        
        # Try to find user in dev or test behaviors
        print(dev_behaviors.info()) 
        user_data = dev_behaviors[dev_behaviors['user_id'] == user_id]
        if user_data.empty:
            user_data = test_behaviors[test_behaviors['user_id'] == user_id]
        
        if user_data.empty:
            print(f"User {user_id} not found in dataset")
            return None
        
        # Get user history
        user_history = user_data.iloc[0]['history'].split() if isinstance(user_data.iloc[0]['history'], str) and pd.notna(user_data.iloc[0]['history']) else []
        print(f"Found user with {len(user_history)} items in history")
    else:
        # Use provided history
        if history is None:
            print("Error: If user_id is not provided, history must be provided")
            return None
        user_history = history
        print(f"Using custom history with {len(history)} items")
    
    # Get candidate news IDs (all available news)
    candidate_news_ids = list(news_features.keys())
    
    # To speed up testing, limit the number of candidates
    max_candidates = 1000
    if len(candidate_news_ids) > max_candidates:
        print(f"Limiting candidates from {len(candidate_news_ids)} to {max_candidates} for faster processing")
        candidate_news_ids = candidate_news_ids[:max_candidates]
    else:
        print(f"Using {len(candidate_news_ids)} news items as candidates")
    
    print("Generating recommendations...")
    
    # SELF-CONTAINED RECOMMENDATION FUNCTION
    # Process user history
    max_history = 20
    history = user_history[:max_history]
    if len(history) < max_history:
        history += ['PAD'] * (max_history - len(history))
        
    # Process history titles
    history_tokens_list = []
    for h_news_id in history:
        title = news_features.get(h_news_id, {}).get('title', '') if h_news_id != 'PAD' else ''
        tokens = tokenizer.tokenize(title)
        history_tokens_list.append(tokens)
    
    history_tokens = torch.stack(history_tokens_list).unsqueeze(0).to(device)  # Add batch dimension
    
    # Process candidates and get scores
    candidate_scores = []
    
    # Process in batches for efficiency
    batch_size = 64
    for i in range(0, len(candidate_news_ids), batch_size):
        batch_news_ids = candidate_news_ids[i:i+batch_size]
        
        # Print progress
        if i % 200 == 0:
            print(f"Processing candidates {i} to {i+len(batch_news_ids)} of {len(candidate_news_ids)}")
        
        batch_tokens_list = []
        for news_id in batch_news_ids:
            title = news_features.get(news_id, {}).get('title', '')
            tokens = tokenizer.tokenize(title)
            batch_tokens_list.append(tokens)
        
        batch_tokens = torch.stack(batch_tokens_list).to(device)
        
        # We need to broadcast history_tokens to match batch_tokens
        batch_history = history_tokens.repeat(len(batch_news_ids), 1, 1)
        
        with torch.no_grad():
            try:
                scores = model(batch_history, batch_tokens).cpu().numpy()
                
                for j, news_id in enumerate(batch_news_ids):
                    candidate_scores.append((news_id, scores[j]))
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    # Sort by score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    recommended_news = [news_id for news_id, _ in candidate_scores[:num_recommendations]]
    
    # Print recommendations
    print(f"\nTop {num_recommendations} recommendations:")
    actual_recommendations = [
        {"news_id": news_id, "title": news_features.get(news_id, {}).get('title', 'Unknown')}
        for news_id in recommended_news
    ]
    
    print(actual_recommendations)
    return actual_recommendations

if __name__ == '__main__':
    model, behaviors_data, news_data, news_features, tokenizer = load_data()
    app.run(debug=True)
    # test_model_with_pth_weights(model, tokenizer, behaviors_data, news_data, news_features, user_id='U254959')
    # test_model_with_pth_weights(model, tokenizer, behaviors_data, news_data, news_features, user_id='U499841')