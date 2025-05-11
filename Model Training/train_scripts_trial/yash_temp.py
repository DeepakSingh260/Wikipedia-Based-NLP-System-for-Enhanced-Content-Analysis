def test_model_with_pth_weights_debug(model_weights_path, user_id=None, history=None, num_recommendations=5, device='cpu'):
    """
    Test the model with .pth weights with a self-contained recommendation function for debugging
    
    Args:
        model_weights_path: Path to the saved model weights (.pth file)
        user_id: User ID from the dataset (if None, will use custom history)
        history: List of news IDs for custom history (required if user_id is None)
        num_recommendations: Number of recommendations to generate
        device: Device to run model on ('cpu' or 'cuda')
    
    Returns:
        List of recommended news items
    """
    # Load the whole model
    print(f"Loading model from {model_weights_path}")
    model = torch.load(model_weights_path, map_location=torch.device(device))
    model = model.to(device)
    model.eval()
    
    # Load dataset components
    print("Loading dataset components...")
    train_news = load_news('MIND_large/train/news.tsv')
    dev_news = load_news('MIND_large/dev/news.tsv')
    test_news = load_news('MIND_large/test/news.tsv')
    
    # Extract news features
    train_news_features = extract_news_features(train_news)
    dev_news_features = extract_news_features(dev_news)
    test_news_features = extract_news_features(test_news)
    
    # Combine news features
    all_news_features = {**train_news_features, **dev_news_features, **test_news_features}
    
    # Get all news titles for tokenizer training
    all_titles = [news['title'] for news in all_news_features.values() if news['title']]
    
    # Initialize tokenizer
    vocab_size = model.news_encoder.embedding.num_embeddings  # Get vocab size from model
    tokenizer = SimpleTokenizer(max_vocab_size=vocab_size, min_freq=3)
    tokenizer.fit(all_titles)
    
    # If user_id is provided, get their history from the dataset
    if user_id is not None:
        print(f"Looking up user {user_id}...")
        dev_behaviors = load_behaviors('MIND_large/dev/behaviors.tsv')
        test_behaviors = load_behaviors('MIND_large/test/behaviors.tsv')
        
        # Try to find user in dev or test behaviors
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
    candidate_news_ids = list(all_news_features.keys())
    
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
        title = all_news_features.get(h_news_id, {}).get('title', '') if h_news_id != 'PAD' else ''
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
            title = all_news_features.get(news_id, {}).get('title', '')
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
    for i, (news_id, score) in enumerate(candidate_scores[:num_recommendations]):
        title = all_news_features.get(news_id, {}).get('title', 'Unknown')
        category = all_news_features.get(news_id, {}).get('category', 'Unknown')
        print(f"{i+1}. [{category}] {title} (Score: {score:.4f})")
    
    return recommended_news