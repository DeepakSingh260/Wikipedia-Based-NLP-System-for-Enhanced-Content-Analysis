## Serving Folder Documentation

This folder contains the code and resources required to serve the Real-Time News Recommendation System. It includes the Flask-based API, model loading, ONNX export, and feedback handling functionalities. Below is an explanation of the folder structure and the purpose of each file.

---

## Folder Structure

```
Serving/
├── __init__.py
├── app.py
├── latest_checkpoint.pth
├── model_train_with_mlflow.py
├── model.pth
├── news_recommendation_model.onnx
├── optimization.ipynb
├── user_recommend.html
├── __pycache__/
│   ├── app.cpython-311.pyc
│   ├── model_train_with_mlflow.cpython-311.pyc
│   ├── model_train_with_mlflow.cpython-312.pyc
├── feedback_data/
│   ├── feedback.csv
├── MINDlarge_dev/
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   ├── news.tsv
│   ├── relation_embedding.vec
├── MINDlarge_test/
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   ├── news.tsv
│   ├── relation_embedding.vec
├── MINDlarge_train/
│   ├── ...
```

---

## File Descriptions

### 1. app.py
This is the main file for serving the API. It uses Flask to expose endpoints for inference, feedback collection, and daily updates. Key functionalities include:
- **Endpoints**:
  - `/`: A welcome message for the API.
  - `/infer`: Accepts a user ID and returns news recommendations.
  - `/feedback`: Collects user feedback on recommended news.
- **Daily Update Task**: A background thread reloads the model and data every 24 hours.
- **Data Loading**: Loads behaviors, news data, and features at startup.
- **Model Loading**: Loads the `NewsRecommendationModel` from the latest checkpoint.

### 2. `model_train_with_mlflow.py`
This file contains the training logic for the `NewsRecommendationModel`. It includes:
- Functions to parse arguments, train the model, and save checkpoints.
- Utilities for loading behaviors, news data, and extracting features.
- The definition of the `NewsRecommendationModel` class.

### 3. `latest_checkpoint.pth`
This file contains the latest trained weights for the `NewsRecommendationModel`. It is loaded at runtime to initialize the model for inference.

### 4. `model.pth`
An alternative checkpoint file for the model. It may be used for testing or backup purposes.

### 5. `news_recommendation_model.onnx`
This is the ONNX-exported version of the `NewsRecommendationModel`. It is used for faster inference with ONNX Runtime.

### 6. `optimization.ipynb`
A Jupyter Notebook that demonstrates:
- Exporting the PyTorch model to ONNX format.
- Running inference using both PyTorch and ONNX models.
- Comparing the performance of PyTorch and ONNX models.

### 7. `user_recommend.html`
A placeholder or template file for a potential user-facing interface. It could be used to display recommendations in a web application.

### 8. `feedback_data/feedback.csv`
A CSV file that stores user feedback. Each row contains:
- `user_id`: The ID of the user providing feedback.
- `news_id`: The ID of the news item.
- `feedback`: The feedback provided by the user (e.g., "like" or "dislike").

### 9. `MINDlarge_dev/`, `MINDlarge_test/`, `MINDlarge_train/`
These folders contain the MIND dataset files used for training and evaluation:
- `behaviors.tsv`: User behaviors (e.g., clicks, impressions).
- `news.tsv`: News metadata (e.g., titles, categories).
- `entity_embedding.vec` and `relation_embedding.vec`: Pre-trained embeddings for entities and relations.

---

## Key Functionalities

### 1. **API Endpoints**
- **Inference (`/infer`)**:
  - Accepts a `user_id` in the request body.
  - Returns a list of recommended news items based on the user's history.
- **Feedback (`/feedback`)**:
  - Accepts `user_id`, `news_id`, and `feedback` in the request body.
  - Logs the feedback to `feedback_data/feedback.csv`.

### 2. **Model Export to ONNX**
The `optimization.ipynb` notebook demonstrates how to export the PyTorch model to ONNX format for faster inference. The exported model is saved as `news_recommendation_model.onnx`.

### 3. **Daily Update Task**
A background thread reloads the model and data every 24 hours to ensure the system stays up-to-date with the latest information.

### 4. **Feedback Handling**
User feedback is logged to a CSV file for further analysis. This feedback can be used to improve the recommendation system.

---

## How to Run the API

1. **Install Dependencies**:
   Ensure you have the required Python packages installed:
   ```bash
   pip install flask flask-cors torch pandas onnx onnxruntime
   ```

2. **Start the API**:
   Run the app.py file:
   ```bash
   python app.py
   ```

3. **Access the API**:
   - Visit `http://127.0.0.1:5000/` for the welcome message.
   - Use tools like Postman or `curl` to interact with the `/infer` and `/feedback` endpoints.

---

## How to Export the Model to ONNX

1. Open the `optimization.ipynb` notebook.
2. Run the cells to:
   - Define dummy inputs for the model.
   - Export the model to ONNX format using `torch.onnx.export`.
   - Save the ONNX model as `news_recommendation_model.onnx`.

---

## How to Use the ONNX Model

1. Load the ONNX model using ONNX Runtime:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("news_recommendation_model.onnx")
   ```
2. Prepare input data and run inference:
   ```python
   inputs = {
       "batch_history": batch_history.numpy(),
       "batch_tokens": batch_tokens.numpy()
   }
   outputs = session.run(None, inputs)
   print(outputs)
   ```

---

## Notes

- Ensure the `latest_checkpoint.pth` file is present in the folder for the model to load successfully.
- The ONNX model (`news_recommendation_model.onnx`) is optional but recommended for faster inference.
- Feedback data is stored in `feedback_data/feedback.csv` and can be used to improve the system over time.

---

This README provides an overview of the Serving folder and its components. For further details, refer to the code and comments within the files.
