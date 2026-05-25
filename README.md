# BERT Sentiment Analysis Microservice

A lightweight, production-ready microservice built with FastAPI that classifies text sentiment into **positive**, **neutral**, or **negative** categories using a fine-tuned DistilBERT architecture.

---

## Project Architecture

```text
/sentiment-service
│
├── train.ipynb            # Jupyter Notebook with 5-fold Stratified Cross-Validation
├── main.py                # FastAPI server routing logic
├── classifier.py          # Machine Learning inference core
├── schemas.py             # Pydantic v2 data validation schemas
├── requirements.txt       # Pinned dependency configurations
└── models/
    └── best_model/        # Output directory for the highest performing model weights
```

---

## Environment Setup

1. Ensure you have Python 3.10+ installed on your machine
2. Clone or navigate to the project directory:

   ```bash
   cd sentiment-service
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:

   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

5. Install the pinned dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Executing the Training Script

1. Launch your Jupyter environment:

   ```bash
   jupyter notebook
   ```

2. Open and execute all cells within `train.ipynb`.
3. The script will dynamically train across 5 stratified folds, log validation metrics (Accuracy, Precision, Recall, Macro F1), and programmatically preserve the absolute best performing configuration into the `./models/best_model` directory.
4. An aggregated confusion matrix heatmap will render at the conclusion of the evaluation block.

## Starting the FastAPI Service Locally

The production web server leverages FastAPI optimized to load the heavy transformer configurations into system RAM exactly once during the server lifecycle boot sequence.

To launch the ASGI server wrapper locally, execute:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Executing a Sample Verification Request

Open a secondary terminal workspace and utilize curl to fire a structured payload against the validation endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love the clean layout of this new application!"}'
```

### Expected Output Payload

```json
{
  "sentiment": "positive",
  "confidence": 0.984,
  "probabilities": {
    "positive": 0.984,
    "neutral": 0.012,
    "negative": 0.004
  }
}
```

Alternatively, navigate your web browser to http://127.0.0.1:8000/docs to interact with the auto-generated Swagger UI.
