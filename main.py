from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import SentimentRequest, SentimentResponse
from classifier import SentimentClassifier

MODEL_DIR = "./models/best_model"
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs ONCE when the server boots up
    print("Loading model into memory...")
    ml_models["classifier"] = SentimentClassifier(MODEL_DIR)
    yield
    # This runs ONCE when the server shuts down
    ml_models.clear()


app = FastAPI(title="BERT Sentiment API", lifespan=lifespan)


@app.post("/api/v1/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    if "classifier" not in ml_models:
        raise HTTPException(status_code=503, detail="Model uninitialized.")

    # Run our fast in-memory inference engine
    return ml_models["classifier"].predict(payload.text)