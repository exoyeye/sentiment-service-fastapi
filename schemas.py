from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., description="The raw sentence to analyze.")


class ProbabilityBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: ProbabilityBreakdown
