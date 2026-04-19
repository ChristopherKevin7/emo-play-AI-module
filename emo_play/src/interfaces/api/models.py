from pydantic import BaseModel, validator
from typing import List


class AnalyzeRequest(BaseModel):
    images: List[str]

    @validator("images")
    def validate_images(cls, v):
        if not v:
            raise ValueError("A lista de imagens não pode estar vazia")
        return v


class Prediction(BaseModel):
    emotion: str
    score: float


class AnalyzeResponse(BaseModel):
    predictions: List[Prediction]