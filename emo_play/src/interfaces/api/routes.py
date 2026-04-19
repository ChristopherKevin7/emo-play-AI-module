import logging

from fastapi import APIRouter, HTTPException

from ...infrastructure.ai.emotion_detector import (
    aggregate_emotions,
    analyze_emotion,
    decode_base64_image,
    preprocess_image,
)
from .models import AnalyzeRequest, AnalyzeResponse, Prediction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    logger.info("[route] Requisição recebida - %d imagens", len(request.images))

    all_emotions = []

    for idx, img_b64 in enumerate(request.images):
        logger.info("[route] Processando imagem %d/%d (%d chars)", idx + 1, len(request.images), len(img_b64))

        try:
            img = decode_base64_image(img_b64)
        except Exception as e:
            logger.error("[route] Falha ao decodificar imagem %d: %s", idx + 1, e)
            raise HTTPException(status_code=400, detail=f"Imagem {idx + 1} inválida: {e}")

        img = preprocess_image(img)

        try:
            emotions = analyze_emotion(img)
        except Exception as e:
            logger.error("[route] Erro na análise da imagem %d: %s", idx + 1, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem {idx + 1}")

        all_emotions.append(emotions)

    predictions = aggregate_emotions(all_emotions)

    response = AnalyzeResponse(
        predictions=[Prediction(**p) for p in predictions]
    )
    logger.info("[route] Resposta final: %s", response.model_dump())
    return response