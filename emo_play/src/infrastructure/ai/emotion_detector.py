import base64
import functools
import logging
import re
import time
from typing import Dict, List

import cv2
import numpy as np
import torch
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer

# PyTorch ≥ 2.6 usa weights_only=True por padrão, mas o checkpoint do
# HSEmotion contém objetos completos (não só state_dict).  Forçamos
# weights_only=False apenas enquanto carregamos o modelo.
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)
DETECTOR_BACKEND = "retinaface"
TOP_N = 3

# ---------------------------------------------------------------------------
# Estratégia ativa: "hsemotion" | "deepface" | "ensemble"
# ---------------------------------------------------------------------------
MODEL_STRATEGY = "hsemotion"

# Mapeamento HSEmotion → nomes padronizados (mesmos do DeepFace)
_HSEMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]
_HSEMOTION_MAP = {
    "Anger": "angry",
    "Contempt": "contempt",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}

# Singleton do modelo HSEmotion
_hsemotion_model: HSEmotionRecognizer | None = None


def _get_hsemotion_model() -> HSEmotionRecognizer:
    global _hsemotion_model
    if _hsemotion_model is None:
        logger.info("[hsemotion] Carregando modelo enet_b0_8_best_vgaf…")
        # Temporariamente aplica patch para weights_only=False
        torch.load = _patched_torch_load
        try:
            _hsemotion_model = HSEmotionRecognizer(
                model_name="enet_b0_8_best_vgaf", device="cpu"
            )
        finally:
            torch.load = _original_torch_load
        logger.info("[hsemotion] Modelo carregado com sucesso")
    return _hsemotion_model


# ── decode / preprocess ────────────────────────────────────────────────────


def decode_base64_image(data: str) -> np.ndarray:
    """Remove prefixo data URI e decodifica base64 para imagem OpenCV (BGR)."""
    prefix_match = re.match(r"^(data:image/\w+;base64,)", data)
    if prefix_match:
        logger.info("[decode] Prefixo encontrado: %s", prefix_match.group(1))
    else:
        logger.warning("[decode] Nenhum prefixo data URI — assumindo base64 puro")

    raw = re.sub(r"^data:image/\w+;base64,", "", data)
    logger.info("[decode] Base64 length: %d chars", len(raw))

    image_bytes = base64.b64decode(raw)
    logger.info("[decode] Bytes decodificados: %d", len(image_bytes))

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("[decode] cv2.imdecode retornou None")
        raise ValueError("Não foi possível decodificar a imagem")

    logger.info("[decode] Imagem decodificada: shape=%s", img.shape)
    return img


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Redimensiona a imagem para TARGET_SIZE."""
    logger.info("[preprocess] Shape original: %s", img.shape)
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    logger.info("[preprocess] Shape final: %s", resized.shape)
    return resized


# ── modelos ────────────────────────────────────────────────────────────────


def analyze_with_deepface(img: np.ndarray) -> Dict[str, float]:
    """Analisa emoções usando DeepFace + RetinaFace."""
    logger.info("[deepface] Iniciando análise — shape=%s, backend=%s", img.shape, DETECTOR_BACKEND)

    result = DeepFace.analyze(
        img,
        actions=["emotion"],
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
    )

    analysis = result[0] if isinstance(result, list) else result
    raw = analysis["emotion"]
    emotions = {k: round(v / 100.0, 4) for k, v in raw.items()}

    logger.info("[deepface] Emoções: %s", emotions)
    return emotions


def analyze_with_hsemotion(img: np.ndarray) -> Dict[str, float]:
    """Analisa emoções usando HSEmotion (enet_b0_8_best_vgaf)."""
    logger.info("[hsemotion] Iniciando análise — shape=%s", img.shape)

    # 1. Detectar rosto com RetinaFace (via DeepFace)
    try:
        faces = DeepFace.extract_faces(
            img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if faces:
            face_img = faces[0]["face"]  # float 0-1, RGB
            face_img = (face_img * 255).astype(np.uint8)
            logger.info("[hsemotion] Rosto extraído: shape=%s", face_img.shape)
        else:
            face_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logger.warning("[hsemotion] Nenhum rosto detectado — usando imagem completa")
    except Exception as e:
        logger.warning("[hsemotion] Falha na detecção facial (%s) — usando imagem completa", e)
        face_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Predição
    model = _get_hsemotion_model()
    emotion_str, scores = model.predict_emotions(face_img, logits=False)

    logger.info("[hsemotion] Emoção predita: %s", emotion_str)
    logger.info("[hsemotion] Scores brutos: %s", scores)

    # 3. Montar dict normalizado
    emotions: Dict[str, float] = {}
    for label, score in zip(_HSEMOTION_LABELS, scores):
        key = _HSEMOTION_MAP[label]
        emotions[key] = round(float(score), 4)

    logger.info("[hsemotion] Emoções normalizadas: %s", emotions)
    return emotions


# ── orquestrador ───────────────────────────────────────────────────────────


def analyze_emotion(img: np.ndarray) -> Dict[str, float]:
    """Seleciona o modelo conforme MODEL_STRATEGY e retorna emoções."""
    start = time.time()
    logger.info("[analyze] Estratégia: %s", MODEL_STRATEGY)

    if MODEL_STRATEGY == "deepface":
        result = analyze_with_deepface(img)
    elif MODEL_STRATEGY == "hsemotion":
        result = analyze_with_hsemotion(img)
    elif MODEL_STRATEGY == "ensemble":
        pred1 = analyze_with_deepface(img)
        pred2 = analyze_with_hsemotion(img)
        result = combine_models(pred1, pred2)
    else:
        raise ValueError(f"Estratégia desconhecida: {MODEL_STRATEGY}")

    elapsed = time.time() - start
    logger.info("[analyze] Concluído em %.3fs", elapsed)
    return result


def combine_models(
    pred1: Dict[str, float], pred2: Dict[str, float]
) -> Dict[str, float]:
    """Futuro: média ponderada ou votação entre modelos."""
    # TODO: implementar lógica de ensemble
    return pred1


# ── agregação ──────────────────────────────────────────────────────────────


def aggregate_emotions(all_emotions: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Calcula a média das emoções de múltiplas imagens e retorna top N."""
    n = len(all_emotions)
    logger.info("[aggregate] Agregando resultados de %d imagens", n)

    merged: Dict[str, float] = {}
    for emotions in all_emotions:
        for emotion, score in emotions.items():
            merged[emotion] = merged.get(emotion, 0.0) + score

    averaged = {k: round(v / n, 4) for k, v in merged.items()}
    logger.info("[aggregate] Médias: %s", averaged)

    top = sorted(averaged.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    predictions = [{"emotion": emotion, "score": score} for emotion, score in top]
    logger.info("[aggregate] Top %d: %s", TOP_N, predictions)

    return predictions
