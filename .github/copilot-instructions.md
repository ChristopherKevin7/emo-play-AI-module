# EMO-Play - Módulo de IA (Reconhecimento de Emoções)

## Visão Geral

Módulo Python (FastAPI) responsável exclusivamente pelo **reconhecimento facial e detecção de emoções**. Recebe múltiplas imagens em base64 via JSON, analisa cada uma e retorna o top 3 emoções (média).

Suporta dois modelos: **HSEmotion** (ativo) e **DeepFace** (disponível). Variável `MODEL_STRATEGY` controla qual modelo é usado (`"hsemotion"` | `"deepface"` | `"ensemble"`).

O backend principal (.NET) consome este módulo via HTTP.

## Stack

- **Framework**: FastAPI
- **Detecção Facial**: RetinaFace (via DeepFace)
- **Análise de Emoções**: HSEmotion (principal) + DeepFace (backup)
- **Processamento de Imagem**: OpenCV + NumPy
- **Linguagem**: Python 3.10+

## Estrutura do Projeto

```
src/
  main.py                              # Entrypoint FastAPI
  infrastructure/
    ai/emotion_detector.py             # decode_base64_image(), preprocess_image(),
                                       # analyze_with_deepface(), analyze_with_hsemotion(),
                                       # analyze_emotion(), combine_models(),
                                       # aggregate_emotions()
  interfaces/
    api/
      routes.py                         # POST /api/v1/analyze
      models.py                         # AnalyzeRequest / AnalyzeResponse / Prediction
```

## Endpoint

### POST /api/v1/analyze

Recebe lista de imagens em base64 (JSON), analisa cada uma e retorna top 3 emoções (média).

**Request:**
```json
{
  "images": [
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,..."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "emotion": "happy", "score": 0.65 },
    { "emotion": "surprise", "score": 0.20 },
    { "emotion": "neutral", "score": 0.15 }
  ]
}
```

## Emoções Suportadas

`angry`, `contempt`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

## Estratégia de Modelo

Controlada pela variável `MODEL_STRATEGY` em `emotion_detector.py`:

- `"hsemotion"` — HSEmotion enet_b0_8_best_vgaf (padrão, 8 emoções)
- `"deepface"` — DeepFace com backend RetinaFace (7 emoções)
- `"ensemble"` — futuro: combina ambos via `combine_models()`

## Como Executar

```bash
cd emo_play
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

