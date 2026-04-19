# EMO-Play — Módulo de IA

Módulo Python (FastAPI) responsável pelo **reconhecimento facial e detecção de emoções** do projeto EMO-Play. Recebe múltiplas imagens em base64 via JSON, analisa cada uma e retorna o top 3 emoções (média ponderada).

Desenvolvido como parte do TCC — suporte à comunicação emocional de crianças com Transtorno do Espectro Autista (TEA).

---

## Stack

| Camada | Tecnologia |
|---|---|
| Framework web | FastAPI + Uvicorn |
| Detecção facial | RetinaFace (via DeepFace) |
| Análise de emoções | HSEmotion `enet_b0_8_best_vgaf` (principal) |
| Backup / ensemble | DeepFace |
| Processamento de imagem | OpenCV + NumPy |
| Linguagem | Python 3.10+ |

---

## Estrutura do Projeto

```
emo_play/
├── requirements.txt
├── setup.py
└── src/
    ├── main.py                          # Entrypoint FastAPI
    ├── infrastructure/
    │   └── ai/
    │       └── emotion_detector.py      # Toda a lógica de análise
    └── interfaces/
        └── api/
            ├── routes.py                # POST /api/v1/analyze
            └── models.py               # AnalyzeRequest / AnalyzeResponse
```

---

## Endpoint

### `POST /api/v1/analyze`

Recebe uma lista de imagens em base64, analisa cada uma e retorna a média das top 3 emoções.

**Request:**
```json
{
  "images": [
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,..."
  ]
}
```

> As imagens podem conter ou não o prefixo `data:image/...;base64,` — ambos os formatos são aceitos.

**Response:**
```json
{
  "predictions": [
    { "emotion": "happy",    "score": 0.65 },
    { "emotion": "surprise", "score": 0.20 },
    { "emotion": "neutral",  "score": 0.15 }
  ]
}
```

---

## Emoções Suportadas

`angry` · `contempt` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Estratégia de Modelo

Controlada pela variável `MODEL_STRATEGY` em `emotion_detector.py`:

| Valor | Descrição |
|---|---|
| `"hsemotion"` | HSEmotion `enet_b0_8_best_vgaf` — **padrão**, 8 emoções |
| `"deepface"` | DeepFace com backend RetinaFace — 7 emoções |
| `"ensemble"` | Combinação dos dois modelos *(futuro)* |

---

## Como Executar

### 1. Pré-requisitos

- Python 3.10+
- pip

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
cd emo_play
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
cp emo_play/.env.example emo_play/.env
```

Variáveis disponíveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `PORT` | `8000` | Porta do servidor |
| `ALLOWED_ORIGINS` | `*` | Origens CORS permitidas |
| `DEBUG` | `false` | Ativa logs detalhados (DEBUG level) |

### 5. Iniciar o servidor

```bash
cd emo_play
python -m uvicorn src.main:app --reload --port 8000
```

Acesse a documentação interativa em: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Notas de Compatibilidade

| Pacote | Versão testada | Observação |
|---|---|---|
| `timm` | `0.9.x` | `<1.0` obrigatório — versões ≥1.0 introduzem `conv_s2d` incompatível com o checkpoint |
| `torch` | `2.11+` | Necessário para compatibilidade com NumPy 2.x |
| `numpy` | `2.2.x` | Requer `torch ≥ 2.4` |

---

## Consumo pelo Backend Principal

Este módulo é consumido pelo backend .NET via HTTP. Exemplo de chamada:

```
POST http://localhost:8000/api/v1/analyze
Content-Type: application/json
```
