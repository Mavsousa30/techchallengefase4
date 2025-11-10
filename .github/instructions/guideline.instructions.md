# GUIDELINE.md — Tech Challenge Fase 4 (Copilot-Driven)

## 0) Escopo oficial (resumo do PDF)
- Construir uma aplicação de **análise de vídeo** com:
  1) reconhecimento facial
  2) análise de **emoções**
  3) **detecção de atividades**
  4) **resumo automático** do que foi detectado.
- Entregáveis: **repositório Git** com README claro, **relatório** com total de frames e **anomalias** detectadas, e **vídeo demo** (≤10 min) no YouTube; PDF final com links.

## 1) Stack sugerida (foco em produtividade com Copilot)
- **Linguagem**: Python 3.10+
- **Core de vídeo**: OpenCV (leitura, frames, draw), ffmpeg.
- **Rostos**: face_recognition ou mediapipe.
- **Emoções**: deepface ou modelo leve (PyTorch/TensorFlow).
- **Atividades**: mediapipe Pose ou modelo leve pré-treinado.
- **Resumo**: agregação de eventos + regras.
- **Persistência**: CSV/Parquet + JSON para métricas.
- **Métricas obrigatórias**: frames_total, anomalies_total.
- **Empacotamento**: Poetry ou pip + requirements.txt; Makefile/Nox para tasks.

## 2) Estrutura de repositório
```
tech-challenge-fase4/
├─ src/
│  ├─ io/
│  │  ├─ video_reader.py
│  │  └─ writer.py
│  ├─ face/
│  │  ├─ detector.py
│  │  └─ tracker.py
│  ├─ emotion/
│  │  └─ classifier.py
│  ├─ activity/
│  │  └─ recognizer.py
│  ├─ pipeline/
│  │  ├─ inference.py
│  │  └─ summarizer.py
│  ├─ metrics/
│  │  └─ reporter.py
│  ├─ utils/
│  │  ├─ boxes.py
│  │  └─ viz.py
│  └─ main.py
├─ models/
├─ data/
│  └─ input_video/
├─ outputs/
│  ├─ logs/
│  ├─ frames/
│  ├─ metrics.json
│  └─ report.md
├─ tests/
├─ notebooks/
├─ README.md
├─ requirements.txt
├─ Makefile
└─ .github/
   ├─ workflows/ci.yml
   └─ ISSUE_TEMPLATE.md
```

## 3) Setup rápido
```
opencv-python
numpy
pandas
deepface
face-recognition
torch
torchvision
mediapipe
tqdm
pydantic
```

## 4) Blueprint de implementação (por módulo)
### 4.1 VideoReader
Classe iterável com fps(), frame_count(), tratamento de erros.

### 4.2 FaceDetector
Detecta rostos e landmarks.

### 4.3 EmotionClassifier
Usa deepface para classificar emoções.

### 4.4 ActivityRecognizer
Detecta atividades simples via mediapipe Pose.

### 4.5 AnomalyDetection
Detecta movimentos atípicos (z-score).

### 4.6 Inference Pipeline
Orquestra módulos e grava vídeo anotado.

### 4.7 Summarizer
Gera report.md com frames_total e anomalies_total.

### 4.8 Reporter
Salva métricas em JSON/Markdown.

## 5) Padrões de código
- Docstrings, type hints, Pytest, Lint (ruff/flake8).
- Logs em outputs/logs.
- CI com GitHub Actions.

## 6) Prompts Copilot
- Gere classes com docstrings e type hints.
- Crie funções pequenas e bem nomeadas.
- Use comentários para contextualizar propósito da função.

## 7) README checklist
- Título, descrição, instalação, execução, métricas.
- report.md e metrics.json gerados automaticamente.
- Licenças e limitações.

## 8) Roteiro do vídeo (≤10 min)
1. Objetivo + pipeline
2. Execução do script
3. Demonstração do vídeo anotado
4. Relatório e métricas
5. Considerações finais

## 9) Sprints sugeridas
- D0–D1: estrutura e vídeo base
- D2–D3: faces e emoções
- D4–D5: atividades e anomalias
- D6: pipeline e resumo
- D7: testes, CI, demo

## 10) Definition of Done
✔ Reconhecimento facial
✔ Emoções
✔ Atividades
✔ Resumo automático
✔ frames_total e anomalies_total
✔ README e vídeo publicados

## 11) Definition of Done
- Sempre atualizar README.md com progresso.
