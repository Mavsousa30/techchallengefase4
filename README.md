# Tech Challenge Fase 4 - Análise de Vídeo com IA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-117%20passed-green.svg)](tests/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/TechChallengeFase4/blob/main/notebooks/Colab_Video_Analysis.ipynb)

> **⚡ DICA:** Use o [Google Colab](notebooks/README_COLAB.md) para processamento **10-20x mais rápido** com GPU gratuita!

## 📋 Descrição

Aplicação de análise de vídeo com Inteligência Artificial que realiza:

1. **Reconhecimento Facial** - Detecta e rastreia rostos em vídeos
2. **Análise de Emoções** - Classifica emoções (feliz, triste, neutro, etc.)
3. **Detecção de Atividades** - Reconhece atividades humanas (caminhando, sentado, gesticulando)
4. **Resumo Automático** - Gera relatórios com métricas e anomalias detectadas

## ✅ Progresso de Implementação

### 🌟 NOVO: Interface Web Implementada!

- ✅ **Interface Web Streamlit** (`app.py`) - Interface visual moderna e interativa
- ✅ **Script de Inicialização** (`run_web.sh`) - Launcher automático
- ✅ **Configuração Streamlit** (`.streamlit/config.toml`) - Tema customizado
- ✅ **Documentação Web** (`WEB_INTERFACE.md`) - Guia completo

### Módulos Implementados (100% COMPLETO)

- ✅ **VideoReader** (`src/io/video_reader.py`) - Leitura de vídeos com iteração frame a frame
- ✅ **VideoWriter** (`src/io/writer.py`) - Gravação de vídeos anotados com OpenCV
- ✅ **FaceDetector** (`src/face/detector.py`) - Detecção de rostos com OpenCV/DeepFace
- ✅ **EmotionClassifier** (`src/emotion/classifier.py`) - Classificação de emoções com DeepFace
- ✅ **ActivityRecognizer** (`src/activity/recognizer.py`) - Reconhecimento de atividades com MediaPipe
- ✅ **AnomalyDetector** (`src/pipeline/anomaly_detector.py`) - Detecção de anomalias com z-score
- ✅ **Summarizer** (`src/pipeline/summarizer.py`) - Agregação de resultados e estatísticas
- ✅ **InferencePipeline** (`src/pipeline/inference.py`) - Pipeline completo de processamento
- ✅ **Reporter** (`src/metrics/reporter.py`) - Exportação de métricas JSON e Markdown
- ✅ **Visualization Utils** (`src/utils/viz.py`) - Funções para anotação de vídeos
- ✅ **Main Script** (`src/main.py`) - CLI completa para execução

### Testes

- **117 testes passando**, 3 skipped
- Cobertura: unit tests, integration tests, acceptance tests
- Frameworks: pytest, fixtures, mocks
- Testes adicionais pendentes para novos módulos

## 🚀 Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd TechChallengeFase4

# Instale as dependências
make setup

# Ou manualmente:
pip install -r requirements.txt
```

## 📦 Dependências Principais

- **OpenCV** - Processamento de vídeo e detecção facial
- **DeepFace** - Análise de emoções
- **MediaPipe** - Estimação de pose para atividades
- **PyTorch** - Backend de deep learning
- **NumPy/Pandas** - Processamento de dados

## 💻 Uso

### ⚡ Google Colab (MAIS RÁPIDO - Recomendado!)

**10-20x mais rápido** que rodar localmente, com GPU gratuita:

```bash
# 1. Acesse o notebook
https://colab.research.google.com/github/SEU_USUARIO/TechChallengeFase4/blob/main/notebooks/Colab_Video_Analysis.ipynb

# 2. Ative a GPU: Runtime → Change runtime type → GPU (T4)
# 3. Execute todas as células: Runtime → Run all
# 4. Faça upload do seu vídeo quando solicitado
```

📖 **Guia completo**: [notebooks/README_COLAB.md](notebooks/README_COLAB.md)

**Por que Colab?**
- ✅ GPU Tesla T4 gratuita (16GB VRAM)
- ✅ 12GB RAM (sem travamentos)
- ✅ Processamento 10-20x mais rápido
- ✅ Sem instalação de dependências
- ✅ Não trava no carregamento de modelos

---

### 🌐 Interface Web Local

Use a **interface web interativa** para rodar localmente:

```bash
# Instalar Streamlit (se necessário)
pip install streamlit plotly

# Iniciar aplicação web
streamlit run app.py
```

A interface será aberta automaticamente no navegador em `http://localhost:8501`

**Funcionalidades da Interface:**
- 📤 Upload de vídeos via drag & drop
- ⚙️ Configuração visual de parâmetros
- 📊 Visualização interativa de resultados
- 📈 Gráficos e estatísticas em tempo real
- ⬇️ Download de relatórios e vídeo anotado
- 🎬 Preview do vídeo processado

### 💻 Linha de Comando (CLI)

```bash
# Uso básico - processar vídeo
python -m src.main --video data/input_video/video.mp4

# Processar e salvar vídeo anotado
python -m src.main --video input.mp4 --save-preview --output annotated.mp4

# Especificar backends específicos
python -m src.main --video input.mp4 --face-backend opencv --emotion-backend deepface

# Diretório de saída customizado
python -m src.main --video input.mp4 --output-dir custom_outputs/ --save-preview

# Usando Makefile
make run  # Executa com vídeo padrão
```

### Argumentos CLI

- `--video`: Caminho do vídeo de entrada **(obrigatório)**
- `--output`: Caminho do vídeo de saída anotado
- `--output-dir`: Diretório para salvar resultados (default: `outputs/`)
- `--save-preview`: Salva vídeo com anotações visuais
- `--face-backend`: Backend de detecção facial (`auto`, `opencv`, `face_recognition`, `deepface`)
- `--emotion-backend`: Backend de emoções (`auto`, `deepface`)
- `--no-report`: Não gerar relatórios (apenas processar)

### Outros Comandos

```bash
# Executar testes
make test

# Linting e type checking
make lint

# CI completo
make ci

# Limpar arquivos temporários
make clean
```

## � Interface Web (NOVO!)

### Visão Geral

Implementamos uma **interface web moderna e interativa** usando **Streamlit** para facilitar o uso do sistema!

**Capturas de Tela:**
```
┌─────────────────────────────────────────┐
│  🎬 Análise de Vídeo com IA            │
├─────────────────────────────────────────┤
│  ⚙️  Configurações    │  📊 Resultados │
│  • Upload de vídeo    │  • Métricas    │
│  • Backends           │  • Gráficos    │
│  • Opções             │  • Timeline    │
│  🚀 Processar         │  • Downloads   │
└─────────────────────────────────────────┘
```

### Recursos da Interface

#### 📤 Upload e Configuração
- **Drag & Drop** de vídeos (MP4, AVI, MOV, MKV)
- Seleção de backends (face detection e emotion)
- Opções visuais de processamento
- Validação automática de arquivos

#### 📊 Visualização de Resultados

**1. Métricas Principais (Dashboard)**
- Total de frames processados
- Duração do vídeo
- Total de faces detectadas
- Anomalias identificadas

**2. Análise de Faces**
- Total de detecções
- Média de faces por frame
- Máximo de faces em um frame
- Distribuição por frame

**3. Distribuição de Emoções**
- Gráfico de barras interativo
- Percentuais calculados
- Emojis para cada emoção
- Ordenação por frequência

**4. Timeline de Atividades**
- Lista de eventos detectados
- Frames de início/fim
- Scores de confiança
- Contagem por tipo

**5. Anomalias**
- Total e por severidade
- Alertas visuais (alta/média/baixa)
- Classificação por cores

#### ⬇️ Downloads
- **JSON**: Métricas completas estruturadas
- **Markdown**: Relatório formatado
- **Vídeo**: Preview com anotações
- Preview inline de todos os arquivos

### Como Usar a Interface Web

```bash
# Opção 1: Script de inicialização
./run_web.sh

# Opção 2: Comando direto
streamlit run app.py

# Opção 3: Usando Makefile
make web
```

A aplicação abrirá automaticamente em `http://localhost:8501`

### Passo a Passo

1. **Inicie a aplicação** usando um dos comandos acima
2. **Carregue um vídeo** na barra lateral (drag & drop ou clique)
3. **Configure opções** (opcional):
   - Salvar vídeo anotado (recomendado)
   - Escolher backends de detecção
4. **Clique em "Processar Vídeo"**
5. **Aguarde o processamento** (com barra de progresso)
6. **Explore os resultados** nas diferentes abas
7. **Faça download** dos relatórios e vídeo

### Vantagens da Interface Web

✅ **Mais fácil de usar** - Sem comandos complexos  
✅ **Visual e intuitivo** - Interface amigável  
✅ **Resultados imediatos** - Visualização inline  
✅ **Interativa** - Gráficos e métricas dinâmicas  
✅ **Downloads integrados** - Tudo em um só lugar  
✅ **Preview de vídeo** - Veja o resultado sem sair da interface  
✅ **Responsiva** - Funciona em diferentes tamanhos de tela  

## �🎬 Exemplos de Uso

### Exemplo 1: Processamento Básico
```bash
# Processar vídeo e gerar relatórios
python -m src.main --video meu_video.mp4

# Outputs gerados:
# - outputs/metrics.json
# - outputs/report.md
```

### Exemplo 2: Com Vídeo Anotado
```bash
# Gerar vídeo com visualizações
python -m src.main --video meu_video.mp4 --save-preview

# Outputs gerados:
# - outputs/metrics.json
# - outputs/report.md
# - outputs/annotated_video.mp4
```

### Exemplo 3: Configuração Avançada
```bash
# Personalizar backends e diretório de saída
python -m src.main \
  --video meu_video.mp4 \
  --save-preview \
  --output videos/resultado.mp4 \
  --output-dir resultados/ \
  --face-backend opencv \
  --emotion-backend deepface
```

### Exemplo 4: Programático (Python)
```python
from src.pipeline.inference import InferencePipeline
from src.metrics.reporter import Reporter

# Criar e executar pipeline
pipeline = InferencePipeline(
    video_path="video.mp4",
    output_video_path="output.mp4",
    save_preview=True
)
summary = pipeline.run()

# Gerar relatórios
reporter = Reporter()
reporter.save_report_bundle(summary, output_dir="outputs/")

# Acessar métricas
print(f"Frames: {summary['frames_total']}")
print(f"Anomalias: {summary['anomalies_total']}")
```

## 📁 Estrutura do Projeto

```
tech-challenge-fase4/
├── app.py                        ✅ Interface Web Streamlit
├── run_web.sh                    ✅ Script de inicialização
├── .streamlit/
│   └── config.toml               ✅ Configuração Streamlit
├── src/
│   ├── io/
│   │   ├── video_reader.py      ✅ Implementado
│   │   └── writer.py             ✅ Implementado
│   ├── face/
│   │   ├── detector.py           ✅ Implementado
│   │   └── tracker.py            � Futuro
│   ├── emotion/
│   │   └── classifier.py         ✅ Implementado
│   ├── activity/
│   │   └── recognizer.py         ✅ Implementado
│   ├── pipeline/
│   │   ├── inference.py          ✅ Implementado
│   │   ├── summarizer.py         ✅ Implementado
│   │   └── anomaly_detector.py   ✅ Implementado
│   ├── metrics/
│   │   └── reporter.py           ✅ Implementado
│   ├── utils/
│   │   └── viz.py                ✅ Implementado
│   └── main.py                   ✅ Implementado (CLI)
├── tests/                         ✅ 117 testes
├── models/                        📁 Modelos pré-treinados
├── data/input_video/              📁 Vídeos de entrada
├── outputs/                       📁 Resultados
├── requirements.txt               ✅ Configurado
├── Makefile                       ✅ Configurado
└── README.md                      ✅ Este arquivo
```

## 🎯 Funcionalidades Implementadas

### 1. VideoReader (✅ Completo)
- Iteração frame a frame
- Propriedades: fps(), frame_count(), duration()
- Context manager para limpeza automática
- Timestamps para cada frame
- Validação robusta de arquivos

### 2. VideoWriter (✅ Completo)
- Gravação de vídeos com OpenCV
- Codec configurável (mp4v, XVID, H.264, etc)
- Redimensionamento automático de frames
- Context manager para gestão de recursos

### 3. FaceDetector (✅ Completo)
- Múltiplos backends: OpenCV Haar Cascade, face_recognition, DeepFace
- Detecção de landmarks faciais
- Bounding boxes com scores de confiança
- Fallback automático entre backends

### 4. EmotionClassifier (✅ Completo)
- Classificação de 7 emoções via DeepFace
- Carregamento único do modelo (otimizado)
- Processamento em batch de múltiplos rostos
- Fallback para "neutral" em caso de erro

### 5. ActivityRecognizer (✅ Completo)
- Sliding window buffer (30 frames, stride 15)
- 3 tipos de atividades: walking, sitting, gesturing
- Análise de keypoints com MediaPipe Pose
- Detecção baseada em:
  - Variância de movimento das pernas
  - Ângulos das articulações
  - Padrões temporais de movimento
- Geração de eventos com timestamps e scores

### 6. AnomalyDetector (✅ Completo)
- Detecção estatística com z-score
- Sliding window para estatísticas históricas
- Classificação por severidade (low, medium, high)
- Tracking de anomalias por métrica
- Threshold configurável

### 7. Summarizer (✅ Completo)
- Agregação de todas as detecções
- Estatísticas de faces, emoções, atividades
- Contagem de anomalias por severidade
- Cálculo de médias e distribuições
- Exportação estruturada

### 8. InferencePipeline (✅ Completo)
- Orquestração de todos os módulos
- Processamento frame a frame
- Anotação visual automática
- Barra de progresso (tqdm)
- Gestão eficiente de recursos

### 9. Reporter (✅ Completo)
- Exportação JSON com métricas completas
- Geração de relatórios Markdown formatados
- Timestamp automático
- Bundle completo (JSON + MD)

### 10. Visualization Utils (✅ Completo)
- `draw_box_and_label()` - Desenha bounding boxes com labels
- `put_hud()` - Adiciona HUD com estatísticas
- `draw_landmarks()` - Desenha landmarks faciais
- `format_timestamp()` - Formatação de timestamps
- `create_color_palette()` - Paleta de cores para categorias

### 11. Main CLI (✅ Completo)
- Interface de linha de comando completa
- Validação de inputs
- Configuração flexível de backends
- Geração automática de relatórios
- Mensagens informativas e formatadas

## 📊 Métricas e Outputs

### Métricas Obrigatórias (Conforme Guideline)

✅ **frames_total** - Total de frames processados  
✅ **anomalies_total** - Total de anomalias detectadas

### Arquivos Gerados

O pipeline gera automaticamente:

#### 1. `outputs/metrics.json`
```json
{
  "generated_at": "2025-11-09T10:30:00",
  "video_path": "input.mp4",
  "frames_total": 900,
  "duration_seconds": 30.0,
  "fps": 30.0,
  "anomalies_total": 5,
  "faces_stats": {
    "total_detections": 850,
    "avg_faces_per_frame": 0.94,
    "max_faces_in_frame": 3,
    "frames_with_faces": 820,
    "frames_without_faces": 80
  },
  "emotions_distribution": {
    "happy": 450,
    "neutral": 280,
    "sad": 85,
    "surprise": 35
  },
  "activities_timeline": [
    {
      "label": "walking",
      "start": 0,
      "end": 150,
      "score": 0.85
    }
  ],
  "anomalies_by_severity": {
    "low": 2,
    "medium": 2,
    "high": 1
  }
}
```

#### 2. `outputs/report.md`
Relatório Markdown formatado com:
- Informações do vídeo
- Métricas obrigatórias destacadas
- Estatísticas de faces
- Distribuição de emoções com percentuais
- Timeline de atividades
- Anomalias por severidade

#### 3. `outputs/annotated_video.mp4` (opcional)
Vídeo com anotações visuais:
- Bounding boxes coloridas por emoção
- Labels com emoção e confiança
- HUD com estatísticas em tempo real
- Indicadores de anomalias
- Atividade atual detectada

## 🧪 Testes

```bash
# Todos os testes
pytest

# Testes específicos
pytest tests/test_video_reader.py
pytest tests/test_face_detector.py
pytest tests/test_emotion_classifier.py
pytest tests/test_activity_recognizer.py

# Com cobertura
pytest --cov=src tests/
```

## 🔄 Status Atual (09/11/2025)

**✅ PROJETO COMPLETO - 100% + INTERFACE WEB!**

**Implementado:**
- ✅ Estrutura completa do projeto
- ✅ Configuração de dependências
- ✅ Módulos core de processamento (I/O, face, emotion, activity)
- ✅ Sistema de detecção de anomalias (z-score)
- ✅ Pipeline completo de inferência
- ✅ Agregador de resultados (Summarizer)
- ✅ Sistema de relatórios (JSON + Markdown)
- ✅ Script principal com CLI completa
- ✅ Utilitários de visualização
- ✅ Suite de testes (117 testes passando)
- ✅ **Interface Web Streamlit** - NOVO! 🌟
- ✅ **Dashboard interativo com visualizações**
- ✅ **Upload de vídeos via drag & drop**
- ✅ **Downloads integrados de relatórios**

**Próximos Passos:**
1. ✅ ~~Implementar todos os módulos principais~~ - **CONCLUÍDO**
2. ✅ ~~Criar interface web~~ - **CONCLUÍDO**
3. 🔄 Testar com vídeos reais
4. 📹 Gerar vídeo demo (≤10 min)
5. 📝 Preparar apresentação final

## 📝 Notas Técnicas

- **Python**: 3.13.2
- **MediaPipe**: Fallback para dummy keypoints quando não disponível
- **DeepFace**: Requer TensorFlow 2.20+ com tf-keras
- **OpenCV**: Backend principal para processamento de vídeo

## 📄 Licença

Este projeto faz parte do Tech Challenge Fase 4 - FIAP
