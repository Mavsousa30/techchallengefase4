# 🚀 Guia Rápido: Google Colab

## Por que usar o Google Colab?

### ✅ **MUITO mais rápido que rodar localmente**

| Recurso | Local (Seu Mac) | Google Colab (Gratuito) |
|---------|----------------|-------------------------|
| **GPU** | ❌ Sem GPU | ✅ Tesla T4 (16GB VRAM) |
| **RAM** | ~8-16GB | ✅ 12-13GB garantidos |
| **Velocidade** | 1x (CPU apenas) | ⚡ **10-20x mais rápido** |
| **Travamentos** | ❌ Comum (mutex lock) | ✅ Raro |
| **Setup** | ❌ Complicado | ✅ Plug & Play |

### 🎯 **Vantagens Específicas:**

1. **GPU Grátis**: DeepFace e modelos de emoção rodam em GPU
2. **Não trava**: 12GB RAM suficiente para carregar todos os modelos
3. **Pré-instalado**: OpenCV, PyTorch já configurados
4. **Sem instalação**: Tudo roda no navegador
5. **Compartilhável**: Envie o link para outros testarem

---

## 📖 Como Usar

### **Opção 1: Upload direto do notebook** (Mais fácil)

1. Acesse: https://colab.research.google.com/
2. Clique em **"Upload"**
3. Selecione o arquivo: `notebooks/Colab_Video_Analysis.ipynb`
4. **Ative a GPU**: Runtime → Change runtime type → GPU (T4)
5. Execute célula por célula (`Shift + Enter`)

### **Opção 2: Via GitHub** (Mais rápido)

1. Faça commit deste notebook no seu repositório
2. Acesse: `https://colab.research.google.com/github/SEU_USUARIO/TechChallengeFase4/blob/main/notebooks/Colab_Video_Analysis.ipynb`
3. **Ative a GPU**: Runtime → Change runtime type → GPU (T4)
4. Execute célula por célula

---

## 🎬 Passo a Passo Completo

### 1️⃣ **Configuração Inicial** (2-3 minutos)
```python
# Célula 1: Verificar GPU
!nvidia-smi -L

# Célula 2: Clone o repositório
!git clone https://github.com/SEU_USUARIO/TechChallengeFase4.git
%cd TechChallengeFase4

# Célula 3: Instalar dependências
!pip install -q opencv-python numpy pandas deepface torch torchvision
```

### 2️⃣ **Upload do Vídeo** (< 1 minuto)
```python
from google.colab import files
uploaded = files.upload()  # Escolha seu vídeo
```

### 3️⃣ **Processar** (depende do tamanho do vídeo)
```python
from src.pipeline.inference import InferencePipeline

pipeline = InferencePipeline(
    video_path="video.mp4",
    save_preview=True
)
summary = pipeline.run()
```

### 4️⃣ **Baixar Resultados**
```python
from google.colab import files
files.download("outputs/annotated_video.mp4")
files.download("outputs/metrics.json")
files.download("outputs/report.md")
```

---

## ⚡ Performance Esperada

| Vídeo | Local (CPU) | Colab (GPU) | Speedup |
|-------|-------------|-------------|---------|
| 30 seg | ~15-20 min | ⚡ **2-3 min** | 7x |
| 1 min | ~30-40 min | ⚡ **5-7 min** | 6x |
| 2 min | ~1h+ | ⚡ **10-15 min** | 5-6x |

*Tempos aproximados. Varia com resolução, número de faces, etc.*

---

## 🐛 Resolução de Problemas

### **"No module named 'src'" ou erro de importação**
```python
# Execute esta célula para corrigir
import sys
import os

# Configurar path
if '/content/TechChallengeFase4' not in sys.path:
    sys.path.insert(0, '/content/TechChallengeFase4')
os.chdir('/content/TechChallengeFase4')

# Verificar
print("✅ Diretório:", os.getcwd())
!ls -la src/
```

**Ou:**
- Certifique-se que executou a célula de clone do Git
- Use o notebook alternativo: `Colab_Video_Analysis_Upload.ipynb`

### **"Sem GPU disponível"**
- Vá em: Runtime → Change runtime type → Hardware accelerator → GPU
- Se não aparecer GPU: aguarde ou tente outro horário (limite diário)

### **"Out of memory"**
- Use vídeos menores (< 2 minutos)
- Reduza resolução do vídeo antes
- Reinicie o runtime: Runtime → Restart runtime

### **"Model download failing"**
- DeepFace baixa modelos na primeira vez (~200MB)
- Aguarde 2-3 minutos
- Se falhar, execute novamente a célula

### **"Can't find video file"**
- Certifique-se que fez upload do vídeo
- Use o nome correto do arquivo
- Verifique com: `!ls -lh *.mp4`

---

## 💡 Dicas de Otimização

### **Para Vídeos Grandes:**
```python
# Processar apenas parte do vídeo (para testes)
# Modifique o VideoReader para limitar frames
```

### **Melhor Backend:**
```python
# OpenCV é mais rápido, DeepFace mais preciso
pipeline = InferencePipeline(
    video_path="video.mp4",
    face_backend="opencv",      # Mais rápido
    emotion_backend="deepface"   # Mais preciso
)
```

### **Salvar na Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Salvar resultados na Drive
!cp -r outputs/* /content/drive/MyDrive/video_analysis/
```

---

## 📊 Comparação: Local vs Colab

### **Seu Problema Local:**
```
[mutex.cc : 452] RAW: Lock blocking...
```
**Causa**: DeepFace travando ao carregar modelos (CPU limitado + baixa RAM)

### **No Colab:**
- ✅ GPU acelera carregamento de modelos
- ✅ 12GB RAM suficiente
- ✅ Sem travamentos
- ✅ Progresso visível
- ✅ 10x mais rápido

---

## 🎓 Recursos Adicionais

- **Documentação**: [../README.md](../README.md)
- **Interface Web Local**: [../WEB_INTERFACE.md](../WEB_INTERFACE.md)
- **Quickstart**: [../QUICKSTART_WEB.md](../QUICKSTART_WEB.md)
- **Colab Oficial**: https://colab.research.google.com/

---

## 🚀 Começar Agora!

1. **Abra o notebook**: `Colab_Video_Analysis.ipynb`
2. **Ative GPU**: Runtime → GPU (T4)
3. **Execute tudo**: Runtime → Run all
4. **Aguarde**: ~5-10 minutos
5. **Baixe resultados**: Última célula

**Muito mais rápido que rodar localmente!** ⚡

---

## ❓ Perguntas Frequentes

**Q: É grátis?**  
A: Sim! Google Colab oferece GPU gratuita com limites diários.

**Q: Quanto tempo tenho?**  
A: ~12 horas de sessão contínua. Suficiente para vários vídeos.

**Q: Meus arquivos ficam salvos?**  
A: Não. Faça download ou salve na Google Drive ao final.

**Q: Posso processar vários vídeos?**  
A: Sim! Execute o pipeline múltiplas vezes na mesma sessão.

**Q: E a privacidade?**  
A: Seus vídeos ficam temporários no Colab. Delete após uso.

---

**🎬 Divirta-se analisando vídeos com IA! Muito mais rápido no Colab! ⚡**
