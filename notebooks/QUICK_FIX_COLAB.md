# 🚀 QUICK FIX - Erro de Importação no Colab

## ❌ Problema: "No module named 'src.pipeline.inference'"

### ✅ SOLUÇÃO RÁPIDA (copie e cole):

```python
# Cole esta célula ANTES de importar os módulos
import sys
import os

# Encontrar e configurar o projeto
project_paths = [
    '/content/TechChallengeFase4',
    '/content'
]

for path in project_paths:
    if os.path.exists(os.path.join(path, 'src')):
        if path not in sys.path:
            sys.path.insert(0, path)
        os.chdir(path)
        print(f"✅ Projeto encontrado: {path}")
        break
else:
    print("❌ ERRO: Projeto não encontrado!")
    print("💡 Solução:")
    print("   1. Execute a célula de git clone")
    print("   2. Ou faça upload manual do projeto")

# Verificar
print(f"\n📂 Diretório atual: {os.getcwd()}")
print(f"\n📦 Módulos disponíveis:")
!ls -la src/ 2>/dev/null || echo "❌ Pasta src/ não existe"
```

---

## 📋 Checklist Completo:

### ✅ Antes de processar vídeo:

```python
# CÉLULA 1: Ativar GPU
# Runtime → Change runtime type → GPU (T4)

# CÉLULA 2: Instalar dependências
!pip install -q opencv-python numpy pandas deepface torch torchvision tqdm pydantic rich moviepy

# CÉLULA 3: Clone OU Upload
# Opção A: Git clone
!git clone https://github.com/SEU_USUARIO/TechChallengeFase4.git
%cd TechChallengeFase4

# Opção B: Upload manual (se git falhar)
from google.colab import files
import zipfile
uploaded = files.upload()
for f in uploaded:
    with zipfile.ZipFile(f, 'r') as z:
        z.extractall('.')

# CÉLULA 4: Configurar path (USE O CÓDIGO ACIMA)

# CÉLULA 5: Testar importações
from src.pipeline.inference import InferencePipeline
from src.metrics.reporter import Reporter
print("✅ Tudo OK!")

# CÉLULA 6: Upload do vídeo
from google.colab import files
uploaded = files.upload()
video_filename = list(uploaded.keys())[0]

# CÉLULA 7: Processar
pipeline = InferencePipeline(
    video_path=video_filename,
    output_video_path="outputs/annotated.mp4",
    save_preview=True,
    face_backend="opencv",
    emotion_backend="deepface"
)
summary = pipeline.run()

# CÉLULA 8: Download
from google.colab import files
files.download("outputs/annotated.mp4")
```

---

## 🎯 3 Notebooks Disponíveis:

### 1. **Colab_Video_Analysis.ipynb** ⭐ Principal
- Usa Git clone (mais fácil)
- Requer repositório público no GitHub
- **Use este se tiver o projeto no GitHub**

### 2. **Colab_Video_Analysis_Upload.ipynb** 🔄 Alternativo
- Upload manual do projeto (ZIP)
- Não precisa de GitHub
- **Use este se tiver erro de git clone**

### 3. **Mini versão inline** (células acima)
- Copie e cole diretamente
- Para testes rápidos
- **Use para debug**

---

## 🔍 Debug Rápido:

```python
# Cole e execute para diagnóstico completo
import sys, os
print("📂 Dir:", os.getcwd())
print("🐍 Path:", sys.path[0])
print("📦 src/:", "✅" if os.path.exists('src') else "❌")
print("🎮 GPU:", "✅" if __import__('torch').cuda.is_available() else "❌")

try:
    from src.pipeline.inference import InferencePipeline
    print("✅ Importação OK!")
except ImportError as e:
    print(f"❌ Erro: {e}")
    print("\n💡 Execute o código de configuração de path acima")
```

---

## 📚 Documentação Completa:

- **Guia Colab**: [README_COLAB.md](README_COLAB.md)
- **Troubleshooting**: [TROUBLESHOOTING_COLAB.md](TROUBLESHOOTING_COLAB.md)
- **README Principal**: [../README.md](../README.md)

---

## 💬 Ainda com erro?

1. **Restart Runtime**: Runtime → Restart runtime
2. **Execute TODAS as células em ordem**
3. **Verifique se a pasta `src/` existe**: `!ls -la`
4. **Use o notebook alternativo de upload manual**
5. **Veja**: [TROUBLESHOOTING_COLAB.md](TROUBLESHOOTING_COLAB.md)
