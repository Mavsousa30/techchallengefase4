# 🔧 Troubleshooting - Colab

## 🚨 Erros Mais Comuns

### ⚠️ **NOVO**: VideoWriterError: Failed to write frame
**→ [Solução Completa aqui](FIX_VIDEOWRITER_ERROR.md)**

**Solução Rápida:**
```python
# Use save_preview=False (não salva vídeo, mas gera todos os relatórios)
pipeline = InferencePipeline(
    video_path=video_file,
    save_preview=False,  # ← Solução!
    face_backend='opencv',
    emotion_backend='deepface'
)
```

---

## ❌ Erro: "No module named 'src'"

### **Causa:**
O Python não está encontrando os módulos do projeto.

### **Solução 1: Verificar se clonou o repositório**
```python
# Execute esta célula
import os
print("Diretório atual:", os.getcwd())
print("\nArquivos disponíveis:")
!ls -la
```

**Se não aparecer a pasta `src/`**, você precisa:
1. Voltar à célula de clone do Git
2. Executar: `!git clone ...`
3. Executar: `%cd TechChallengeFase4`

### **Solução 2: Configurar Python path manualmente**
```python
import sys
import os

# Adicionar projeto ao path
if '/content/TechChallengeFase4' not in sys.path:
    sys.path.insert(0, '/content/TechChallengeFase4')

# Mudar para o diretório
os.chdir('/content/TechChallengeFase4')

print("✅ Path configurado!")
print("Diretório:", os.getcwd())
```

### **Solução 3: Usar upload manual**
Use o notebook alternativo: `Colab_Video_Analysis_Upload.ipynb`

---

## ❌ Erro: "Failed to clone repository"

### **Causa:**
- Repositório não existe ou está privado
- URL incorreta

### **Solução: Upload manual do projeto**

**Opção A: ZIP completo**
```python
from google.colab import files
import zipfile

# 1. Faça upload do ZIP
uploaded = files.upload()

# 2. Descompactar
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

# 3. Entrar na pasta
%cd TechChallengeFase4
```

**Opção B: Upload via Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copiar projeto da Drive
!cp -r /content/drive/MyDrive/TechChallengeFase4 /content/
%cd TechChallengeFase4
```

---

## ❌ Erro: "GPU not available"

### **Verificar GPU:**
```python
import torch
print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")
```

### **Solução: Ativar GPU**
1. Menu: **Runtime** → **Change runtime type**
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (padrão gratuito)
4. Clique **Save**
5. **Restart runtime**

### **Se não aparecer GPU:**
- Colab gratuito tem limite diário de GPU
- Tente em outro horário (menos concorrido)
- Considere Colab Pro ($9.99/mês)

**Alternativa sem GPU:**
```python
# Funciona, mas será mais lento (10-20x)
pipeline = InferencePipeline(
    video_path="video.mp4",
    face_backend="opencv",  # Mais rápido em CPU
    emotion_backend="fallback"  # Evita DeepFace pesado
)
```

---

## ❌ Erro: "Out of Memory" (OOM)

### **Causa:**
Vídeo muito grande ou muitos frames.

### **Solução 1: Usar vídeo menor**
```python
# Cortar vídeo nos primeiros 30 segundos
!ffmpeg -i video.mp4 -t 30 -c copy video_short.mp4

# Processar versão curta
pipeline = InferencePipeline(video_path="video_short.mp4", ...)
```

### **Solução 2: Reduzir resolução**
```python
# Redimensionar vídeo
!ffmpeg -i video.mp4 -vf scale=640:-1 -c:a copy video_low.mp4

pipeline = InferencePipeline(video_path="video_low.mp4", ...)
```

### **Solução 3: Limpar memória**
```python
# Limpar GPU
import torch
torch.cuda.empty_cache()

# Reiniciar runtime
# Runtime → Restart runtime
```

---

## ❌ Erro: "DeepFace model download failed"

### **Causa:**
DeepFace tenta baixar modelos (~200MB) na primeira execução.

### **Solução 1: Aguardar**
- Primeira execução demora 2-5 minutos
- Modelos são salvos e não precisam ser baixados novamente

### **Solução 2: Download manual**
```python
# Forçar download dos modelos
from deepface import DeepFace
import numpy as np

print("📥 Baixando modelos do DeepFace...")
dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

try:
    DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
    print("✅ Modelos carregados!")
except:
    pass
```

### **Solução 3: Usar backend leve**
```python
pipeline = InferencePipeline(
    video_path="video.mp4",
    face_backend="opencv",
    emotion_backend="fallback"  # Sem DeepFace
)
```

---

## ❌ Erro: "Video file not found"

### **Verificar arquivo:**
```python
import os
print("Arquivos disponíveis:")
!ls -lh *.mp4
```

### **Solução:**
```python
# Verificar nome exato do vídeo
video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
print("Vídeos encontrados:", video_files)

# Usar o nome correto
video_filename = video_files[0]  # Primeiro vídeo encontrado
```

---

## ❌ Erro: "Session disconnected"

### **Causa:**
- Sessão do Colab expira após ~12 horas
- Ou após 90 minutos de inatividade

### **Solução:**
```javascript
// Execute no console do navegador (F12)
// Mantém sessão ativa
function KeepAlive() { 
    console.log("Keeping alive..."); 
    document.querySelector("colab-connect-button").click(); 
}
setInterval(KeepAlive, 60000); // A cada 1 minuto
```

**Ou:**
- Salve resultados frequentemente na Drive
- Use Colab Pro para sessões mais longas

---

## 🐛 Debug Geral

### **Verificar tudo está OK:**
```python
import sys
import os

print("=" * 50)
print("🔍 DIAGNÓSTICO COMPLETO")
print("=" * 50)

# 1. Diretório
print(f"\n📂 Diretório: {os.getcwd()}")

# 2. Python path
print(f"\n🐍 Python path:")
for p in sys.path[:3]:
    print(f"   - {p}")

# 3. Estrutura do projeto
print(f"\n📦 Estrutura:")
!ls -la src/ 2>/dev/null || echo "❌ Pasta src/ não encontrada"

# 4. GPU
import torch
print(f"\n🎮 GPU:")
print(f"   CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# 5. Memória
import psutil
mem = psutil.virtual_memory()
print(f"\n💾 Memória:")
print(f"   Total: {mem.total / 1e9:.1f} GB")
print(f"   Disponível: {mem.available / 1e9:.1f} GB")

# 6. Importações
print(f"\n📚 Testando importações:")
try:
    from src.pipeline.inference import InferencePipeline
    print("   ✅ InferencePipeline")
except ImportError as e:
    print(f"   ❌ InferencePipeline: {e}")

try:
    from src.metrics.reporter import Reporter
    print("   ✅ Reporter")
except ImportError as e:
    print(f"   ❌ Reporter: {e}")

print("\n" + "=" * 50)
```

---

## 💡 Dicas de Otimização

### **1. Processar vídeos curtos primeiro**
```python
# Testar com 10 segundos
!ffmpeg -i video.mp4 -t 10 -c copy test.mp4
```

### **2. Usar backends mais rápidos**
```python
pipeline = InferencePipeline(
    video_path="video.mp4",
    face_backend="opencv",      # Mais rápido
    emotion_backend="deepface"  # Ou "fallback" para testar
)
```

### **3. Salvar progresso na Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Salvar resultados
!cp -r outputs /content/drive/MyDrive/video_results/
```

---

## 📞 Ainda com problemas?

### **Checklist:**
- [ ] Executou TODAS as células em ordem?
- [ ] Ativou a GPU no Runtime?
- [ ] Pasta `src/` existe no diretório?
- [ ] Python path está configurado?
- [ ] Vídeo foi carregado com sucesso?

### **Opções:**
1. **Reiniciar runtime**: Runtime → Restart runtime
2. **Executar tudo novamente**: Runtime → Run all
3. **Usar notebook alternativo**: `Colab_Video_Analysis_Upload.ipynb`
4. **Verificar logs**: Procure por mensagens de erro específicas

---

## 📚 Recursos Adicionais

- [README Colab](README_COLAB.md)
- [Documentação Principal](../README.md)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
