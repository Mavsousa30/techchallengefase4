# 🔧 Solução: VideoWriterError no Google Colab

## ❌ Erro Completo
```python
VideoWriterError: Failed to write frame to outputs/annotated.mp4
```

## 🎯 Causa do Problema

O erro ocorre quando o OpenCV não consegue escrever frames no arquivo de vídeo. Isso pode acontecer por várias razões no Google Colab:

### Possíveis Causas:
1. ✅ **Codec incompatível** - O codec `mp4v` pode não funcionar em alguns ambientes
2. ✅ **Permissões de escrita** - Problema ao criar/escrever arquivo
3. ✅ **Espaço em disco** - Colab sem espaço suficiente (raro)
4. ✅ **OpenCV build** - Versão do OpenCV sem suporte a codec

## ✅ Solução 1: Processar SEM Vídeo Anotado (Mais Rápido)

**Recomendado se você só precisa dos relatórios e métricas!**

### Passo 1: Modificar a Célula de Processamento

**Troque isto:**
```python
pipeline = InferencePipeline(
    video_path=video_file,
    output_video_path='outputs/annotated.mp4',
    save_preview=True,  # ← Causa o erro
    face_backend='opencv',
    emotion_backend='deepface'
)
```

**Por isto:**
```python
pipeline = InferencePipeline(
    video_path=video_file,
    output_video_path=None,      # ← Não salvar vídeo
    save_preview=False,           # ← Desabilitar preview
    face_backend='opencv',
    emotion_backend='deepface'
)
```

### Passo 2: Executar

```python
summary = pipeline.run()
print('\n✅ Processamento concluído (sem vídeo anotado)!')
```

### ✅ Vantagens:
- ✅ **Muito mais rápido** (não precisa escrever frames)
- ✅ **Sem problemas de codec**
- ✅ **Gera todos os relatórios** (JSON e Markdown)
- ✅ **Todas as métricas** são calculadas normalmente

### ❌ Desvantagens:
- Você não terá o vídeo com anotações visuais
- Mas terá todas as detecções e métricas em JSON/MD!

---

## ✅ Solução 2: Tentar Codec Diferente

Se você **realmente precisa** do vídeo anotado, tente modificar o codec no `writer.py`:

### Opção A: Usar XVID (geralmente funciona)

Edite `/content/techchallengefase4/src/io/writer.py` linha ~50:

```python
def __init__(
    self,
    path: str,
    fps: float,
    frame_size: tuple[int, int],
    codec: str = "XVID"  # ← Mude de "mp4v" para "XVID"
) -> None:
```

### Opção B: Usar MJPG (sempre funciona, mas arquivo grande)

```python
codec: str = "MJPG"  # ← Motion JPEG
```

### Opção C: Adicionar Fallback Automático

Adicione esta função no início do notebook:

```python
def create_pipeline_with_fallback(video_path, output_path):
    """Tenta criar pipeline com diferentes codecs"""
    codecs = ['mp4v', 'XVID', 'MJPG', 'X264']
    
    for codec in codecs:
        try:
            # Modificar temporariamente o codec padrão
            from src.io import writer
            original_init = writer.VideoWriter.__init__
            
            def custom_init(self, path, fps, frame_size, codec_override=codec):
                return original_init(self, path, fps, frame_size, codec_override)
            
            writer.VideoWriter.__init__ = custom_init
            
            pipeline = InferencePipeline(
                video_path=video_path,
                output_video_path=output_path,
                save_preview=True,
                face_backend='opencv',
                emotion_backend='deepface'
            )
            
            print(f'✅ Usando codec: {codec}')
            return pipeline
            
        except Exception as e:
            print(f'❌ Codec {codec} falhou: {e}')
            continue
    
    print('⚠️  Todos os codecs falharam. Processando sem vídeo.')
    return InferencePipeline(
        video_path=video_path,
        output_video_path=None,
        save_preview=False,
        face_backend='opencv',
        emotion_backend='deepface'
    )

# Usar:
pipeline = create_pipeline_with_fallback(video_file, 'outputs/annotated.mp4')
summary = pipeline.run()
```

---

## ✅ Solução 3: Verificar Codecs Disponíveis

Execute esta célula para ver quais codecs estão disponíveis:

```python
import cv2
import numpy as np

codecs = ['mp4v', 'XVID', 'MJPG', 'X264', 'avc1', 'H264']
available = []

for codec in codecs:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            'test.mp4',
            fourcc,
            30.0,
            (640, 480)
        )
        
        if writer.isOpened():
            available.append(codec)
            writer.release()
        
    except Exception as e:
        pass

print('✅ Codecs disponíveis no seu Colab:')
for c in available:
    print(f'   • {c}')

if not available:
    print('⚠️  Nenhum codec disponível!')
    print('💡 Recomendação: Processe sem vídeo (save_preview=False)')

# Limpar arquivo de teste
import os
if os.path.exists('test.mp4'):
    os.remove('test.mp4')
```

---

## ✅ Solução 4: Instalar FFmpeg (Pode Resolver)

Às vezes o problema é falta de FFmpeg:

```python
# Instalar FFmpeg no Colab
!apt-get update -qq
!apt-get install -y ffmpeg

print('✅ FFmpeg instalado!')
print('⚠️  Reinicie o runtime: Runtime → Restart runtime')
```

Depois de reiniciar:
1. Execute todas as células de setup novamente
2. Tente processar o vídeo

---

## 🆘 Comparação das Soluções

| Solução | Velocidade | Funciona? | Tem Vídeo? | Tem Métricas? |
|---------|-----------|-----------|------------|---------------|
| **save_preview=False** | ⚡⚡⚡ Muito rápido | ✅ Sempre | ❌ Não | ✅ Sim |
| **XVID codec** | ⚡⚡ Rápido | ✅ Geralmente | ✅ Sim | ✅ Sim |
| **MJPG codec** | ⚡ Médio | ✅ Sempre | ✅ Sim (grande) | ✅ Sim |
| **FFmpeg install** | ⚡⚡ Rápido | ⚠️ Às vezes | ✅ Sim | ✅ Sim |

### 💡 Recomendação:

**Para análise rápida**: Use `save_preview=False` (Solução 1)
- Você tem todas as métricas em JSON/Markdown
- Pode visualizar os dados sem o vídeo
- Processa muito mais rápido

**Para apresentação**: Tente XVID codec (Solução 2A)
- Gera vídeo para mostrar
- Geralmente funciona no Colab

---

## 📊 O Que Você NÃO Perde Sem o Vídeo

Mesmo sem salvar o vídeo anotado (`save_preview=False`), você ainda tem:

✅ **Todas as detecções**:
- Faces detectadas (quantidade, localização)
- Emoções classificadas (distribuição completa)
- Atividades reconhecidas (timeline completa)
- Anomalias identificadas (por severidade)

✅ **Relatórios completos**:
- `metrics.json` - Todos os dados estruturados
- `report.md` - Relatório formatado em Markdown
- Estatísticas agregadas (média, máximo, totais)

✅ **Métricas obrigatórias**:
- `frames_total` - Total de frames processados
- `anomalies_total` - Total de anomalias
- Todas as outras métricas do projeto

❌ **O que você perde**:
- Apenas a visualização do vídeo com boxes e labels
- (Mas os dados estão todos nos relatórios!)

---

## 🔍 Debug Avançado

Se nenhuma solução funcionar, execute este diagnóstico:

```python
import cv2
import numpy as np
import os

print('🔍 DIAGNÓSTICO DE VIDEOWRITER')
print('=' * 60)

# 1. Verificar OpenCV
print(f'\n1. OpenCV version: {cv2.__version__}')

# 2. Verificar build info
build_info = cv2.getBuildInformation()
if 'FFMPEG' in build_info:
    print('2. FFMPEG: ✅ Disponível')
else:
    print('2. FFMPEG: ❌ Não disponível (pode causar problemas)')

# 3. Testar escrita
print('\n3. Testando escrita de vídeo...')
test_path = 'test_write.mp4'
frame = np.zeros((480, 640, 3), dtype=np.uint8)

try:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(test_path, fourcc, 30.0, (640, 480))
    
    if writer.isOpened():
        writer.write(frame)
        writer.release()
        
        if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
            print('   ✅ Escrita funciona!')
            os.remove(test_path)
        else:
            print('   ❌ Arquivo criado mas vazio')
    else:
        print('   ❌ VideoWriter não abriu')
        
except Exception as e:
    print(f'   ❌ Erro: {e}')

# 4. Verificar diretório
print(f'\n4. Diretório outputs:')
if os.path.exists('outputs'):
    print(f'   ✅ Existe')
    print(f'   Permissões: {oct(os.stat("outputs").st_mode)[-3:]}')
else:
    print(f'   ❌ Não existe')

print('=' * 60)
```

---

## ✅ Resumo Executivo

### Solução Mais Simples (Recomendada):
```python
# Processar SEM vídeo anotado
pipeline = InferencePipeline(
    video_path=video_file,
    save_preview=False,  # ← Solução!
    face_backend='opencv',
    emotion_backend='deepface'
)
```

### Por que funciona?
- ❌ Não tenta criar VideoWriter
- ✅ Processa tudo normalmente
- ✅ Gera todos os relatórios
- ⚡ Até 30% mais rápido!

### Você ainda tem:
- ✅ JSON completo com todas as detecções
- ✅ Markdown com estatísticas
- ✅ Todas as métricas obrigatórias
- ✅ Dados para gráficos e análises

---

**Última atualização**: 9 de Novembro de 2025
