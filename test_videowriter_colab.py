#!/usr/bin/env python3
"""
Script de teste para diagnosticar problemas com VideoWriter no Colab
"""

import os
import sys
import cv2
import numpy as np

print("🔍 DIAGNÓSTICO DE VIDEOWRITER - GOOGLE COLAB")
print("=" * 70)

# 1. Verificar diretório atual
print("\n1️⃣ DIRETÓRIO ATUAL")
print(f"   Diretório: {os.getcwd()}")
print(f"   Conteúdo: {os.listdir('.')[:10]}")  # Primeiros 10 arquivos

# 2. Verificar se outputs existe
print("\n2️⃣ DIRETÓRIO OUTPUTS")
if os.path.exists('outputs'):
    print(f"   ✅ outputs/ existe")
    print(f"   Permissões: {oct(os.stat('outputs').st_mode)[-3:]}")
    print(f"   Conteúdo: {os.listdir('outputs')}")
else:
    print(f"   ❌ outputs/ NÃO existe")
    print(f"   Criando...")
    os.makedirs('outputs', exist_ok=True)
    if os.path.exists('outputs'):
        print(f"   ✅ Criado com sucesso!")
    else:
        print(f"   ❌ Falha ao criar!")
        sys.exit(1)

# 3. Testar criação de arquivo simples
print("\n3️⃣ TESTE DE ESCRITA SIMPLES")
test_file = 'outputs/test.txt'
try:
    with open(test_file, 'w') as f:
        f.write('test')
    print(f"   ✅ Arquivo texto criado: {test_file}")
    print(f"   Tamanho: {os.path.getsize(test_file)} bytes")
    os.remove(test_file)
except Exception as e:
    print(f"   ❌ Erro ao criar arquivo: {e}")
    sys.exit(1)

# 4. Verificar OpenCV
print("\n4️⃣ OPENCV")
print(f"   Versão: {cv2.__version__}")

# 5. Testar codecs disponíveis
print("\n5️⃣ CODECS DISPONÍVEIS")
codecs_to_test = ['mp4v', 'XVID', 'MJPG', 'X264', 'avc1']
available_codecs = []

for codec in codecs_to_test:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Tentar criar writer
        writer = cv2.VideoWriter(
            'outputs/test_codec.mp4',
            fourcc,
            30.0,
            (640, 480)
        )
        
        if writer.isOpened():
            available_codecs.append(codec)
            print(f"   ✅ {codec} - Disponível")
            writer.release()
            
            # Tentar escrever um frame
            writer = cv2.VideoWriter(
                'outputs/test_codec.mp4',
                fourcc,
                30.0,
                (640, 480)
            )
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)
            writer.release()
            
            if os.path.exists('outputs/test_codec.mp4'):
                size = os.path.getsize('outputs/test_codec.mp4')
                print(f"      Arquivo criado: {size} bytes")
                os.remove('outputs/test_codec.mp4')
            else:
                print(f"      ⚠️  Arquivo não foi criado")
        else:
            print(f"   ❌ {codec} - Não disponível (isOpened=False)")
    except Exception as e:
        print(f"   ❌ {codec} - Erro: {e}")

# 6. Teste completo com mp4v
print("\n6️⃣ TESTE COMPLETO COM MP4V")
video_path = 'outputs/test_complete.mp4'

try:
    print(f"   Criando vídeo: {video_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    if not writer.isOpened():
        print(f"   ❌ VideoWriter não abriu!")
    else:
        print(f"   ✅ VideoWriter aberto")
        
        # Escrever 10 frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
        print(f"   ✅ 10 frames escritos")
        
        if os.path.exists(video_path):
            size = os.path.getsize(video_path)
            print(f"   ✅ Arquivo criado: {size:,} bytes")
            
            if size > 0:
                print(f"   ✅ SUCESSO TOTAL!")
            else:
                print(f"   ⚠️  Arquivo vazio")
            
            os.remove(video_path)
        else:
            print(f"   ❌ Arquivo não foi criado")
            
except Exception as e:
    print(f"   ❌ Erro: {e}")
    import traceback
    traceback.print_exc()

# 7. Resumo
print("\n" + "=" * 70)
print("📊 RESUMO")
print(f"   Diretório OK: {'✅' if os.path.exists('outputs') else '❌'}")
print(f"   Escrita OK: ✅")
print(f"   OpenCV: {cv2.__version__}")
print(f"   Codecs disponíveis: {', '.join(available_codecs) if available_codecs else 'NENHUM'}")

if available_codecs:
    print(f"\n✅ VideoWriter deve funcionar com: {available_codecs[0]}")
    print(f"   Use: InferencePipeline(..., save_preview=True)")
else:
    print(f"\n⚠️  Nenhum codec funcional detectado!")
    print(f"   Recomendação: Use save_preview=False")
    print(f"   Ou instale: !apt-get install -y ffmpeg")

print("=" * 70)
