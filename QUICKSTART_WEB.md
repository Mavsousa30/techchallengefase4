# 🚀 Guia Rápido - Interface Web

## Instalação e Inicialização

### 1. Instalar Dependências
```bash
pip install streamlit plotly
# ou
pip install -r requirements.txt
```

### 2. Iniciar Aplicação
```bash
# Opção mais fácil
./run_web.sh

# Ou diretamente
streamlit run app.py

# Ou via Make
make web
```

### 3. Acessar no Navegador
Abra automaticamente ou acesse: `http://localhost:8501`

## Como Usar

### Passo 1: Upload do Vídeo
1. Clique em "Browse files" ou arraste o vídeo
2. Formatos aceitos: MP4, AVI, MOV, MKV
3. Tamanho máximo: 500MB

### Passo 2: Configurar (Opcional)
**Sidebar - Opções:**
- ☑️ Salvar vídeo anotado (recomendado)
- Backend de faces (auto/opencv/face_recognition/deepface)
- Backend de emoções (auto/deepface)

### Passo 3: Processar
1. Clique no botão **"🚀 Processar Vídeo"**
2. Acompanhe a barra de progresso
3. Aguarde a conclusão (pode levar alguns minutos)

### Passo 4: Visualizar Resultados

**Dashboard Principal:**
- 📹 Total de frames
- ⏱️ Duração
- 👤 Faces detectadas
- ⚠️ Anomalias

**5 Abas de Análise:**

1. **👤 Faces**
   - Estatísticas de detecção
   - Média e máximo por frame
   - Distribuição

2. **😊 Emoções**
   - Gráfico de barras interativo
   - Percentuais por emoção
   - Top emoções com emojis

3. **🏃 Atividades**
   - Timeline completa
   - Eventos detectados
   - Scores de confiança

4. **⚠️ Anomalias**
   - Total e por severidade
   - Alertas coloridos
   - Classificação (alta/média/baixa)

5. **📄 Relatórios**
   - Download JSON
   - Download Markdown
   - Download vídeo anotado
   - Preview inline

### Passo 5: Download dos Resultados
1. Vá para a aba "📄 Relatórios"
2. Clique nos botões de download:
   - ⬇️ Download JSON (métricas)
   - ⬇️ Download Markdown (relatório)
   - ⬇️ Download Vídeo (anotado)
3. Use "👁️ Preview" para ver antes de baixar

## Dicas Úteis

### ⚡ Performance
- Vídeos menores processam mais rápido
- Use backend "opencv" para velocidade
- Feche outras abas do browser

### 🎯 Qualidade
- Vídeos com boa iluminação
- Pessoas visíveis e próximas
- Formato MP4 H.264 recomendado

### 💾 Armazenamento
- Arquivos temporários são limpos automaticamente
- Resultados salvos em `/outputs`
- Vídeo anotado pode ser grande

### 🐛 Problemas Comuns

**Upload falha:**
- Verifique o tamanho (< 500MB)
- Confirme o formato
- Tente outro vídeo

**Processamento lento:**
- Normal para vídeos longos
- Aguarde pacientemente
- Veja log de processamento

**Erro de módulos:**
```bash
pip install -r requirements.txt
```

## Exemplo de Uso Completo

```bash
# 1. Instalar
pip install streamlit plotly

# 2. Iniciar
streamlit run app.py

# 3. No browser:
#    - Upload: video.mp4
#    - Config: ☑️ Salvar vídeo anotado
#    - Backend: auto
#    - Clicar: 🚀 Processar Vídeo

# 4. Aguardar processamento...

# 5. Explorar resultados nas 5 abas

# 6. Download dos relatórios
```

## Atalhos de Teclado

- `Ctrl + R` - Recarregar aplicação
- `Ctrl + C` (terminal) - Encerrar servidor
- `Ctrl + Shift + R` - Hard refresh

## Próximos Passos

Após processar seu primeiro vídeo:
1. Explore todas as abas de análise
2. Faça download dos relatórios
3. Assista o vídeo anotado
4. Experimente diferentes configurações
5. Compare resultados de diferentes vídeos

## Suporte

Para mais informações, veja:
- `README.md` - Documentação completa
- `WEB_INTERFACE.md` - Guia detalhado da interface
- Issues no GitHub - Reporte problemas

---

**Desenvolvido com ❤️ para o Tech Challenge Fase 4**
