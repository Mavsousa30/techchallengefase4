# 🌐 Interface Web - Tech Challenge Fase 4

## Visão Geral

Interface web moderna e interativa desenvolvida com **Streamlit** para facilitar a análise de vídeos com IA.

## 🚀 Inicialização Rápida

```bash
# Opção 1: Script automático (recomendado)
./run_web.sh

# Opção 2: Comando direto
streamlit run app.py

# Opção 3: Make
make web
```

## 📋 Requisitos

```bash
pip install streamlit plotly
```

Ou instale todas as dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Funcionalidades

### 1. Upload de Vídeos
- Interface drag & drop
- Suporte para MP4, AVI, MOV, MKV
- Validação automática de formato
- Upload de até 500MB

### 2. Configuração Visual
- Seleção de backends de detecção
- Opções de processamento
- Configuração de saída
- Interface intuitiva na sidebar

### 3. Processamento
- Barra de progresso em tempo real
- Log de processamento (opcional)
- Tratamento de erros visual
- Feedback constante

### 4. Visualização de Resultados

#### Dashboard de Métricas
- Total de frames processados
- Duração do vídeo
- Faces detectadas
- Anomalias encontradas

#### Análise Detalhada (5 Abas)

**👤 Faces**
- Estatísticas de detecção
- Média por frame
- Distribuição

**😊 Emoções**
- Gráfico de barras interativo
- Percentuais por emoção
- Emojis visuais
- Top emoções

**🏃 Atividades**
- Timeline completa
- Eventos detectados
- Scores de confiança
- Resumo por tipo

**⚠️ Anomalias**
- Total e por severidade
- Alertas visuais coloridos
- Classificação (alta/média/baixa)

**📄 Relatórios**
- Download de JSON
- Download de Markdown
- Download de vídeo anotado
- Preview inline

## 🎨 Interface

### Layout

```
┌─────────────────────────────────────────────────┐
│           🎬 Análise de Vídeo com IA            │
├──────────────┬──────────────────────────────────┤
│  ⚙️ Sidebar  │         📊 Conteúdo             │
│              │                                  │
│  Upload      │  • Dashboard de Métricas        │
│  Config      │  • Tabs de Análise              │
│  Processo    │  • Visualizações                │
│  Info        │  • Downloads                    │
│              │                                  │
└──────────────┴──────────────────────────────────┘
```

### Cores e Tema

- **Primary**: Azul (#1f77b4)
- **Background**: Branco (#ffffff)
- **Secondary**: Cinza claro (#f0f2f6)
- **Text**: Cinza escuro (#262730)

## 📊 Outputs

### 1. Visualizações na Interface
- Métricas em cards
- Gráficos interativos (Plotly)
- Tabelas formatadas
- Preview de vídeo

### 2. Downloads Disponíveis
- **metrics.json**: Métricas completas
- **report.md**: Relatório formatado
- **annotated_video.mp4**: Vídeo processado

## 🔧 Configuração Avançada

### Arquivo `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
maxUploadSize = 500
enableCORS = false

[browser]
gatherUsageStats = false
```

### Variáveis de Ambiente

```bash
# Porta customizada
streamlit run app.py --server.port 8080

# Modo headless (sem abrir browser)
streamlit run app.py --server.headless true
```

## 🐛 Troubleshooting

### Problema: Streamlit não encontrado
```bash
pip install streamlit plotly
```

### Problema: Erro de upload
- Verifique o tamanho do arquivo (máx 500MB)
- Confirme o formato (MP4, AVI, MOV, MKV)

### Problema: Processamento lento
- Normal para vídeos longos
- Considere usar vídeos menores para testes
- Verifique recursos do sistema (CPU/RAM)

### Problema: Módulos não encontrados
```bash
pip install -r requirements.txt
```

## 📖 Documentação

### Estrutura do Código

```python
app.py
├── main()                    # Função principal
├── show_welcome_screen()     # Tela inicial
├── process_video()           # Processamento
├── show_results()            # Exibição de resultados
├── show_faces_stats()        # Tab de faces
├── show_emotions_stats()     # Tab de emoções
├── show_activities_stats()   # Tab de atividades
├── show_anomalies_stats()    # Tab de anomalias
└── show_reports()            # Tab de relatórios
```

### Fluxo de Execução

1. **Inicialização**: Configuração da página e estilos
2. **Sidebar**: Upload e configurações
3. **Validação**: Verificação do arquivo
4. **Processamento**: Execução do pipeline
5. **Resultados**: Visualização em tabs
6. **Downloads**: Geração e disponibilização

## 🎯 Casos de Uso

### 1. Análise Rápida
- Upload de vídeo curto
- Configuração padrão
- Visualização imediata

### 2. Análise Detalhada
- Vídeo completo
- Backends específicos
- Exportação de todos os dados

### 3. Comparação de Backends
- Mesmo vídeo
- Diferentes backends
- Análise de performance

### 4. Demonstração
- Vídeos de exemplo
- Apresentação interativa
- Explicação visual

## 💡 Dicas

1. **Performance**: Para vídeos longos (>5 min), considere processar via CLI primeiro
2. **Memória**: Feche outras abas do browser durante processamento
3. **Qualidade**: Use vídeos com boa iluminação para melhores resultados
4. **Formato**: MP4 com codec H.264 é o mais compatível
5. **Preview**: O vídeo anotado pode ser grande, use download se necessário

## 🔗 Links Úteis

- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Docs](https://plotly.com/python/)
- [OpenCV Docs](https://docs.opencv.org/)

## 📝 Changelog

### v1.0.0 (2025-11-09)
- ✨ Interface web completa
- 📊 Dashboard de métricas
- 📈 Visualizações interativas
- ⬇️ Sistema de downloads
- 🎬 Preview de vídeo
- ⚙️ Configuração visual

## 📄 Licença

Este projeto faz parte do Tech Challenge Fase 4 - FIAP
