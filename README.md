# 🚀 Jump Insight - GenAI Agent for Financial Data Science

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-orange)](https://mlflow.org/)
[![DeepSeek API](https://img.shields.io/badge/DeepSeek-API-4D6BFE?logo=deepseek&logoColor=white)](https://platform.deepseek.com/)

## 📌 Visão Geral
Este projeto é uma prova de conceito (PoC) desenvolvida para atender aos requisitos de uma vaga de **Cientista de Dados Sênior**, focada em **Alta Performance, Setor Financeiro Regulado e IA Generativa**. Ele simula um agente autônomo que conversa com dados massivos, treina modelos (Sklearn/PyTorch), documenta resultados e garante explicabilidade (SHAP) - tudo orquestrado com MLOps (MLflow).

## 🧠 Stack Tecnológico (Alinhado 100% com a Vaga)
- **Manipulação de Dados:** `Pandas`, `NumPy`, `SQLAlchemy`
- **Machine Learning:** `Scikit-learn`, `PyTorch`
- **GenAI & NLP:** `LangChain` + `Azure OpenAI (GPT-4)`
- **MLOps & Deploy:** `MLflow` (tracking e registro de modelos)
- **Explicabilidade:** `SHAP` (essencial para regulação)
- **Cloud Ready:** Projetado para `Azure` e `Databricks` (via MLflow)

## ⚙️ Como Executar

1. **Clone o repositório:**
```bash
git clone https://github.com/Diego-Math1/jump-insight-genai.git
cd jump-insight-genai

📌 Estrutura do projeto
├── config/            # Configurações centralizadas
├── src/               # Código fonte modular (Data, Features, Models, Agent)
├── notebooks/         # Análises exploratórias
├── tests/             # Testes unitários
├── data/              # Datasets (raw/processado)
├── main.py            # Orquestrador principal
└── README.md          # Este arquivo