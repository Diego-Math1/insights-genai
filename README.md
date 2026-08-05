[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# 🚀 Jump Insight - GenAI Agent for Financial Data Science

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-orange)](https://mlflow.org/)
[![DeepSeek API](https://img.shields.io/badge/DeepSeek-API-4D6BFE?logo=deepseek&logoColor=white)](https://platform.deepseek.com/)

## 📌 Visão Geral
Este projeto é uma prova de conceito (PoC) desenvolvida para atender aos requisitos de uma vaga de **Cientista de Dados Sênior**, focada em **Alta Performance, Setor Financeiro Regulado e IA Generativa**. Ele simula um agente autônomo que conversa com dados massivos, treina modelos (Sklearn/PyTorch), documenta resultados e garante explicabilidade (SHAP) - tudo orquestrado com MLOps (MLflow).

## 🧠 Stack Tecnológico 
- **Manipulação de Dados:** `Pandas`, `NumPy`, `SQLAlchemy`
- **Machine Learning:** `Scikit-learn`, `PyTorch`
- **GenAI & NLP:** `LangChain` + `Azure OpenAI (GPT-4)`
- **MLOps & Deploy:** `MLflow` (tracking e registro de modelos)
- **Explicabilidade:** `SHAP` (essencial para regulação)
- **Cloud Ready:** Projetado para `Azure` e `Databricks` (via MLflow)

📄 Veja o [relatório completo de performance](reports/model_performance.md) para mais detalhes.

### ⚙️ Executando o projeto

```bash
# Ative o ambiente virtual
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Execute o projeto (com suporte ao MLflow)
export MLFLOW_ALLOW_FILE_STORE=true
python main.py

## Estrutura

├── config/                    # Configurações centralizadas
├── src/                       # Código fonte modular
│   ├── data_loader.py         # Carregamento de dados (real ou sintético)
│   ├── feature_engineering.py # Preparação automática de features
│   ├── model_trainer.py       # Treino com Sklearn + PyTorch + MLflow
│   ├── explainability.py      # SHAP para explicabilidade
│   ├── agent.py               # Agente GenAI com DeepSeek
│   └── utils.py               # Funções auxiliares
├── notebooks/                 # Análises exploratórias (futuras)
├── tests/                     # Testes unitários (futuros)
├── reports/                   # 📊 RELATÓRIOS GERADOS
│   └── model_performance.md   # Relatório de performance do modelo
├── data/                      # Datasets
│   └── raw/                   # Dados brutos (ignorados pelo Git)
├── main.py                    # Orquestrador principal
├── requirements.txt           # Dependências do projeto
├── setup.py                   # Configuração do pacote
├── .env.example               # Exemplo de variáveis de ambiente
├── .gitignore                 # Arquivos ignorados pelo Git
├── LICENSE                    # Licença MIT
├── README.md                  # Este arquivo
└── historico_perguntas.csv    # 💬 Histórico de conversas com o agente

## Artefatos

O projeto gera automaticamente os seguintes artefatos durante a execução:

| Arquivo | Descrição |
| :--- | :--- |
| `reports/model_performance.md` | Relatório detalhado com métricas (AUC, Recall, Feature Importance) |
| `historico_perguntas.csv` | Histórico completo das perguntas feitas ao agente e suas respostas |
| `shap_summary.png` | Gráfico de importância das features (SHAP) para explicabilidade |
| `mlruns/` | Experimentos e modelos versionados pelo MLflow |

### 📈 Resultados Obtidos

O projeto foi testado com um dataset real de 10.000 transações financeiras, alcançando os seguintes resultados:

| Métrica | Valor |
| :--- | :--- |
| **ROC-AUC** | **0.9993** |
| **Recall (Fraude)** | **98%** |
| **Precisão (Fraude)** | 85% |
| **Feature Mais Importante** | `transaction_hour` (27% de importância) |

### 🔍 Principais Insights
- **Horário da transação** é o maior preditor de fraude (transações entre 0h-6h são altamente suspeitas).
- **Dispositivos com baixa confiança** (trust score < 40) aumentam o risco em 6x.
- O modelo captura **98% das fraudes**, com apenas 15% de falsos positivos.
> 💡 **Dica:** O relatório de performance mostra que o modelo atingiu **AUC 0.9993** e **Recall de 98%** para detecção de fraudes.

