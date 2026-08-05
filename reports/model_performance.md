# Relatório de Performance - Modelo de Detecção de Fraudes

## 📊 Visão Geral
- **Dataset:** 10.000 transações financeiras, dataset extraído de 
"https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset"
- **Taxa de Fraude:** 1,51% (151 casos)
- **Features utilizadas:** 11 (após One-Hot Encoding)

## 🏆 Métricas do Modelo (Random Forest)
| Métrica | Valor |
| :--- | :--- |
| **ROC-AUC** | **0.9993** |
| **Recall (Fraude)** | **0.98** |
| **Precisão (Fraude)** | 0.85 |
| **Acurácia** | 1.00 |

## 🔍 Feature Importance (Top 5)
| Feature | Importância | Impacto |
| :--- | :--- | :--- |
| **transaction_hour** | 27.08% | Horário da transação (madrugada = maior risco) |
| **device_trust_score** | 26.08% | Confiança do dispositivo (baixo = maior risco) |
| **velocity_last_24h** | 13.86% | Movimentação recente da conta |
| **location_mismatch** | 12.81% | Inconsistência de localização |
| **foreign_transaction** | 12.74% | Transação internacional |

## 💡 Insights de Negócio
1. **Horário é crítico:** Fraudes ocorrem 3x mais entre 0h e 6h.
2. **Dispositivo confiável:** Trust Score médio em fraudes é 37 (vs 62 em legítimas).
3. **Transações internacionais:** Aumentam o risco de fraude em 6x.
4. **Parcelamento:** (se aplicável ao dataset) não foi a feature principal, mas pode ser explorada.

## 🚀 Recomendações para Produção
- Deploy no **Azure Databricks** com MLflow.
- Monitorar drift das features `transaction_hour` e `device_trust_score`.
- Criar regras de negócio em tempo real para transações com **hora < 6h** e **trust_score < 40**.
