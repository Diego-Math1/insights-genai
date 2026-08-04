import shap
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

def log_shap_explanation(model, X_train, X_test):
    """Gera e loga gráfico SHAP para explicabilidade (obrigatório em finanças)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
    plt.title("Feature Importance (SHAP) - Modelo Financeiro")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png")
    plt.close()
    return shap_values