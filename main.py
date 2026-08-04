import pandas as pd
from src.data_loader import load_financial_data
from src.feature_engineering import prepare_features
from src.model_trainer import train_and_log_models
from src.explainability import log_shap_explanation
from src.agent import get_jump_agent
import mlflow

def main():
    print("🚀 Iniciando Jump Insight GenAI Project...")
    
    # 1. Dados
    df = load_financial_data(50000)  # Menor para teste rápido
    print(f"✅ Dataset carregado: {df.shape}")
    
    # 2. Features
    X_train, X_test, y_train, y_test, _ = prepare_features(df)
    print(f"✅ Features preparadas: {X_train.shape[1]} dimensões")
    
    # 3. Modelo + MLflow
    model, metrics = train_and_log_models(X_train, y_train, X_test, y_test)
    print(f"✅ Modelo treinado. AUC: {metrics['RF_AUC']:.4f}")
    
    # 4. Explicabilidade (SHAP)
    log_shap_explanation(model, X_train, X_test)
    print("✅ Explicabilidade SHAP gerada e logada.")
    
    # 5. Agente GenAI (interage com a vaga)
    print("\n--- 🤖 AGENTE GENAI EM AÇÃO ---")
    agent = get_jump_agent(df)
    
    perguntas = [
        "Qual o perfil de idade que mais comete fraudes?",
        "Considerando os dados, qual seria a feature mais importante para prever fraude? Justifique.",
        "Escreva um relatório técnico executivo de 3 parágrafos sobre este projeto, citando uso de Python, Sklearn, PyTorch e deploy em Azure para um time não técnico."
    ]
    
    for q in perguntas:
        print(f"\n❓ Pergunta: {q}")
        resposta = agent.run(q)
        print(f"📝 Resposta:\n{resposta}\n{'-'*50}")
    
    print("\n🎯 Projeto finalizado. Confira os artifacts no diretório 'mlruns'.")

if __name__ == "__main__":
    main()