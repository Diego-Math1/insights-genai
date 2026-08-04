import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # DeepSeek
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    
    # Geral
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

settings = Settings()

# Validação rápida (opcional, mas ajuda a evitar erros bobos)
if not settings.DEEPSEEK_API_KEY:
    print("⚠️ATENÇÃO: DEEPSEEK_API_KEY não encontrada no arquivo .env!")
