import pandas as pd
import numpy as np
from config.settings import settings
import os

def load_financial_data(n_rows: int = None) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV real (se existir) ou gera dados sintéticos.
    
    Args:
        n_rows: número de linhas a carregar (se None, carrega tudo)
    """
    # Caminho para o dataset real
    real_data_path = "data/raw/transacoes_reais.csv"
    
    # Se o arquivo existir, carrega ele
    if os.path.exists(real_data_path):
        print(f"📂 Carregando dataset real de: {real_data_path}")
        df = pd.read_csv(real_data_path)
        
        # Se n_rows for especificado, pega apenas as primeiras N linhas
        if n_rows is not None and n_rows < len(df):
            df = df.head(n_rows)
            
        print(f"✅ Dataset real carregado com {len(df)} registros e {len(df.columns)} colunas.")
        return df
    
    # Se não existir, gera dados sintéticos (fallback)
    else:
        print("⚠️ Arquivo real não encontrado. Gerando dados sintéticos...")
        return _generate_synthetic_data(n_rows or 50000)

def _generate_synthetic_data(n_rows: int) -> pd.DataFrame:
    """Função auxiliar que gera dados sintéticos (igual ao código anterior)."""
    np.random.seed(settings.RANDOM_SEED)
    data = {
        'cliente_id': np.arange(1, n_rows + 1),
        'valor_transacao': np.random.exponential(500, n_rows),
        'idade': np.clip(np.random.normal(40, 15, n_rows).astype(int), 18, 90),
        'tempo_conta_meses': np.random.poisson(60, n_rows),
        'descricao_comercio': np.random.choice(
            ['Pagamento Servico', 'Compra Online', 'Saque ATM', 'Transferencia', 'Investimento'], n_rows
        ),
        'qtd_parcelas': np.random.choice([1, 3, 6, 12], n_rows, p=[0.6, 0.2, 0.15, 0.05]),
        'flag_fraude': np.random.binomial(1, 0.03, n_rows)
    }
    df = pd.DataFrame(data)
    df.loc[df['qtd_parcelas'] > 6, 'flag_fraude'] = np.random.binomial(1, 0.01, sum(df['qtd_parcelas'] > 6))
    return df
