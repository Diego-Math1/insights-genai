import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config.settings import settings

def prepare_features(df: pd.DataFrame, target_col: str = None, test_size: float = 0.3):
    """
    Prepara features para ML de forma automática.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo. Se None, tenta detectar automaticamente.
        test_size: Proporção do conjunto de teste.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    
    # 1. Detecta a coluna alvo (target)
    if target_col is None:
        # Lista de nomes comuns para coluna alvo (fraude/classificação)
        possible_targets = ['flag_fraude', 'fraude', 'target', 'Class', 'classe', 'label', 'y']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        # Se não encontrou, assume que a última coluna é a alvo (padrão em muitos datasets)
        if target_col is None:
            target_col = df.columns[-1]
            print(f"⚠️ Coluna alvo não identificada. Usando a última coluna: '{target_col}'")
    
    print(f"🎯 Coluna alvo detectada: '{target_col}'")
    
    # 2. Separa features (X) e target (y)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 3. Remove colunas de identificação (ID) para não vazar informação
    id_cols = [col for col in X.columns if 'id' in col.lower()]
    if id_cols:
        print(f"🗑️ Removendo colunas de ID: {id_cols}")
        X = X.drop(columns=id_cols)
    
    # 4. Separa colunas numéricas e categóricas
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"📊 Colunas numéricas: {num_cols}")
    print(f"📊 Colunas categóricas: {cat_cols}")
    
    # 5. Trata valores nulos (opcional: preenche com média/moda)
    if X.isnull().sum().sum() > 0:
        print("⚠️ Valores nulos detectados. Preenchendo...")
        for col in num_cols:
            X[col] = X[col].fillna(X[col].mean())
        for col in cat_cols:
            X[col] = X[col].fillna('Desconhecido')
    
    # 6. One-Hot Encoding para colunas categóricas
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        print(f"✅ One-Hot Encoding aplicado. Novas features: {X.shape[1]}")
    
    # 7. Padronização das colunas numéricas (ainda existentes)
    scaler = StandardScaler()
    # Apenas as numéricas que não foram transformadas em dummies
    num_cols_remaining = X.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    if num_cols_remaining:
        X[num_cols_remaining] = scaler.fit_transform(X[num_cols_remaining])
    
    # 8. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=settings.RANDOM_SEED, stratify=y
    )
    
    print(f"✅ Dados preparados: {X_train.shape[1]} features, {len(X_train)} treino, {len(X_test)} teste")
    
    return X_train, X_test, y_train, y_test, scaler
