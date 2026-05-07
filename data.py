import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset CSV brut depuis le chemin spécifié."""
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nettoie les données, convertit les types et sépare les features de la cible."""
    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean = df_clean.dropna()
    df_clean = df_clean.drop(columns=['customerID'])
    
    y = (df_clean['Churn'] == 'Yes').astype(int)
    X = df_clean.drop(columns=['Churn'])
    
    return X, y