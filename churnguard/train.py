from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def train_model(X: pd.DataFrame, y: pd.Series, model_name: str, params: dict) -> Pipeline:
    """Configure le pipeline de preprocessing et entraîne le modèle RandomForest."""
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess_pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    model = Pipeline([
        ('prep', preprocess_pipe),
        ('clf', RandomForestClassifier(**params)),
    ])

    model.fit(X, y)
    return model