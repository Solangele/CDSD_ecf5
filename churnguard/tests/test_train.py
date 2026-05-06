from train import train_model
from data import load_data, preprocess
from evaluate import compute_metrics

from sklearn.pipeline import Pipeline
import pytest
import pandas as pd

DATA_PATH = "data/telco_churn.csv"

def test_train_model_returns_fitted_pipeline():
    """Vérifie que la fonction train_model entraîne et est capable faire des prédictions"""
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    
    params = {"n_estimators": 10, "max_depth": 2}
    model = train_model(X, y, "rf_test", params)
    
    assert isinstance(model, Pipeline)
    preds = model.predict(X.head(5))
    assert len(preds) == 5


def test_compute_metrics_returns_expected_keys():
    """Vérifie que la fonction compute_metrics renvoi bien les 5 metriques attendus (accuracy, precision, recall, f1 et roc-auc)"""
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    model = train_model(X, y, "rf_test", {"n_estimators": 10})
    metrics = compute_metrics(model, X, y)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}

    assert expected_keys.issubset(metrics.keys())
    for val in metrics.values():
        assert isinstance(val, float)
