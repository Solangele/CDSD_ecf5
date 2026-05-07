from data import load_data, preprocess
import pytest
import pandas as pd

DATA_PATH = "data/telco_churn.csv"

def test_load_data_returns_dataframe():
    """Vérifie que la fonction load_data retourne un Dataframe avec les bonnes dimensions"""
    df = load_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 7043


def test_load_data_has_expected_columns():
    """Vérifie que la fonction load_data retourne le Dataframe avec l'intégralité des variables"""
    df = load_data(DATA_PATH)
    expected_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    assert list(df.columns) == expected_columns


def test_preprocess_returns_features_and_target():
    """Vérifie que la fonction preprocess sépare correctement X et y"""
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "Churn" not in X.columns
    assert len(X) == len(y)


def test_preprocess_handles_missing_total_charges():
    """Vérifie que la convertion des espaces vides dans TotalCharges se déroule correctement"""
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    assert X['TotalCharges'].isnull().sum() == 0


