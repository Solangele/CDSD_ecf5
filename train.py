import mlflow
import mlflow.sklearn
import shutil
import os
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from mlflow.tracking import MlflowClient


from data import load_data, preprocess
from evaluate import compute_metrics


def train_model(X, y, model_name: str, params: dict) -> Pipeline:
    """Entraîne le modèle et retourne le pipeline (Fonction demandée étape 1.2)."""

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess_pipe = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Sélection de l'algorithme selon le nom (pour l'étape 1.4)
    if "Logistic" in model_name:
        clf = LogisticRegression(**params)
    elif "Gradient" in model_name:
        clf = GradientBoostingClassifier(**params)
    else:
        clf = RandomForestClassifier(**params)

    model = Pipeline(
        [
            ("prep", preprocess_pipe),
            ("clf", clf),
        ]
    )

    model.fit(X, y)
    return model


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Telco_Churn_Industrialization")

    df = load_data("data/telco_churn.csv")
    X, y = preprocess(df)

    experiments = [
        ("Logistic_Regression", {"max_iter": 1000}),
        ("Random_Forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        (
            "Gradient_Boosting",
            {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        ),
    ]

    for run_name, params in experiments:
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Lancement de l'entraînement : {run_name}")

            model = train_model(X, y, run_name, params)

            metrics = compute_metrics(model, X, y)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X, model.predict(X))
            input_example = X.head(1)

            temp_model_path = f"temp_model_{run_name}"
            if os.path.exists(temp_model_path):
                shutil.rmtree(temp_model_path)

            mlflow.sklearn.save_model(
                sk_model=model,
                path=temp_model_path,
                signature=signature,
                input_example=input_example,
            )

            # 2. Envoi vers le serveur MLflow comme un simple dossier d'artéfacts
            mlflow.log_artifacts(temp_model_path, artifact_path="model")

            # 3. Nettoyage
            shutil.rmtree(temp_model_path)
            run_id = run.info.run_id
            print(f"Modèle loggé avec succès ! Run ID: {run_id}")

    # --- ÉTAPE 1.5 AUTOMATISÉE : SÉLECTION DU MEILLEUR MODÈLE ---
    print("Analyse des résultats pour sélection du meilleur modèle...")

    # 1. On récupère tous les runs de l'expérience
    experiment = mlflow.get_experiment_by_name("Telco_Churn_Industrialization")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # 2. On trie par F1-Score (tu peux changer par 'metrics.accuracy' si tu préfères)
    # On s'assure que la colonne existe pour éviter les erreurs
    best_run = runs.sort_values(by="metrics.f1", ascending=False).iloc[0]

    best_run_id = best_run["run_id"]
    best_model_name = best_run["tags.mlflow.runName"]
    best_f1 = best_run["metrics.f1"]

    print(f"Le meilleur modèle est {best_model_name} avec un F1-score de {best_f1:.4f}")

    # 3. Enregistrement dans le Registry
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, "churnguard")

    # 4. Promotion immédiate en Production
    client = MlflowClient()
    client.transition_model_version_stage(
        name="churnguard", version=result.version, stage="Production"
    )

    print(
        f"Succès : {best_model_name} (v{result.version}) est désormais en Production."
    )

    # --- VÉRIFICATION DU CHARGEMENT (Demande de l'étape 1.5) ---
    print("Vérification du chargement du modèle promu...")
    prod_model = mlflow.pyfunc.load_model("models:/churnguard/Production")
    print("Le modèle Production est opérationnel !")


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import pandas as pd

# def train_model(X: pd.DataFrame, y: pd.Series, model_name: str, params: dict) -> Pipeline:
#     """Configure le pipeline de preprocessing et entraîne le modèle RandomForest."""

#     num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
#     cat_cols = [c for c in X.columns if c not in num_cols]

#     preprocess_pipe = ColumnTransformer([
#         ('num', StandardScaler(), num_cols),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
#     ])

#     model = Pipeline([
#         ('prep', preprocess_pipe),
#         ('clf', RandomForestClassifier(**params)),
#     ])

#     model.fit(X, y)
#     return model
