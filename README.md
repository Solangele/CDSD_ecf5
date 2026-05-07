## Résultat de la vérification des tests (1.3)

```bash
python -m pytest --cov=. tests/
================================================================================================ test session starts ================================================================================================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\Administrateur\Desktop\CDSD_ecf5\churnguard
plugins: anyio-4.12.1, cov-7.1.0
collected 6 items                                                                                                                                                                                                    

tests\test_data.py ....                                                                                                                                                                                        [ 66%]
tests\test_train.py ..                                                                                                                                                                                         [100%]

================================================================================================== tests coverage ===================================================================================================
__________________________________________________________________________________ coverage: platform win32, python 3.11.9-final-0 __________________________________________________________________________________

Name                  Stmts   Miss  Cover
-----------------------------------------
__init__.py               0      0   100%
data.py                  11      0   100%
evaluate.py               7      0   100%
tests\__init__.py         0      0   100%
tests\test_data.py       23      0   100%
tests\test_train.py      24      0   100%
train.py                 12      0   100%
-----------------------------------------
TOTAL                    77      0   100%
================================================================================================= 6 passed in 3.88s =================================================================================================
```


PROJET_CHURN/
├── .github/workflows/       # CI/CD (GitHub Actions)
│   └── ci.yaml
├── data/                    # Données (ignorées par git mais présentes pour Docker)
│   └── telco_churn.csv
├── churnguard/              # Ton package (Code source)
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── api.py               <-- Nouveau : Ton serveur FastAPI
├── tests/                   # Tes tests unitaires
│   ├── test_data.py
│   └── test_train.py
├── scripts/                 # Utilitaires
│   └── download_data.py
├── Dockerfile               <-- Nouveau : Image pour l'API
├── docker-compose.yaml      <-- Nouveau : Orchestration (API + MLflow)
├── requirements.txt         <-- Nouveau : Dépendances
└── mlflow.db                # Base SQLite pour MLflow (générée)



## Création des 3 modèles
![alt text](image.png)

### Lancement des modèles avec MLflow et choix du meilleur
![alt text](image-1.png)

## API
![alt text](image-2.png)

## Lancement de docker Api et MLflow 
![alt text](image-3.png)
![alt text](image-4.png)