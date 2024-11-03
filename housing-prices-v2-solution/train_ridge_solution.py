import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from params import ridge_param_grid
from mlflow.entities import Dataset
from utils import eval_metrics


# Lese die Daten ein
data = pd.read_csv('data/Boston.csv')

# Konvertiere alle Spalten in Typ float
data = data[data.columns].astype(float)

# Definiere Features und Zielvariable
X = data.drop('medv', axis=1).copy()  # Unabhängige Variablen
y = data['medv'].copy()               # Zielvariable

# Splitte die Daten in Trainings- und Validierungsdatensätze
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Speichere Trainings- und Validierungsdaten als CSV
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv("data/train_data.csv", index=False)
val_data = pd.concat([X_val, y_val], axis=1)
val_data.to_csv("data/val_data.csv", index=False)


# Schleife über die Hyperparameter-Kombinationen und logge Ergebnisse in separaten Runs
for params in ParameterGrid(ridge_param_grid):
    print("Parameter: ", params)
    
    # Beginne einen neuen MLflow Run
    with mlflow.start_run():

        # Initialisiere und trainiere das ElasticNet-Modell mit den aktuellen Parametern
        lr = Ridge(**params)
        lr.fit(X_train, y_train)

        # Erstelle Vorhersagen auf dem Validierungsdatensatz
        y_pred = lr.predict(X_val)

        # Logge die Train- und Validierungsdatensätze als Artefakte in MLflow
        mlflow.log_artifact("data/train_data.csv", artifact_path="data")
        mlflow.log_artifact('data/Boston.csv', artifact_path="data")
        mlflow.log_artifact('data/val_data.csv', artifact_path="data")
        
        # Logge die Eingabedaten als MLflow-Dataset
        mlflow.log_input(mlflow.data.from_numpy(train_data.to_numpy()), context='Training dataset')

        # Berechne und logge die Metriken
        metrics = eval_metrics(y_val, y_pred)

        # Logge die Hyperparameter des Modells
        mlflow.log_params(params)

        # Logge die berechneten Metriken
        mlflow.log_metrics(metrics)

        # Logge das trainierte Modell
        mlflow.sklearn.log_model(
            lr,
            "model",
            input_example=X_train,   # Beispielinput
            code_paths=['train_ridge.py', 'params.py']  # Pfade zu den verwendeten Python-Dateien
        )
