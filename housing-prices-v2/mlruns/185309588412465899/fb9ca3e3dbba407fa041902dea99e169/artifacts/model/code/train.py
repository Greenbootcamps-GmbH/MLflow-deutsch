import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import ParameterGrid
from params import ridge_param_grid, elasticnet_param_grid
from mlflow.entities import Dataset
from utils import eval_metrics


# Reading data
data = pd.read_csv('data/Boston.csv')

data = data[data.columns].astype(float)

# Define features and target variable
X = data.drop('medv', axis=1).copy()
y = data['medv'].copy()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv("data/train_data.csv", index=False)
val_data = pd.concat([X_val, y_val], axis=1)
val_data.to_csv("data/val_data.csv", index=False)

# experiment_name = "ElasticNet-v3"
# exp = mlflow.set_experiment(experiment_name=experiment_name)
# print(exp.experiment_id)

# Loop through the hyperparameter combinations and log results in separate runs
for params in ParameterGrid(elasticnet_param_grid):
    print()
    print(params)
    print()
    with mlflow.start_run():

        lr = ElasticNet(**params)

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        mlflow.log_artifact("data/train_data.csv", artifact_path="data")
        mlflow.log_artifact('data/Boston.csv', artifact_path="data")
        mlflow.log_artifact('data/val_data.csv', artifact_path="data")
        mlflow.log_input(mlflow.data.from_numpy(train_data.to_numpy()), context='Training dataset')

        metrics = eval_metrics(y_val, y_pred)

        # Logging hyperparameters
        mlflow.log_params(params)

        # Logging metrics
        mlflow.log_metrics(metrics)

        # Log the trained model
        mlflow.sklearn.log_model(
            lr,
            "model",
             input_example=X_train,
             code_paths=['train.py','params.py']
        )