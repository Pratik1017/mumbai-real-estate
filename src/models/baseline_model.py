import pandas as pd
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score , mean_squared_error
from src.features.feature_engineering import create_trf
import mlflow
import dagshub
import yaml

with open("params.yaml","r") as f:
    params = yaml.safe_load(f)

    learning_rate = params['baseline_model']['learning_rate']
    max_depth = params['baseline_model']['max_depth']
    random_state = params['baseline_model']['random_state']
    subsample = params['baseline_model']['subsample']



dagshub.init(repo_owner='Pratik1017', repo_name='mumbai-real-estate', mlflow=True)

import mlflow

mlflow.set_experiment("XGBoost_Baseline")


def create_pipe():
    try:
        pipe = Pipeline(
             steps=[
                  ("trf1",create_trf()),
                  ("scaler",StandardScaler()),
                  ('model' , XGBRegressor(
                      learning_rate = learning_rate,
                      max_depth = max_depth,
                      subsample = subsample,
                      random_state=random_state
                  ))
             ]
        )
        return pipe

    except Exception as e:
            
            print(f"Error loading data: {e}")
            return None


def predict(X_train, X_test, y_train, y_test, pipe):
    try:
        with mlflow.start_run(run_name="XGBoost_Baseline"):

            # Log model parameters
            model = pipe.named_steps["model"]

            mlflow.log_params({
                "model_name": "XGBRegressor",
                "learning_rate": model.learning_rate,
                "max_depth": model.max_depth,
                "subsample": model.subsample,
                "random_state": model.random_state
            })

            # Train model
            pipe.fit(X_train, np.log1p(y_train))

            # Predictions
            y_train_pred = pipe.predict(X_train)
            y_test_pred = pipe.predict(X_test)

            # Metrics
            metrics = {
                "train_r2": r2_score(y_train, np.expm1(y_train_pred)),
                "test_r2": r2_score(y_test, np.expm1(y_test_pred)),
                "test_mae": mean_absolute_error(
                    y_test, np.expm1(y_test_pred)
                ),
                "test_rmse": np.sqrt(
                    mean_squared_error(y_test, np.expm1(y_test_pred))
                )
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log trained pipeline
            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="xgboost_pipeline"
            )

            print("Metrics:", metrics)

            return metrics

    except Exception as e:
        print(f"Error during model training: {e}")
        return None


if __name__ == "__main__":

    # Data Loading
    X_train = pd.read_csv("data/interim/X_train_cleaned_v2.csv")
    X_test = pd.read_csv("data/interim/X_test_cleaned_v2.csv")
    y_train = pd.read_csv("data/raw/y_train.csv")
    y_test = pd.read_csv("data/raw/y_test.csv")

    pipe = create_pipe()

    predict(X_train, X_test , y_train , y_test, pipe )