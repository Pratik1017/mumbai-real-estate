
import pandas as pd
import optuna
import mlflow
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from src.features.feature_engineering import create_trf
import dagshub
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dagshub.init(repo_owner='Pratik1017', repo_name='mumbai-real-estate', mlflow=True)



with open("params.yaml","r") as f:
    params = yaml.safe_load(f)

    n_estimators = params['final_model']['n_estimators']
    max_depth = params['final_model']['max_depth']
    min_samples_split = params['final_model']['min_samples_split']
    min_samples_leaf = params['final_model']['min_samples_leaf']
    max_features = params['final_model']['max_features']


mlflow.set_experiment("Random_Forest_Final_Model")



def create_pipe():
    try:
        pipe = Pipeline(
             steps=[
                  ("trf1",create_trf()),
                  ("scaler",StandardScaler()),
                  ('model' ,RandomForestRegressor(
                        n_estimators= 109,
                        max_depth= 21,
                        min_samples_split= 13,
                        min_samples_leaf= 3,
                        max_features= "sqrt"
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
                "model_name": "RandomForestRegressor",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
                "max_features": model.max_features
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
                name="random_forest_pipeline"
            )

            print("Metrics:", metrics)

            return metrics

    except Exception as e:
        print(f"Error during model training: {e}")
        return None



if __name__ == "__main__":

    # Data Loading
    X_train = pd.read_csv("data/processed/X_train_cleaned_v2.csv")
    X_test = pd.read_csv("data/processed/X_test_cleaned_v2.csv")
    y_train = pd.read_csv("data/raw/y_train.csv")
    y_test = pd.read_csv("data/raw/y_test.csv")

    pipe = create_pipe()

    predict(X_train, X_test , y_train , y_test, pipe)