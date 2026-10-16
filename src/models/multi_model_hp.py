
import pandas as pd
import optuna
import mlflow

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor

from src.features.feature_engineering import create_trf
import dagshub

dagshub.init(repo_owner='Pratik1017', repo_name='mumbai-real-estate', mlflow=True)


X_train = None
y_train = None


def objective(trial):

    regressor_name = trial.suggest_categorical(
        "regressor_name",
        ["XGBoost", "Random", "Gradient", "Light"]
    )

    if regressor_name == "XGBoost":

        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            random_state=42,
            n_jobs=-1
        )

    elif regressor_name == "Random":

        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            max_features=trial.suggest_categorical(
                "max_features", [1.0, "sqrt", "log2"]
            ),
            random_state=42,
            n_jobs=-1
        )

    elif regressor_name == "Gradient":

        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=42
        )

    else:

        model = LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            num_leaves=trial.suggest_int("num_leaves", 15, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

    pipe = Pipeline(
        steps=[
            ("trf1", create_trf()),
            ("scaler", StandardScaler()),
            ("model", model)
        ]
    )

    with mlflow.start_run(nested=True):

        mlflow.log_params(trial.params)

        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=5,
            scoring="r2",
            n_jobs=1
        )

        mean_r2 = scores.mean()

        mlflow.log_metric("mean_cv_r2", mean_r2)
        mlflow.log_metric("std_cv_r2", scores.std())

        print(
            f"Model: {regressor_name} | "
            f"Mean CV R2: {mean_r2:.4f}"
        )

    return mean_r2


if __name__ == "__main__":

    # Data Loading
    X_train = pd.read_csv(
        "data/processed/X_train_cleaned_v2.csv"
    )

    y_train = pd.read_csv(
        "data/raw/y_train.csv"
    ).squeeze("columns")

    mlflow.set_experiment("Multiple_Model_HP_Optuna")

    with mlflow.start_run(run_name="Optuna_Optimization"):

        study = optuna.create_study(direction="maximize")

        study.optimize(objective, n_trials=30)

        print("Best CV R2:", study.best_value)
        print("Best Parameters:", study.best_params)

        mlflow.log_metric("best_cv_r2", study.best_value)
        mlflow.log_params({
            f"best_{key}": value
            for key, value in study.best_params.items()
        })

        # Save Best Parameters
        best_params = pd.DataFrame(
            [study.best_params]
        )

        best_params.to_json(
            "best_params.json",
            orient="records",
            indent=4
        )

        mlflow.log_artifact(
            "best_params.json"
        )

        
