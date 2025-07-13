import numpy as np
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import joblib

def get_best_lightgbm_model(X_train, X_test, y_train, y_test, n_trials=20):
    def objective(trial):
        # Parameter space
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 200,  # Fixed
            "num_leaves": trial.suggest_int("num_leaves", 3, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "force_col_wise": True
        }
        
        # Train model
        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            callbacks=[early_stopping(100, verbose=False)]
        )
        
        # Get metrics
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Best params
    print("Best hyperparameters:", study.best_params)

    # Train final model with best params
    best_params = study.best_params
    best_lgbm_model = LGBMRegressor(
        **best_params,
        n_estimators=200,
        random_state=42,
        verbosity=-1  # Suppress model messages
    )
    best_lgbm_model.fit(X_train, y_train)  # No verbose parameter needed here

    return best_lgbm_model

def export_model(model, path):
    joblib.dump(model, path)