import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_best_lightgbm_model(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    n_trials: int = 20
) -> LGBMRegressor:
    """
    Performs hyperparameter tuning using Optuna to automatically find the best
    LightGBM regression model.

    The model is trained and evaluated using the RMSE (Root Mean Squared Error) metric.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training features.

    X_test : pd.DataFrame
        Testing features.

    y_train : pd.Series
        Training labels.

    y_test : pd.Series
        Testing labels.

    n_trials : int, optional (default=20)
        Number of Optuna trials for hyperparameter search.
        
    Returns:
    -------
    LGBMRegressor
        Trained LightGBM model.
    """

    def objective(trial):
        # Parameter space
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 200,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64),
            
            # Regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0),

            # Optional: Light regularization on splits
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "force_col_wise": True
        }
        
        # Train the LightGBM model
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


def export_model(model: LGBMRegressor, path: str) -> None:
    """
    After training, the model is saved to disk in a joblib file.

    Parameters:
    ----------
    model : LGBMRegressor
        The trained model.

    path : str
        Destination path for saving the model artifact.
    """

    # Export the model using joblib
    joblib.dump(model, path)