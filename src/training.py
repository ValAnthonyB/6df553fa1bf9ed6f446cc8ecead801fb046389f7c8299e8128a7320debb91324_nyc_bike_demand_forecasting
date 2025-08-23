import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_best_rf_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_trials: int = 25,
) -> RandomForestRegressor:
    """
    Performs hyperparameter tuning using Optuna to automatically find the best
    Random Forest regression model.
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
    RandomForestRegressor
        Trained Random Forest model.
    """

    def objective(trial):
        # Parameter space for Random Forest
        params = {
            "n_estimators": 300,
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "n_jobs": -1,
            "random_state": 42,
        }

        # Train the Random Forest model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Get predictions and calculate RMSE
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return rmse

    # Run hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Best params
    print("Best hyperparameters:", study.best_params)

    # Train final model with best params
    best_params = study.best_params
    best_rf_model = RandomForestRegressor(
        **best_params, n_estimators=300, random_state=42, n_jobs=-1
    )
    best_rf_model.fit(X_train, y_train)

    return best_rf_model


def export_model(model: RandomForestRegressor, path: str) -> None:
    """
    After training, the model is saved to disk in a joblib file.

    Parameters:
    ----------
    model : RandomForestRegressor
        The trained model.
    path : str
        Destination path for saving the model artifact.
    """
    # Export the model using joblib
    joblib.dump(model, path)
