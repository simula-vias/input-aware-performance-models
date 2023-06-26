import warnings

import numpy as np

from modAL.models import CommitteeRegressor

from catboost import CatBoostRegressor

from functools import partial
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from hyperopt import hp, tpe, space_eval
from hyperopt.pyll import scope
from hyperopt.fmin import fmin
from hyperopt import STATUS_OK, Trials

TUNE_BUDGET = 100


def get_model(
    model_name,
    random_state,
    X_train,
    y_train,
    tune_hyperparams=False,
):
    if model_name.lower() == "gp":
        return get_optimizer_gp(
            X_train,
            y_train,
            random_state,
            tune_hyperparams=tune_hyperparams,
        )
    elif model_name.lower() == "rf":
        return get_optimizer_rf_ensemble(
            X_train,
            y_train,
            random_state,
            tune_hyperparams=tune_hyperparams,
        )
    elif model_name.lower() == "gbdt-cb":
        return get_optimizer_catboost(
            X_train,
            y_train,
            random_state,
            tune_hyperparams=tune_hyperparams,
        )
    else:
        raise Exception(f"Model {model_name} not supported.")


def get_optimizer_gp(
    X_train,
    y_train,
    random_state,
    tune_hyperparams,
):
    # if prev_model is not None:
    #     model = prev_model

    if tune_hyperparams:
        params = tune_hyperparams_gp(X_train, y_train, random_state=random_state)
    else:
        params = {
            "length_scale": 1.0,
            "length_scale_bound": 5,
            "constant_value": 1.0,
            "nu": 1.5,
            "n_restarts_optimizer": 0,
        }

    kernel = ConstantKernel(
        constant_value=params["constant_value"],
        constant_value_bounds="fixed",
    ) + Matern(
        length_scale=params["length_scale"],
        nu=params["nu"],
        length_scale_bounds=(
            10 ** -params["length_scale_bound"],
            10 ** params["length_scale_bound"],
        ),
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=random_state,
        n_restarts_optimizer=params["n_restarts_optimizer"],
    )
    model.fit(X_train, y_train)

    return model, params


def tune_hyperparams_gp(X, y, budget=TUNE_BUDGET, random_state=None):
    space = {
        "length_scale": hp.lognormal("length_scale", 0.1, 2),
        "length_scale_bound": hp.choice("length_scale_bound", [5, 6, 7, 8]),
        "constant_value": hp.uniform("constant_value", 0.1, 3),
        "n_restarts_optimizer": scope.int(
            hp.quniform("n_restarts_optimizer", 0, 5, q=1)
        ),
        "nu": hp.choice("nu", [0.5, 1.5, 2.5, float("inf")]),
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(params, random_state=random_state, cv=kf, X=X, y=y):
        losses = []

        for train_index, test_index in cv.split(X):
            kernel = ConstantKernel(
                constant_value=params["constant_value"], constant_value_bounds="fixed"
            ) + Matern(
                length_scale=params["length_scale"],
                nu=params["nu"],
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=random_state,
                n_restarts_optimizer=params["n_restarts_optimizer"],
            )

            model.fit(X[train_index], y[train_index])
            y_pred = model.predict(X[test_index])
            score = mean_squared_error(y[test_index], y_pred, squared=False)
            losses.append(score)

        results = {"loss": np.mean(losses), "status": STATUS_OK}
        return results

    TRIALS = Trials()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(
            fn=objective, space=space, algo=tpe.suggest, max_evals=budget, trials=TRIALS
        )

    best_params = space_eval(space, best)
    return best_params


def get_optimizer_rf_ensemble(
    X_train,
    y_train,
    random_state,
    tune_hyperparams,
):
    if tune_hyperparams:
        params = tune_hyperparams_rf(X_train, y_train, random_state=random_state)
    else:
        params = {
            "max_depth": 10,
            "n_estimators": 100,
            "max_features": "auto",
            "max_samples": None,
        }

    model = train_rf_ensemble(X_train, y_train, params, random_state)
    return model, params


def train_rf_ensemble(X_train, y_train, params, random_state, ensemble_size=10):
    members = []

    for ens_idx in range(ensemble_size):
        rgr = RandomForestRegressor(
            random_state=random_state * (ens_idx + 1),
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            max_features=params["max_features"],
            max_samples=params["max_samples"],
        )
        rgr.fit(X_train, y_train)
        members.append(rgr)

    model = CommitteeRegressor(members)
    return model


def tune_hyperparams_rf(X, y, budget=TUNE_BUDGET, random_state=None):
    space = {
        "max_depth": hp.choice("max_depth", [4, 6, 8, None]),
        "n_estimators": hp.choice("n_estimators", [50, 100, 150, 200, 250]),
        "max_samples": hp.choice("max_samples", [0.5, 0.75, 1.0]),
        "max_features": hp.choice("max_features", [0.5, 0.75, 1.0]),
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(params, random_state=random_state, cv=kf, X=X, y=y):
        losses = []

        for train_index, test_index in cv.split(X):
            model_ensemble = train_rf_ensemble(
                X[train_index], y[train_index], params, random_state
            )
            y_pred = model_ensemble.predict(X[test_index])
            score = mean_squared_error(y[test_index], y_pred, squared=False)
            losses.append(score)

        results = {"loss": np.mean(losses), "status": STATUS_OK}
        return results

    TRIALS = Trials()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(
            fn=objective, space=space, algo=tpe.suggest, max_evals=budget, trials=TRIALS
        )

    best_params = space_eval(space, best)
    return best_params


def get_optimizer_catboost(
    X_train,
    y_train,
    random_state,
    tune_hyperparams,
):
    if tune_hyperparams:
        params = tune_hyperparams_catboost(X_train, y_train, random_state=random_state)
    else:
        params = {
            # "learning_rate":
            "iterations": 1000,
            "depth": 8,
        }

    model = CatBoostRegressor(
        loss_function="RMSEWithUncertainty",
        bootstrap_type="Bernoulli",
        custom_metric="RMSE",
        verbose=False,
        random_state=random_state,
        thread_count=1,
        **params,
    )
    # TODO Is init_model helpful?
    model.fit(X_train, y_train)  # , init_model=prev_model

    return model, params


def tune_hyperparams_catboost(X, y, budget=TUNE_BUDGET, random_state=None):
    space = {
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.3),
        "depth": hp.choice("depth", [4, 6, 8, 10]),
        "l2_leaf_reg": hp.choice("l2_leaf_reg", [3, 5, 10]),
        "iterations": hp.choice("iterations", [500, 1000, 1500, 2000, 2500]),
        # "subsample": hp.choice("subsample", [0.5, 0.75, 1.0])  # CB sets this itself
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(params, random_state=random_state, cv=kf, X=X, y=y):
        losses = []

        for train_index, test_index in cv.split(X):
            model = CatBoostRegressor(
                loss_function="RMSEWithUncertainty",
                bootstrap_type="Bernoulli",
                custom_metric="RMSE",
                verbose=False,
                random_state=random_state,
                thread_count=1,
                **params,
            )
            model.fit(
                X[train_index],
                y[train_index],
                eval_set=(X[test_index], y[test_index]),
                early_stopping_rounds=10,
                use_best_model=True,
            )
            y_pred = model.predict(X[test_index])
            score = mean_squared_error(y[test_index], y_pred[:, 0], squared=False)
            losses.append(score)

        results = {"loss": np.mean(losses), "status": STATUS_OK}
        return results

    TRIALS = Trials()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(
            fn=objective, space=space, algo=tpe.suggest, max_evals=budget, trials=TRIALS
        )

    best_params = space_eval(space, best)
    return best_params


def get_uncertainty_function(
    model_name
):
    if model_name == "gbdt-cb":
        return partial(
            CB_uncertainty,
            uncertainty="uncertainty",
        )

    if model_name == "rf":
        return RF_uncertainty

    if model_name == "gp":
        return GP_regression_std

    raise Exception(f"Model {model_name} not supported.")


def GP_uncertainty(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    return std


def CB_uncertainty(
    regressor,
    X_unlabeled,
    uncertainty="uncertainty",
    ensemble_size=10,
):
    preds = regressor.virtual_ensembles_predict(
        X_unlabeled,
        prediction_type="TotalUncertainty",
        virtual_ensembles_count=ensemble_size,
    )

    if uncertainty == "uncertainty":
        # Total uncertainty
        uncertainties = preds[:, 1] + preds[:, 2]
    elif uncertainty == "knowledgeunc":
        uncertainties = preds[:, 1]
    elif uncertainty == "dataunc":
        uncertainties = preds[:, 2]

    return uncertainties


def RF_uncertainty(
    regressor,
    X_unlabeled,
):
    _, uncertainties = regressor.predict(X_unlabeled, return_std=True)
    uncertainties = uncertainties.reshape(
        X_unlabeled.shape[0],
    )

    return uncertainties


def GP_regression_std(
    regressor,
    X_unlabeled,
):
    _, uncertainties = regressor.predict(X_unlabeled, return_std=True)

    return uncertainties



