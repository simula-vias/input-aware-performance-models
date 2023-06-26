from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn import preprocessing
from active_models import get_model, get_uncertainty_function
import click
import os
import common
import json
import numpy as np
import time
from joblib import dump

methods = ["rf", "gbdt-cb", "gp"]
query_strategies = ["combined", "config", "input", "uncertainty", "random"]


@click.command()
@click.argument("run_name")
@click.argument("system")
@click.argument("method", type=click.Choice(methods, case_sensitive=False))
@click.option(
    "-o",
    "--output_directory",
    default="models",
    help="Directory to store models and logs",
)
# @click.option("-ns", "--num-samples", default=1, type=int)
@click.option(
    "-s",
    "--query-strategy",
    default="uncertainty",
    type=click.Choice(query_strategies, case_sensitive=False),
)
@click.option("-pp", "--performance-property")
@click.option("--seed", default=None, type=int)
@click.option("--tune/--no-tune", default=False)
@click.option("--data-dir", default="data/")
def run_active(
    run_name,
    system,
    method,
    output_directory,
    query_strategy,
    performance_property,
    seed,
    tune,
    data_dir,
):
    print(f"Start run {run_name}...")
    print(f"System: {system}")
    print(f"Learning method: {method}")
    print(f"Query strategy: {query_strategy}")
    print(f"Output directory: {output_directory}")
    print(f"Random seed: {seed}")

    identifier = get_identifier(
        run_name,
        system,
        method,
        query_strategy,
        performance_property,
        seed,
        tune,
    )

    train_dir = os.path.join(output_directory, identifier)
    os.makedirs(train_dir, exist_ok=True)
    final_model_path = os.path.join(train_dir, "final_model.joblib")
    final_result_path = os.path.join(train_dir, "eval_metrics.json")

    if os.path.exists(final_result_path):
        print(
            f"{identifier} - Run directory already contains evaluation results... stop training"
        )
        return

    cfg_dict = {
        "identifier": identifier,
        "run_name": run_name,
        "system": system,
        "method": method,
        "performance_property": performance_property,
        "output_directory": output_directory,
        "strategy": query_strategy,
        "seed": seed,
        "data_dir": data_dir,
        "tune": tune,
    }

    full_data = common.load_data(data_dir, system=system)

    full_train_data, test_data, train_inputs, train_configs = common.split_data(
        data=full_data["data"],
        system=system,
        inputs_count=full_data["input_counts"],
        config_feat_cols=full_data["feature_columns"],
        random_seed=seed,
    )
    assert performance_property in full_data["performance_properties"][system]
    input_data = full_data["input_properties_data"][system]
    input_feature_columns = full_data["input_properties"][system]
    config_feature_columns = full_data["feature_columns"][system]

    # TODO Should we consider multi-output models? E.g. scikit-rf allows it

    # TODO Select initial subset with kmeans

    train_data_matrix, y_train_input_ids = common.merge_matrix(
        full_train_data,
        None,
        config_feature_columns,
        input_data,
        input_feature_columns,
    )
    X_train = train_data_matrix[config_feature_columns + input_feature_columns]
    y_train = train_data_matrix[performance_property].values
    train_config_ids = (
        X_train[config_feature_columns]
        .agg(lambda x: "-".join([str(s) for s in x]), axis=1)
        .factorize()[0]
    )
    train_input_ids = np.array(y_train_input_ids)

    test_data_matrix, y_test_input_ids = common.merge_matrix(
        test_data,
        None,  # no filter by configuration
        config_feature_columns,
        input_data,
        input_feature_columns,
    )
    X_test = test_data_matrix[config_feature_columns + input_feature_columns]
    y_test = test_data_matrix[performance_property].values

    assert query_strategy in query_strategies

    start_time = time.time()

    if method == "gp":
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    uncertainty_func = get_uncertainty_function(method)

    start_time_tune = time.time()
    # TODO Hyperparameter search
    tune_duration = time.time() - start_time_tune

    train_duration = 0  # Initialize to subtract test time in loop

    n_queries = 10
    initial_idx = np.random.choice(
        np.arange(len(X_train)), size=n_queries, replace=False
    )
    pool_mask = np.ones(X_train.shape[0], dtype=bool)
    pool_mask[initial_idx] = False

    rgr, _ = get_model(
        model_name=method,
        random_state=seed,
        X_train=X_train[~pool_mask],
        y_train=y_train[~pool_mask],
        tune_hyperparams=tune,
    )

    if method == "gbdt-cb":
        y_test_pred = np.array(rgr.predict(X_test))[:, 0].reshape(-1, 1)
    else:
        y_test_pred = np.array(rgr.predict(X_test)).reshape(-1, 1)

    y_test_categories = common.get_test_sample_categories(
        train_data_matrix[~pool_mask],
        test_data_matrix,
        config_feature_columns,
        train_input_ids[~pool_mask],
        y_test_input_ids,
    )

    init_metrics = common.evaluate(
        y_test,
        y_test_pred,
        y_test_categories,
    )
    eval_metrics = {k: [v] for k, v in init_metrics.items()}
    eval_metrics["train_size"] = [(~pool_mask).sum()]

    n_queries = 200 #pool_mask.sum()  # TODO
    for idx in range(n_queries):
        if query_strategy == "combined":
            query_idx = select_general_input_general_config(
                regressor=rgr,
                X=X_train[pool_mask],
                train_config_ids=train_config_ids[pool_mask],
                train_input_ids=train_input_ids[pool_mask],
                uncertainty_func=uncertainty_func,
            )
        elif query_strategy == "config":
            query_idx = select_general_group_max_entry(
                regressor=rgr,
                X=X_train[pool_mask],
                grouping_ids=train_config_ids[pool_mask],
                uncertainty_func=uncertainty_func,
            )
        elif query_strategy == "input":
            query_idx = select_general_group_max_entry(
                regressor=rgr,
                X=X_train[pool_mask],
                grouping_ids=train_input_ids[pool_mask],
                uncertainty_func=uncertainty_func,
            )
        elif query_strategy == "uncertainty":
            query_idx = select_max_uncertainty(
                regressor=rgr,
                X=X_train[pool_mask],
                uncertainty_func=uncertainty_func,
            )
        elif query_strategy == "random":
            query_idx = select_random_query(X=X_train[pool_mask])

        pool_mask[np.arange(len(pool_mask))[pool_mask][query_idx]] = False

        rgr, _ = get_model(
            model_name=method,
            random_state=seed,
            X_train=X_train[~pool_mask],
            y_train=y_train[~pool_mask],
        )

        start_test_time = time.time()

        if method == "gbdt-cb":
            y_test_pred = np.array(rgr.predict(X_test))[:, 0].reshape(-1, 1)
        else:
            y_test_pred = np.array(rgr.predict(X_test)).reshape(-1, 1)

        y_test_categories = common.get_test_sample_categories(
            train_data_matrix[~pool_mask],
            test_data_matrix,
            config_feature_columns,
            train_input_ids[~pool_mask],
            y_test_input_ids,
        )

        iter_metrics = common.evaluate(
            y_test,
            y_test_pred,
            y_test_categories,
        )
        eval_metrics["train_size"].append((~pool_mask).sum())
        for k, v in iter_metrics.items():
            eval_metrics[k].append(v)

        train_duration -= time.time() - start_test_time

    train_duration += time.time() - start_time

    start_time = time.time()
    if method == "gbdt-cb":
        y_test_pred = np.array(rgr.predict(X_test))[:, 0].reshape(-1, 1)
    else:
        y_test_pred = np.array(rgr.predict(X_test)).reshape(-1, 1)

    prediction_duration = time.time() - start_time

    # Can only be calculated after subset selection, because it might exclude some inputs/configs from training
    y_test_categories = common.get_test_sample_categories(
        train_data_matrix[~pool_mask],
        test_data_matrix,
        config_feature_columns,
        train_input_ids[~pool_mask],
        y_test_input_ids,
    )

    final_metrics = common.evaluate(
        y_test,
        y_test_pred,
        y_test_categories,
    )
    eval_metrics["train_size"].append((~pool_mask).sum())
    for k, v in final_metrics.items():
        eval_metrics[k].append(v)
    eval_metrics.update(cfg_dict)
    eval_metrics["input_selection_duration"] = 0
    eval_metrics["config_selection_duration"] = 0
    eval_metrics["train_duration"] = train_duration
    eval_metrics["tune_duration"] = tune_duration
    eval_metrics["prediction_duration"] = prediction_duration

    json.dump(eval_metrics, open(final_result_path, "w"), cls=common.NpEncoder)
    dump(rgr, final_model_path, compress=True)

    print(f"{identifier} - Done")


def get_identifier(
    run_name,
    system,
    method,
    query_strategy,
    performance_property,
    seed,
    tune,
):
    identifier = (
        f"al-{run_name}-{system}-{method}-{query_strategy}-{performance_property}"
    )

    if tune:
        identifier += "-tune_yes"
    else:
        identifier += "-tune_no"

    identifier += f"-seed{seed}"

    return identifier


def select_max_uncertainty(regressor, X, uncertainty_func):
    std = uncertainty_func(regressor, X)
    query_idx = np.argmax(std)
    return query_idx


def select_random_query(X):
    n_instances = 1
    query_idx = np.random.choice(
        range(X.shape[0]),
        size=min(X.shape[0], n_instances),
        replace=False,
    )
    return query_idx


def select_general_group_max_entry(regressor, X, uncertainty_func, grouping_ids):
    """Selects the group with max. mean uncertainty and in it the entry with the highest uncertainty."""
    std = uncertainty_func(regressor, X)
    indices = np.arange(len(std))
    uniq_ids = np.unique(grouping_ids)
    mean_stds = np.empty(len(uniq_ids))
    max_ingroup = np.empty(len(uniq_ids), dtype=int)

    for idx, grp_id in enumerate(uniq_ids):
        cond = grouping_ids == grp_id
        mean_stds[idx] = std[cond].mean()
        max_ingroup[idx] = indices[cond][np.argmax(std[cond])]

    query_idx = max_ingroup[np.argmax(mean_stds)]
    return query_idx


def select_general_input_general_config(
    regressor, X, uncertainty_func, train_config_ids, train_input_ids
):
    """Selects the both input and configs with max. mean uncertainty and combines them.

    If the combination is not in the dataset, an exception is thrown."""
    std = uncertainty_func(regressor, X)

    max_mean_std = -1

    for grp_id in np.unique(train_config_ids):
        mean_std = std[train_config_ids == grp_id].mean()
        if mean_std > max_mean_std:
            max_mean_std = mean_std
            max_cfg_id = grp_id

    max_mean_std = -1

    for grp_id in np.unique(train_input_ids):
        mean_std = std[train_input_ids == grp_id].mean()
        if mean_std > max_mean_std:
            max_mean_std = mean_std
            max_inp_id = grp_id

    matches = (
        (train_config_ids == max_cfg_id) & (train_input_ids == max_inp_id)
    ).nonzero()

    if len(matches) != 1:
        raise Exception("Could not find entry for query selection")

    query_idx = matches[0]

    # print(query_idx, std[query_idx].round(5), np.argmax(std), np.max(std).round(5))
    return query_idx


if __name__ == "__main__":
    run_active()
