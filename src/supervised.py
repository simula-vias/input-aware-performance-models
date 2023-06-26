from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    # HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from autosklearn.regression import AutoSklearnRegressor
import click
import os
import common
import json
import numpy as np
import time
from joblib import dump

methods = ["rf", "gbdt", "gbdt-cb", "lr", "automl"]


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
@click.option("-ni", "--num-inputs", default=None, type=int)
@click.option(
    "-is",
    "--input-selection",
    default="kmeans",
    type=click.Choice(["kmeans", "random", "submodular"], case_sensitive=False),
)
@click.option("-nc", "--num-configs", default=None, type=int)
@click.option(
    "-cs",
    "--config-selection",
    default="kmeans",
    type=click.Choice(["kmeans", "random", "submodular"], case_sensitive=False),
)
@click.option("-pp", "--performance-property")
@click.option("--seed", default=None, type=int)
@click.option("--tune/--no-tune", default=False)
@click.option("--data-dir", default="data/")
def run_supervised(
    run_name,
    system,
    method,
    output_directory,
    num_inputs,
    input_selection,
    num_configs,
    config_selection,
    performance_property,
    seed,
    tune,
    data_dir,
):
    print(f"Start run {run_name}...")
    print(f"System: {system}")
    print(f"Learning method: {method}")
    print(f"Output directory: {output_directory}")
    print(f"Max. number of training inputs: {num_inputs}")
    print(f"Max. number of configurations: {num_configs}")
    print(f"Random seed: {seed}")

    identifier = get_identifier(
        run_name,
        system,
        method,
        num_inputs,
        input_selection,
        num_configs,
        config_selection,
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
        "num_inputs": num_inputs,
        "input_selection": input_selection if num_inputs is not None else None,
        "num_configs": num_configs,
        "config_selection": config_selection if num_configs is not None else None,
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

    start_time = time.time()

    if num_inputs is None:
        selected_train_data = full_train_data
        input_subset = []
    else:
        input_subset = common.data_selection(
            num_samples=num_inputs,
            mode=input_selection,
            data=input_data.loc[train_inputs],
            selection_columns=input_feature_columns,
            random_seed=seed,
        )
        input_indices = input_subset.index.tolist()
        selected_train_data = {
            (s, i): v for (s, i), v in full_train_data.items() if i in input_indices
        }
        assert len(selected_train_data) == num_inputs

    input_selection_duration = time.time() - start_time

    start_time = time.time()

    if num_configs is None:
        config_subset = None
    else:
        config_subset = common.data_selection(
            num_samples=num_configs,
            mode=config_selection,
            data=train_configs,
            selection_columns=config_feature_columns,
            random_seed=seed,
        )
        assert len(config_subset) == num_configs

    config_selection_duration = time.time() - start_time

    # TODO Make selection method that considers inputs + configs in union

    if config_subset is not None and len(config_subset) == 1 and len(input_subset) == 1:
        print(f"{identifier} - Only one resulting training sample... stop training")
        return

    # TODO Should we consider multi-output models? E.g. scikit-rf allows it

    train_data_matrix, y_train_input_ids = common.merge_matrix(
        selected_train_data,
        config_subset,
        config_feature_columns,
        input_data,
        input_feature_columns,
    )
    X_train = train_data_matrix[config_feature_columns + input_feature_columns]
    y_train = train_data_matrix[performance_property].values

    test_data_matrix, y_test_input_ids = common.merge_matrix(
        test_data,
        None,  # no filter by configuration
        config_feature_columns,
        input_data,
        input_feature_columns,
    )
    X_test = test_data_matrix[config_feature_columns + input_feature_columns]
    y_test = test_data_matrix[performance_property].values

    start_time_train = time.time()


    if method == "rf":
        rgr = RandomForestRegressor(random_state=seed)
    elif method == "gbdt":
        rgr = GradientBoostingRegressor(random_state=seed)
    # elif method == "gbdt-hist":
    #     rgr = HistGradientBoostingRegressor(random_state=seed)
    elif method == "gbdt-cb":
        rgr = CatBoostRegressor(random_state=seed, verbose=False, use_best_model=True)
        # grid = {'learning_rate': [0.001, 0.01, 0.1],
        # 'depth': [4, 6, 8, 10]}
    elif method == "lr":
        rgr = LinearRegression()
    elif method == "automl":
        rgr = AutoSklearnRegressor(
            time_left_for_this_task=7200,
            per_run_time_limit=300,
            tmp_folder=os.path.join(train_dir, "automl"),
            delete_tmp_folder_after_terminate=False,
            memory_limit=6000,
        )
    # elif method == "autokeras":
    #     rgr = ak.
    else:
        raise Exception(f"Method {method} not supported")

    # TODO Hyperparameter search
    
    start_time_tune = time.time()

    if tune:
        # train_df, val_df = train_test_split(train_df, test_size=0.2)
        # # TODO (Optional) tune hyperparameters?
        # # lr = [0.03, 0.07, 0.15, 0.3]
        # # max_bin = [64, 128, 256]
        # # max_depth = [4, 6, 8, 10]
        pass

    tune_duration = time.time() - start_time_tune

    rgr.fit(X_train, y_train)

    train_duration = time.time() - start_time_train

    start_time = time.time()
    y_test_pred = np.array(rgr.predict(X_test)).reshape(-1, 1)

    prediction_duration = time.time() - start_time

    # Can only be calculated after subset selection, because it might exclude some inputs/configs from training
    y_test_categories = common.get_test_sample_categories(
        train_data_matrix,
        test_data_matrix,
        config_feature_columns,
        y_train_input_ids,
        y_test_input_ids,
    )

    eval_metrics = common.evaluate(
        y_test,
        y_test_pred,
        y_test_categories,
    )
    eval_metrics.update(cfg_dict)
    eval_metrics["input_selection_duration"] = input_selection_duration
    eval_metrics["config_selection_duration"] = config_selection_duration
    eval_metrics["train_duration"] = train_duration
    eval_metrics["tune_duration"] = tune_duration
    eval_metrics["prediction_duration"] = prediction_duration

    json.dump(eval_metrics, open(final_result_path, "w"))
    dump(rgr, final_model_path, compress=True)

    print(f"{identifier} - Done")


def get_identifier(
    run_name,
    system,
    method,
    num_inputs,
    input_selection,
    num_configs,
    config_selection,
    performance_property,
    seed,
    tune,
):
    identifier = f"sl-{run_name}-{system}-{method}-{performance_property}"

    if num_inputs is None:
        identifier += "-inp_all"
    else:
        identifier += f"-inp_{input_selection}_{num_inputs}"

    if num_configs is None:
        identifier += "-cfg_all"
    else:
        identifier += f"-cfg_{config_selection}_{num_configs}"

    if tune:
        identifier += "-tune_yes"
    else:
        identifier += "-tune_no"

    identifier += f"-seed{seed}"

    return identifier


if __name__ == "__main__":
    run_supervised()
