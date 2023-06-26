from typing import List
import numpy as np
import pandas as pd
import json
import os.path as osp

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
)
from apricot import FacilityLocationSelection


def data_selection(
    num_samples: int,
    mode: str,
    data: object,
    selection_columns: List[str],
    random_seed: int,
) -> List[str]:
    # df.fillna(config.config_defaults, axis="index", inplace=True)

    if mode == "kmeans":
        cst = KMeans(n_clusters=num_samples, random_state=random_seed)
        X = data[selection_columns]
        cst.fit(X)
        result = data.iloc[cst.transform(X).argmin(axis=0)]
    elif mode == "submodular":
        fls = FacilityLocationSelection(
            num_samples,
            metric="euclidean",
            optimizer="lazy",
            random_state=random_seed,
        ).fit(data[selection_columns].values)
        result = data.iloc[fls.ranking]
    elif mode == "random":
        result = data.sample(n=num_samples, random_state=random_seed)
    else:
        raise NotImplementedError(f"Mode {mode} not supported for config selection")

    return result

def load_data(data_dir, system=None):
    # the name of the systems we are testing
    name_systems = [
        "nodejs",
        "poppler",
        "xz",
        "x264",
        "gcc",
        "lingeling",
        "sqlite",
        "imagemagick",
    ]
    assert system is None or system in name_systems

    # final results
    data = dict()

    inputs_perf = dict()
    inputs_feat = dict()
    inputs_categ = dict()

    # name of the performance properties
    inputs_perf["gcc"] = ["size", "ctime", "exec"]
    inputs_perf["imagemagick"] = ["size", "time"]
    inputs_perf["lingeling"] = ["conflicts", "cps", "reductions"]
    inputs_perf["nodejs"] = ["ops"]
    inputs_perf["poppler"] = ["size", "time"]
    inputs_perf["sqlite"] = ["q" + str(i + 1) for i in range(15)]
    inputs_perf["x264"] = ["size", "kbs", "fps", "etime", "cpu"]
    inputs_perf["xz"] = ["size", "time"]

    # name of features for each system
    inputs_feat["gcc"] = [
        "optim",
        "-floop-interchange",
        "-fprefetch-loop-arrays",
        "-ffloat-store",
        "-fno-asm",
    ]
    inputs_feat["imagemagick"] = [
        "memory_r",
        "posterize_r",
        "gaussian-blur",
        "thread",
        "quality",
    ]
    inputs_feat["lingeling"] = [
        "--boost",
        "--carduse",
        "--decompose",
        "--gluescale",
        "--lkhd",
        "--memlim",
        "--minimize",
        "--prbsimple",
        "--sweepirr",
        "--sweepred",
    ]
    inputs_feat["nodejs"] = [
        "--jitless",
        "--experimental-wasm-modules",
        "--experimental-vm-modules",
        "--preserve-symlinks-main",
        "--no-warnings",
        "--node-memory-debug",
    ]
    inputs_feat["poppler"] = ["format", "j", "jp2", "jbig2", "ccitt"]
    inputs_feat["sqlite"] = [
        "-deserialize",
        "-memtrace",
        "-maxsize",
        "-append",
        "-output",
    ]
    inputs_feat["x264"] = [
        "cabac",
        "ref",
        "deblock",
        "analyse",
        "me",
        "subme",
        "mixed_ref",
        "me_range",
        "trellis",
        "8x8dct",
        "fast_pskip",
        "chroma_qp_offset",
        "bframes",
        "b_pyramid",
        "b_adapt",
        "direct",
        "weightb",
        "open_gop",
        "weightp",
        "scenecut",
        "rc_lookahead",
        "mbtree",
        "qpmax",
        "aq-mode",
    ]
    inputs_feat["xz"] = ["memory", "format", "level", "depth"]

    # just to isolate the options that have categorial values
    # because it is more difficult to handle for ML algorithms
    inputs_categ["gcc"] = ["optim"]
    inputs_categ["imagemagick"] = []
    inputs_categ["lingeling"] = []
    inputs_categ["nodejs"] = []
    inputs_categ["poppler"] = ["format"]
    inputs_categ["sqlite"] = []
    inputs_categ["x264"] = [
        "analyse",
        "me",
        "direct",
        "deblock",
        "b_adapt",
        "b_pyramid",
        "open_gop",
        "rc_lookahead",
        "scenecut",
        "weightb",
    ]
    inputs_categ["xz"] = ["memory", "format"]

    # categorical features for input properties
    inputs_prop_categ = dict()
    inputs_prop_categ["gcc"] = []
    inputs_prop_categ["imagemagick"] = ["description"]
    inputs_prop_categ["lingeling"] = []
    inputs_prop_categ["nodejs"] = []
    inputs_prop_categ["poppler"] = []
    inputs_prop_categ["sqlite"] = []
    inputs_prop_categ["x264"] = ["category"]
    inputs_prop_categ["xz"] = ["type"]

    inputs_prop = dict()
    inputs_prop_names = dict()
    inputs_num = dict()
    inputs_feat_cols = dict()
    inputs_name = dict()
    inputs_count = dict()

    for ns in name_systems:
        if system is not None and ns != system:
            continue

        data_path = osp.join(data_dir, ns)

        df = pd.read_csv(
            osp.join(data_path, "others/properties.csv")
        ).set_index("id")

        non_categ_columns = [s for s in df.columns if s not in inputs_prop_categ[ns]]

        if inputs_prop_categ[ns] != []:
            inputs_prop[ns] = df[non_categ_columns].join(pd.get_dummies(df[inputs_prop_categ[ns]]))
            # inputs_feat_cols[ns] = list(tmpdf.columns)
        else:
            # inputs_feat_cols[ns] = list(numerical_columns)
            inputs_prop[ns] = df[non_categ_columns]

        # Rename columns with same name in inputs/perf. prediction to avoid errors later
        # affects imagemagick
        for c in inputs_prop[ns].columns:
            if c in inputs_perf[ns]:
                inputs_prop[ns].rename(columns={c: f"inp_{c}"}, inplace=True)

        inputs_prop_names[ns] = [s for s in inputs_prop[ns].columns if s != "name"]

        inputs = [str(name) + ".csv" for name in inputs_prop[ns]["name"]]

        inputs_name[ns] = inputs
        inputs_count[ns] = len(inputs)

        inputs_num[ns] = np.setdiff1d(inputs_feat[ns], inputs_categ[ns])
        inputs_feat_cols[ns] = []

        for i in range(len(inputs)):
            loc = osp.join(data_path, inputs[i])
            df = pd.read_csv(loc)
            if inputs_categ[ns] != []:
                tmpdf = df[inputs_num[ns]].join(pd.get_dummies(df[inputs_categ[ns]]))
                inputs_feat_cols[ns] = list(tmpdf.columns)
                data[ns, i] = tmpdf.join(df[inputs_perf[ns]])
            else:
                inputs_feat_cols[ns] = list(inputs_num[ns])
                data[ns, i] = df[inputs_num[ns]].join(df[inputs_perf[ns]])

    return {
        "data": data,
        "performance_properties": inputs_perf,
        "features": inputs_feat,
        "features_numerical": inputs_num,
        "features_categorical": inputs_categ,
        "feature_columns": inputs_feat_cols,
        "input_properties": inputs_prop_names,
        "input_properties_data": inputs_prop,
        "input_properties_categorical": inputs_prop_categ,
        "input_names": inputs_name,
        "input_counts": inputs_count,
    }


def split_data(
    data, system, inputs_count, config_feat_cols, random_seed, test_size=0.2
):
    inputs = range(inputs_count[system])
    configs = pd.concat(
        (data[system, i][config_feat_cols[system]] for i in range(inputs_count[system]))
    ).drop_duplicates()

    split_test_size = test_size / 2
    train_inp, test_inp = train_test_split(
        inputs, test_size=split_test_size, random_state=random_seed
    )
    train_cfg, test_cfg = train_test_split(
        configs, test_size=split_test_size, random_state=random_seed
    )

    train_data = dict()
    test_data = dict()

    # test_cfg -> exclusive to test set, with all inputs
    for i in train_inp:
        train_data[system, i] = pd.merge(
            data[system, i], train_cfg, on=config_feat_cols[system], how="inner"
        )
        test_data[system, i] = pd.merge(
            data[system, i], test_cfg, on=config_feat_cols[system], how="inner"
        )
        assert (
            len(
                pd.merge(
                    train_data[system, i],
                    test_data[system, i],
                    on=config_feat_cols[system],
                    how="inner",
                )
            )
            == 0
        )

    # test_inp -> exclusive to test set, with all configs
    for i in test_inp:
        test_data[system, i] = data[system, i]

    return train_data, test_data, np.array(train_inp), train_cfg


def merge_matrix(
    data, configs, config_feature_columns, input_data, input_feature_columns
):
    data_list = []
    query_ids = []
    for (_, i), values in data.items():
        if configs is not None:
            cfg_data = pd.merge(values, configs, on=config_feature_columns, how="inner")
        else:
            cfg_data = values
        inp_data = input_data.loc[
            np.full(len(cfg_data), i), input_feature_columns
        ].reset_index()
        data_list.append(pd.concat([cfg_data, inp_data], axis=1))
        query_ids.extend([i] * len(cfg_data))
    return pd.concat(data_list), query_ids


def get_test_sample_categories(
    train_data_matrix,
    test_data_matrix,
    config_feature_columns,
    y_train_input_ids,
    y_test_input_ids,
):
    y_test_categories = []
    merged = pd.merge(
        test_data_matrix[config_feature_columns],
        train_data_matrix[config_feature_columns],
        how="left",
        on=config_feature_columns,
        indicator="cfg_exists",
    )
    for merge_status, inp_id in zip(merged["cfg_exists"], y_test_input_ids):
        is_new_config = merge_status == "left_only"
        is_new_input = inp_id not in y_train_input_ids
        if is_new_input and is_new_config:
            y_test_categories.append("new_both")
        elif is_new_input:
            y_test_categories.append("new_input")
        else:
            y_test_categories.append("new_config")

    return y_test_categories


def evaluate(
    y_test,
    y_test_pred,
    y_test_category,
):
    # Performance Prediction Metrics
    ## Split data by categories for MAPE calculation
    data_new_inputs = []
    data_new_configs = []
    data_new_both = []

    for (truth, pred, cat) in zip(y_test, y_test_pred, y_test_category):
        if cat == "new_input":
            data_new_inputs.append((truth, pred))
        elif cat == "new_config":
            data_new_configs.append((truth, pred))
        elif cat == "new_both":
            data_new_both.append((truth, pred))

    # Configuration Selection Metrics (not implemented)
    # Extra parameters required: y_test_input_id for document id, goal, top_n
    # y_test_ranking = np.argsort(y_test_pred)

    return {
        # Performance Prediction Metrics
        "mse": mean_squared_error(y_test, y_test_pred),
        "mape": mean_absolute_percentage_error(y_test, y_test_pred),
        "mape_new_inputs": mean_absolute_percentage_error(*zip(*data_new_inputs)),
        "mape_new_configs": mean_absolute_percentage_error(*zip(*data_new_configs)),
        "mape_new_both": mean_absolute_percentage_error(*zip(*data_new_both)),
        # Configuration Selection Metrics
        # "Precision@1": 0,
        # f"Precision@{top_n}": 0,
    }


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    import joblib

    (y_test, y_test_pred, y_test_input_ids, y_test_categories) = joblib.load("yt.p")
    # print(y_test)
    # print(y_test_pred)
    res = evaluate(
        y_test,
        y_test_pred,
        y_test_categories,
    )
    print(res)
