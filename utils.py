from pathlib import Path

import numpy as np
import pandas as pd
import torch
from python_tools import caching

from train import CLASS_DIMENSION, CCC_DIMENSION


def find_correspondence(a, b):
    ids_a = a["meta_id"]
    begin_a = a["meta_begin"]
    end_a = a["meta_end"]
    if isinstance(ids_a, pd.Series):
        ids_a = ids_a.values
        begin_a = begin_a.values
        end_a = end_a.values
    ids_b = b["meta_id"]
    begin_b = b["meta_begin"]
    end_b = b["meta_end"]
    if isinstance(ids_b, pd.Series):
        ids_b = ids_b.values
        begin_b = begin_b.values
        end_b = end_b.values
    ids_match = ids_a.flatten()[:, None] == ids_b.flatten()[None, :]
    begin_match = np.abs(begin_a.flatten()[:, None] - begin_b.flatten()[None, :]) < 0.2
    end_match = np.abs(end_a.flatten()[:, None] - end_b.flatten()[None, :]) < 0.2
    return ids_match & begin_match & end_match


def select(dictionary, keys):
    return {key: dictionary[key] for key in keys}


def load_transformation_compat(file):
    trans = caching.read_pickle(file)
    if isinstance(trans, tuple):
        trans = trans[0]
    if isinstance(trans, tuple):
        trans = trans[0]
    return trans


def extract_routing(prediction, datum):
    if "meta_p_i" not in datum:
        return datum
    if isinstance(prediction, dict):
        # is transform
        prediction = datum["y_scores"] * prediction["std"] + prediction["mean"]

    modalities = 3 if "meta_m6_l0" in datum else 2
    labels = sum(f"meta_m0_l{i}" in datum for i in range(10))

    is_pytorch = isinstance(datum["meta_m0_l0"], torch.Tensor)
    if is_pytorch:
        # convert to numpy and at the end convert back to pytorch
        for (key,) in datum:
            datum[key] = datum[key].detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    y_uni = np.concatenate(
        [
            sum(datum[f"meta_m{i}_l{label}"] for i in range(modalities))
            for label in range(labels)
        ],
        axis=1,
    )
    y_tri = np.concatenate(
        [
            datum[f"meta_m{6 if modalities == 3 else 2}_l{label}"]
            for label in range(labels)
        ],
        axis=1,
    )
    if modalities == 2:
        y_bi = np.zeros_like(y_uni)
    else:
        y_bi = np.concatenate(
            [
                sum(datum[f"meta_m{i}_l{label}"] for i in range(3, 6))
                for label in range(labels)
            ],
            axis=1,
        )
    bias = np.median(y_uni + y_bi + y_tri - prediction, axis=0)
    datum["meta_uni_y"] = y_uni - bias[None, :]
    datum["meta_bi_y"] = y_bi
    datum["meta_tri_y"] = y_tri
    if is_pytorch:
        for key in datum:
            datum[key] = torch.from_numpy(datum[key])

    return datum


def get_components(file, fold, datum):
    # load transformation
    trans = load_transformation_compat(file.parent / f"partition_{fold}.pickle")
    trans = trans.get(
        "y", {"std": 1, "mean": 0, "min": float("-inf"), "max": float("inf")}
    )
    toy_data = any(x in file.parent.name for x in ("=uni", "=bi", "=tri", "=all"))
    if toy_data:
        trans["std"][:] = 1
        trans["mean"][:] = 0

    # routing
    datum = extract_routing(trans, datum)

    for i, key in enumerate(("uni", "bi", "tri")):
        datum[f"{key}_y_delta"] = (
            datum.get(
                f"meta_{key}_y",
                datum.get(f"meta_y_hat_{i}", np.zeros_like(datum["y_hat"])),
            )
            * trans["std"]
        )

    datum["uni_y_delta"] = datum["uni_y_delta"] + trans["mean"]
    y_uni = datum["uni_y_delta"]
    y_bi = y_uni + datum["bi_y_delta"]
    y_tri = y_bi + datum["tri_y_delta"]

    # clip
    y_uni = np.clip(y_uni, trans["min"], trans["max"])
    y_bi = np.clip(y_bi, trans["min"], trans["max"])
    y_tri = np.clip(y_tri, trans["min"], trans["max"])
    datum["y_hat"] = np.clip(datum["y_hat"], trans["min"], trans["max"])

    if y_tri.shape[1] != 1:
        index = np.argmax(y_tri, axis=1, keepdims=True) == datum["y_hat"]
    else:
        index = np.abs(datum["y_hat"] - y_tri) < 1e-4
    assert index.mean() > 0.99
    return y_uni, y_bi, y_tri, datum


def fs_not_fs(folder: Path) -> Path:
    assert "fs=" not in folder.name
    assert "mult=" not in folder.name
    assert "rerun=" not in folder.name

    # determine metric
    dimension = folder.name.split("dimension=", 1)[-1].split("_", 1)[0]
    toy_data = dimension in ("uni", "bi", "tri", "all")
    name = find_metrics(dimension)
    max_ = name not in ("brier_score", "mae")

    # find fs=True
    experiments = [folder.name]
    fs_folder = folder.name.replace(
        f"dimension={dimension}", f"dimension={dimension}_fs=True"
    )
    if "wordlevel" not in folder.name:
        experiments.append(fs_folder)
    if (
        "wordlevel" not in folder.name
        and "routing" not in folder.name
        and "=uni" not in folder.name
    ):
        # find mult=True and mult+Tur&fs=True
        if (
            all(x in folder.name for x in ("joint=", "res_det="))
            and "init=" not in folder.name
        ):
            experiments.append(
                folder.name.replace("res_det=True", "mult=True").replace(
                    "_joint=True", ""
                )
            )
            experiments.append(
                experiments[1]
                .replace("res_det=True", "mult=True")
                .replace("_joint=True", "")
            )
        elif "nodetach=True" in folder.name:
            experiments.append(
                folder.name.replace("nodetach=True", "mult=True_nodetach=True")
            )
            experiments.append(
                experiments[1].replace("nodetach=True", "mult=True_nodetach=True")
            )
            "mult=True_nodetach=True_"
        else:
            experiments.append(
                folder.name.replace("res_det=True", "mult=True_res_det=True")
            )
            experiments.append(
                experiments[1].replace("res_det=True", "mult=True_res_det=True")
            )
    if "routing" not in folder.name and "=bi" in folder.name:
        experiments = experiments[2:]

    # find best model
    best_score = float("inf")
    if max_:
        best_score = -float("inf")
    for experiment in experiments:
        file = folder.parent / experiment / "overview.csv"
        if not file.is_file() and toy_data and "fs=" in experiment:
            # have no fs
            continue
        assert file.is_file()
        overview = pd.read_csv(file)[f"validation_{name}"]
        if max_ and best_score < overview.max():
            best_experiment = experiment
            best_score = overview.max()
        elif not max_ and best_score > overview.min():
            best_experiment = experiment
            best_score = overview.min()
    return folder.parent / best_experiment


def get_dimension(folder):
    return folder.name.split("dimension=", 1)[-1].split("_", 1)[0]


def find_metrics(dimension) -> str:
    if dimension in CCC_DIMENSION or dimension[0] == dimension[0].upper():
        metric = "ccc"
    elif dimension == "constructs":
        metric = "accuracy"
    elif dimension in ("mosi", "sentiment", "polarity", "happiness"):
        metric = "pearson_r"
    elif dimension == "intent":
        metric = "roc_auc_macro"
    elif dimension in CLASS_DIMENSION:
        metric = "brier_score"
    elif dimension in ("uni", "bi"):
        metric = "mae"
    else:
        raise AssertionError(dimension)
    return metric


def find_test(
    dimension: str,
    smro: bool = False,
    base: bool = False,
    train: bool = False,
    nodetach: bool = False,
    mincov: bool = False,
    tri_only: bool = False,
    sym: bool = False,
    routing: bool = False,
    wordlevel: bool = False,
):
    assert int(smro) + int(base) < 2
    # build name directly
    test = f"dimension={dimension}"
    if base:
        test += f"_joint={base}"
    if mincov:
        test += f"_mincov={mincov}"
    if nodetach:
        test += f"_nodetach={nodetach}"
    if not routing:
        test += "_res_det=True"
    else:
        test += f"_routing={routing}"
    if smro:
        test += f"_stepwise={smro}"
    if sym:
        test += f"_sym={sym}"
    if tri_only:
        test += f"_tri={tri_only}"
    if wordlevel:
        test += f"_wordlevel={wordlevel}"
    match = Path("experiments") / test
    folder = fs_not_fs(match)
    if train:
        folder = folder.parent / f"{folder.name}_train=True"
    try:
        return next(folder.glob("*_test_predictions.pickle"))
    except StopIteration:
        return None
