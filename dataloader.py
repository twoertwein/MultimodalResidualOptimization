#!/usr/bin/env python3
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd
from python_tools import caching
from python_tools.features import (aggregate_to_intervals, apply_masking,
                                   suggest_statistics)
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.markers import get_markers
from python_tools.ml.metrics import concat_dicts
from python_tools.ml.pytorch_tools import dict_to_batched_data
from python_tools.ml.split import stratified_splits


class SEWA:
    def __init__(
        self,
        *,
        dimension: Literal["valence", "arousal", "liking"] = "valence",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        self.n_folds: int = 5
        self.dimension = dimension
        self.ifold = ifold
        self.name = name
        self.folder = Path()
        self.mapping = {}
        if self.__class__.__name__ == "SEWA":
            self.mapping = {x: x for x in self.get_list_of_subjects()}

    def get_cache_path(self) -> Path:
        name = f"SEWA_{self.dimension}_{self.name}_{self.n_folds}_{self.ifold}"
        return Path("cache") / f"{name}.pickle"

    def get_loader(self) -> DataLoader:
        cache = self.get_cache_path()
        data_properties = caching.read_pickle(cache)

        if data_properties is None:
            subject = self.get_list_of_subjects()[0]
            properties = {
                "X_names": [
                    x
                    for x in self.get_features(subject).columns
                    if not x.startswith("meta_") and not x == self.dimension
                ],
                "Y_names": [self.dimension],
            }
            assert len(set(properties["X_names"])) == len(properties["X_names"])
            for key in properties:
                properties[key] = np.asarray(properties[key])

            self.properties = properties
            data = list(self)

            caching.write_pickle(cache, (data, properties))
        else:
            data, properties = data_properties
            self.properties = properties
        assert all(x["X"][0].shape[1] == properties["X_names"].shape[0] for x in data)

        return DataLoader(data, properties=deepcopy(properties))

    def __iter__(self) -> Iterator[dict[str, list[np.ndarray]]]:
        """
        Returns a list of training, validation, test folds
        """
        x_names = self.properties["X_names"]
        # load labels for all subjects
        subjects = self.get_subjects_for_fold()

        for subject in subjects:
            # load openface and combine with labels
            data = self.get_features(subject)

            # create folds
            data_ = {
                "X": data.loc[:, x_names].values.astype(np.float32),
                "Y": data[self.dimension].values[:, None].astype(np.float32),
            }
            data_.update(
                {
                    key: data[key].values[:, None]
                    for key in data.columns
                    if key.startswith("meta_")
                }
            )
            data = data_
            data = dict_to_batched_data(data, batch_size=-1)

            while data:
                yield data.pop()

    def get_transcripts(self, subject: str) -> pd.DataFrame:
        return caching.read_hdfs(Path("transcriptions/") / subject)["df"]

    def get_labels(self, subject: str) -> pd.DataFrame:
        transcripts = self.get_transcripts(subject)
        labels = caching.read_hdfs(Path("labels") / subject)["df"]
        labels = labels.add_prefix("meta_").rename(
            columns={f"meta_{self.dimension}": self.dimension}
        )
        data = []
        for begin_end, row in transcripts.iterrows():
            data.append(labels.loc[begin_end.left : begin_end.right].mean())
        return pd.DataFrame(data, index=transcripts.index)

    def get_features(self, subject: str) -> pd.DataFrame:
        labels = self.get_labels(subject)
        roberta_name = "roberta"
        data = {}
        for name in (
            "openface",
            "opensmile_eGeMAPSv01b",
            "opensmile_prosodyAcf",
            "opensmile_vad_opensource",
            roberta_name,
        ):
            if name == "opensmile_eGeMAPSv01b" and not (self.folder / name).is_dir():
                name = "opensmile_eGeMAPSv01a"
            for key, value in caching.read_hdfs(
                self.folder / name / self.mapping[subject]
            ).items():
                if key == "df":
                    key = name
                else:
                    key = f"{name}_{key}"
                    value = value.add_prefix(f"_{key}")
                if name == "openface":
                    # only keep AUs+head/eye pose
                    value = value.loc[
                        :,
                        [
                            x
                            for x in value.columns
                            if x.startswith("AU")
                            or x.startswith("gaze")
                            or x.startswith("pose")
                            or x == "success"
                        ],
                    ]
                # aggregate to interval
                value = value.add_prefix(f"{name}_")
                stats = suggest_statistics(value.columns.tolist())
                if name != roberta_name:
                    value = apply_masking(value)
                    aggregated = aggregate_to_intervals(
                        value,
                        np.vstack(
                            [labels.index.left.values, labels.index.right.values]
                        ).transpose(),
                        stats,
                    )
                    aggregated.pop("begin")
                    aggregated.pop("end")
                    empty_rows = 0
                    new = defaultdict(list)
                    for begin_end in labels.index:
                        data_slice = value.loc[begin_end.left : begin_end.right, :]
                        if data_slice.shape[0] > 0:
                            for k, v in get_markers(data_slice.copy()).items():
                                if k in aggregated.columns:
                                    continue
                                new[k].extend(([float("NaN")] * empty_rows) + [v])
                            empty_rows = 0
                        else:
                            if not new:
                                empty_rows += 1
                            for key in new.keys():
                                new[key].append(float("NaN"))
                    for k, v in new.items():
                        aggregated[k] = v
                else:
                    aggregated = value
                assert empty_rows == 0
                data[key] = aggregated.reset_index(drop=True)
        # align to labels
        data = pd.concat(
            [pd.concat(data.values(), axis=1), labels.reset_index(drop=True)], axis=1
        )

        # fix NaNs
        data_np = data.values.copy()
        nan_mean = np.nanmean(data.values, axis=0)
        nan_mean[np.isnan(nan_mean)] = 0.0
        inds = np.where(np.isnan(data_np))
        data_np[inds] = np.take(nan_mean, inds[1])
        assert not np.isnan(data_np).any()
        columns = data.columns.tolist()

        data = pd.DataFrame(data_np, columns=columns)
        data["meta_begin"] = labels.index.left
        data["meta_end"] = labels.index.right
        data["meta_id"] = int(subject.split("_", 1)[1])
        return data

    def get_list_of_subjects(self) -> list[str]:
        """
        Returns the list of subject ids.
        """
        return [f"Train_{i:02d}" for i in range(1, 35)] + [
            f"Devel_{i:02d}" for i in range(1, 15)
        ]

    def get_subjects_for_fold(self) -> list[str]:
        if self.name == "test":
            return [f"Devel_{i:02d}" for i in range(1, 15)]

        # load labels for all subjects
        subjects = [f"Train_{i:02d}" for i in range(1, 35)]
        data = [self.get_labels(subject)[self.dimension] for subject in subjects]

        # generate split
        assert self.n_folds >= 4
        groups = np.array([datum.mean() for datum in data])
        fold_index = stratified_splits(groups, np.ones(self.n_folds), new=False)

        # determine relevant subjects
        evaluation = fold_index == self.ifold
        training = ~evaluation
        fold = {"training": training, "validation": evaluation}
        subject_index = fold[self.name]

        return [subject for subject, isin in zip(subjects, subject_index) if isin]


class MOSEI(SEWA):
    def __init__(
        self,
        *,
        dimension: Literal[
            "sentiment",
            "polarity",
            "anger",
            "disgust",
            "fear",
            "surprise",
            "happiness",
            "sadness",
        ] = "sentiment",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        super().__init__(dimension=dimension, ifold=ifold, name=name)
        self.emotions = ("anger", "disgust", "fear", "surprise", "happiness", "sadness")
        self.n_folds = 5
        self.folder = Path("/projects/dataset_processed/MOSEI/twoertwein/")
        sys.path.append(
            str(
                Path(
                    "~/git/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/"
                ).expanduser()
            )
        )
        import cmu_mosei_std_folds

        self.subjects = {
            "training": cmu_mosei_std_folds.standard_train_fold.copy(),
            "validation": cmu_mosei_std_folds.standard_valid_fold.copy(),
            "test": cmu_mosei_std_folds.standard_test_fold.copy(),
        }

        folder = self.folder / "labels/"
        for key, names in self.subjects.items():
            self.subjects[key] = [
                name for name in names if (folder / f"{name}.hdf").is_file()
            ]
        self.mapping = {}
        counter = 0
        for key, names in self.subjects.items():
            for iname, name in enumerate(names):
                assert name not in self.mapping.values()
                self.mapping[f"{key}_{counter}"] = name
                self.subjects[key][iname] = f"{key}_{counter}"
                counter += 1

    def get_cache_path(self) -> Path:
        name = f"MOSEI_{self.dimension}_{self.name}"
        if self.ifold > 0:
            name += f"_{self.ifold}"
        return Path("cache") / f"{name}.pickle"

    def get_subjects_for_fold(self) -> list[str]:
        # first fold is the official split
        if self.ifold == 0 or self.name == "test":
            return self.subjects[self.name]
        # the other 4 training-valdiation folds have the same ratio
        ids = np.array(self.subjects["training"] + self.subjects["validation"])
        means = np.array(
            [self.get_labels(identifier)[self.dimension].mean() for identifier in ids]
        )
        groups = np.ones(round(len(ids) / len(self.subjects["validation"])))
        fold_index = stratified_splits(means, groups)
        if self.name == "training":
            index = fold_index != self.ifold
        else:
            index = fold_index == self.ifold
        return ids[index].tolist()

    def get_list_of_subjects(self) -> list[str]:
        return (
            self.subjects["training"]
            + self.subjects["validation"]
            + self.subjects["test"]
        )

    def get_labels(self, subject: str) -> pd.DataFrame:
        subject = self.mapping[subject]
        data = caching.read_hdfs(self.folder / f"labels/{subject}.hdf")["df"]
        if self.dimension in self.emotions:
            labels = data.loc[:, [self.dimension]].applymap(lambda x: np.nanmean(x))
            sentiment = data.loc[:, ["sentiment"]].applymap(lambda x: np.nanmean(x))
            labels["meta_sentiment"] = sentiment["sentiment"]
            return labels

        labels = data.loc[:, ["sentiment"]].applymap(lambda x: np.nanmean(x))
        labels["meta_sentiment"] = labels["sentiment"]
        if self.dimension == "polarity":
            labels[self.dimension] = labels.pop("sentiment").abs()
        return labels

    def get_loader(self) -> DataLoader:
        loader = super().get_loader()
        # avoid many small clip-level batches
        data = concat_dicts(
            [
                {key: value[0] for key, value in batch.items()}
                for batch in loader.iterator
            ]
        )
        data["meta_begin"] = np.clip(data["meta_begin"], 0.0, None)
        data["meta_end"] = np.clip(data["meta_end"], data["meta_begin"] + 0.1, None)
        if self.dimension == "directionality":
            data["Y"] = data["Y"].astype(int)
            loader.properties["Y_names"] = np.array(["negative", "neutral", "positive"])
        loader.iterator = dict_to_batched_data(data, batch_size=2048, shuffle=True)
        return loader


class MOSI(MOSEI):
    def __init__(
        self,
        *,
        dimension: Literal[
            "mosi",
        ] = "mosi",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        super().__init__(dimension=dimension, ifold=ifold, name=name)
        self.n_folds = 5
        self.folder = Path("MOSI/")
        sys.path.append(
            str(
                Path(
                    "~/git/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI/"
                ).expanduser()
            )
        )
        import cmu_mosi_std_folds

        self.subjects = {
            "training": cmu_mosi_std_folds.standard_train_fold.copy(),
            "validation": cmu_mosi_std_folds.standard_valid_fold.copy(),
            "test": cmu_mosi_std_folds.standard_test_fold.copy(),
        }

        labels = pd.read_csv(self.folder / "labels.csv")
        labels = labels.drop(columns=["Unnamed: 0", "sentence"])
        self.labels = labels.rename(
            columns={
                "name": "meta_id",
                "clip": "meta_clip",
                "begin": "meta_begin",
                "end": "meta_end",
                "sentiment": "mosi",
            }
        )

        ids = self.labels["meta_id"].unique()
        for key, names in self.subjects.items():
            assert all(x in ids for x in names)
        self.mapping = {}
        counter = 0
        for key, names in self.subjects.items():
            for iname, name in enumerate(names):
                assert name not in self.mapping.values()
                self.mapping[f"{key}_{counter}"] = name
                self.subjects[key][iname] = f"{key}_{counter}"
                counter += 1

    def get_cache_path(self) -> Path:
        name = f"MOSI_{self.dimension}_{self.name}"
        if self.ifold > 0:
            name += f"_{self.ifold}"
        return Path("cache") / f"{name}.pickle"

    def get_labels(self, subject: str) -> pd.DataFrame:
        subject_id = int(subject.split("_", 1)[-1])
        subject = self.mapping[subject]
        labels = self.labels.loc[self.labels["meta_id"] == subject].drop(
            columns=["meta_id"]
        )
        labels["meta_id"] = subject_id
        labels.index = pd.IntervalIndex.from_arrays(
            labels["meta_begin"] - 0.5, labels["meta_end"] + 0.5
        )
        return labels


class IEMOCAP(SEWA):
    def __init__(
        self,
        *,
        dimension: Literal["Valence", "Arousal", "Dominance"] = "Valence",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
        correct_audio: bool = True,
    ) -> None:
        assert dimension in ("Valence", "Arousal", "Dominance")
        super().__init__(dimension=dimension, ifold=ifold, name=name)
        self.folder = Path("IEMOCAP")
        self.n_folds = 5
        self.correct_audio = correct_audio
        labels = pd.read_csv(self.folder / "labels.csv", index_col=0)
        labels = labels.drop(columns=["sentence"]).rename(
            columns={
                "valence": "meta_Valence",
                "arousal": "meta_Arousal",
                "dominance": "meta_Dominance",
            }
        )
        labels["meta_name"] = list(map(lambda x: x[:-3], labels.index))
        labels["meta_session"] = list(map(lambda x: int(x[3:5]), labels.index))
        labels["meta_impro"] = (
            labels["meta_name"]
            .str.replace("a", "1")
            .str.replace("b", "2")
            .apply(lambda x: int(x.split("_")[1].split("ro")[1]))
        )
        labels["meta_impro"] = pd.Categorical(labels["meta_impro"]).codes
        labels["meta_female"] = (
            labels["meta_name"].apply(lambda x: "_F" in x).astype(int)
        )
        labels["meta_id"] = (
            labels["meta_session"] * 1000
            + labels["meta_impro"] * 10
            + (labels["meta_female"] + 1) * 3
            + labels["meta_name"].apply(lambda x: "F_" in x).astype(int)
        )
        labels["meta_begin"] -= 0.5
        labels["meta_begin"] = labels["meta_begin"].clip(0)
        labels["meta_end"] += 0.5
        labels["file_name"] = labels["meta_name"].apply(
            lambda x: x.replace("_F", "_left").replace("_M", "_right")
            if "F_" in x
            else x.replace("_M", "_left").replace("_F", "_right")
        )
        assert (
            labels.groupby("meta_id")["file_name"].first()
            == labels.groupby("meta_id")["file_name"].last()
        ).all()
        self.mapping = {
            f"_{key}": value
            for key, value in labels.groupby("meta_id")["file_name"]
            .first()
            .to_dict()
            .items()
        }

        self.labels = labels.drop(columns=["meta_name", "file_name"])

    def get_cache_path(self) -> Path:
        name = f"IEMOCAP_{self.dimension}_{self.name}_{self.ifold}"
        if self.correct_audio:
            name += "_audio"
        return Path("cache") / f"{name}.pickle"

    def get_labels(self, subject: str) -> pd.DataFrame:
        labels = self.labels.loc[self.labels["meta_id"] == int(subject[1:]), :].rename(
            columns={f"meta_{self.dimension}": self.dimension}
        )
        labels.index = pd.IntervalIndex.from_arrays(
            labels["meta_begin"], labels["meta_end"]
        )
        return labels

    def get_subjects_for_fold(self) -> list[str]:
        ids = self.get_list_of_subjects()
        sessions = np.arange(5)
        test = sessions == self.ifold
        validation = sessions == ((self.ifold + 1) % self.n_folds)
        assert not (test & validation).any()
        training = (~test) & (~validation)
        if self.name == "training":
            index = training
        elif self.name == "test":
            index = test
        else:
            assert self.name == "validation"
            index = validation
        sessions += 1
        return [
            identifier
            for identifier in ids
            if identifier[1] in sessions[index].astype(str)
        ]

    def get_list_of_subjects(self) -> list[str]:
        return sorted(("_" + self.labels["meta_id"].astype(str).unique()).tolist())

    def get_loader(self) -> DataLoader:
        loader = super().get_loader()
        # avoid many small clip-level batches
        data = concat_dicts(
            [
                {key: value[0] for key, value in batch.items()}
                for batch in loader.iterator
            ]
        )
        loader.iterator = dict_to_batched_data(data, batch_size=2048, shuffle=True)
        return loader


class Test(SEWA):
    def __init__(
        self,
        *,
        dimension: Literal["uni", "bi"] = "uni",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        super().__init__(dimension=dimension, ifold=ifold, name=name)
        self.n_folds = 5

    def get_cache_path(self) -> Path:
        name = f"Test_{self.dimension}_{self.name}_{self.ifold}"
        return Path("cache") / f"{name}.pickle"

    def get_features(self, subject: str) -> pd.DataFrame:
        subject = int(subject)
        rng = np.random.RandomState(subject)
        abc = rng.multivariate_normal(np.zeros(3), np.eye(3), size=50)
        # avoid finite biases
        abc = np.concatenate([abc, -abc], axis=0)
        abc = np.concatenate(
            [abc, abc * [-1, 1, 1], abc * [1, -1, 1], abc * [1, 1, -1]], axis=0
        )
        abc = (abc - abc.mean(axis=0)) / abc.std(axis=0)
        uni = abc.sum(axis=1)
        bi = abc[:, 0] * abc[:, 1] + abc[:, 0] * abc[:, 2] + abc[:, 1] * abc[:, 2]
        data = pd.DataFrame(
            {
                "meta_uni": uni,
                "meta_bi": bi,
                "openface_a": abc[:, 0],
                "opensmile_b": abc[:, 1],
                "roberta_c": abc[:, 2],
            }
        ).rename(columns={f"meta_{self.dimension}": self.dimension})
        data["meta_id"] = subject
        return data

    def get_subjects_for_fold(self) -> list[str]:
        test = [self.ifold]
        validation = [(self.ifold + 1) % self.n_folds]
        assert test[0] != validation[0]
        if self.name == "test":
            return [str(test[0])]
        elif self.name == "validation":
            return [str(validation[0])]
        assert self.name == "training"
        return (
            np.setdiff1d(np.setdiff1d([0, 1, 2, 3, 4], test), validation)
            .astype(str)
            .tolist()
        )

    def get_list_of_subjects(self) -> list[str]:
        return ["0", "1", "2", "3", "4"]


class Instagram(Test):
    def __init__(
        self,
        *,
        dimension: Literal["intent", "contextual", "semiotic"] = "semiotic",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        super().__init__(dimension=dimension, ifold=ifold, name=name)
        self.n_folds = 5

        self.data = pd.read_csv("instagram.csv").drop(
            columns=["id", "tags", "caption", "orig_caption", "url", "likes"]
        )
        semiotic = ["divergent", "parallel", "additive"]
        contextual = ["minimal", "close", "transcendent"]
        intent = [
            "provoke",
            "inform",
            "advocate",
            "entertain",
            "expose",
            "express",
            "promote",
        ]
        self.data["meta_semiotic"] = self.data.pop("semiotic").apply(
            lambda x: semiotic.index(x)
        )
        self.data["meta_contextual"] = self.data.pop("contextual").apply(
            lambda x: contextual.index(x)
        )
        self.data["meta_intent"] = self.data.pop("intent").apply(
            lambda x: intent.index(x)
        )
        self.data = self.data.rename(columns={f"meta_{self.dimension}": self.dimension})
        self.data["meta_id"] = self.data.index
        self.data = self.data.loc[
            :,
            [self.dimension, "fold"]
            + [x for x in self.data.columns if x.startswith("meta_")]
            + [x for x in self.data.columns if x.startswith("resnet_")]
            + [x for x in self.data.columns if x.startswith("roberta_")],
        ]
        self.dimensions = locals()[self.dimension]

    def get_cache_path(self) -> Path:
        name = f"Instagram_{self.dimension}_{self.name}_{self.ifold}"
        return Path("cache") / f"{name}.pickle"

    def get_features(self, subject: str) -> pd.DataFrame:
        return self.data.loc[self.data["fold"] == int(subject), :].drop(
            columns=["fold"]
        )

    def get_loader(self) -> DataLoader:
        loader = super().get_loader()
        loader.properties["Y_names"] = np.array(self.dimensions)
        for batch in loader.iterator:
            batch["Y"][0] = batch["Y"][0].astype(int)
        return loader


class PANAM(SEWA):
    def __init__(
        self,
        *,
        dimension: Literal["constructs"] = "constructs",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        super().__init__(dimension=dimension, ifold=ifold, name=name)

        partitions, Y_names = caching.read_pickle(Path("panam.pickle"))

    def get_cache_path(self) -> Path:
        name = f"PANAM_{self.dimension}_{self.name}_5_{self.ifold}"
        return Path("cache") / f"{name}.pickle"

    def get_loader(self) -> DataLoader:
        cache = self.get_cache_path()
        data_properties = caching.read_pickle(cache)

        if data_properties is None:
            # load old data
            partitions, Y_names = caching.read_pickle(Path("panam.pickle"))

            partition = partitions[self.ifold][self.name]
            data = partition.iterator
            properties = partition.property_dict
            assert len(set(properties["X_names"])) == len(properties["X_names"])

            caching.write_pickle(cache, (data, properties))
        else:
            data, properties = data_properties
            self.properties = properties
        assert all(x["X"][0].shape[1] == properties["X_names"].shape[0] for x in data)

        return DataLoader(data, properties=deepcopy(properties))
