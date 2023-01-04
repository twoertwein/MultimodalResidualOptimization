#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
from python_tools import caching
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.metrics import concat_dicts
from python_tools.ml.pytorch_tools import dict_to_batched_data


class SEWA:
    def __init__(
        self,
        *,
        dimension: str = "valence",
        ifold: int = 0,
        name: Literal["training", "validation", "test"] = "training",
    ) -> None:
        self.dimension = dimension
        self.ifold = ifold
        self.name = name

    def get_cache_path(self) -> Path:
        return Path("cache") / f"SEWA_{self.dimension}_{self.name}_{self.ifold}.pickle"

    def get_loader(self) -> DataLoader:
        cache = self.get_cache_path()
        data_properties = caching.read_pickle(cache)

        data, properties = data_properties
        self.properties = properties
        assert all(x["x"][0].shape[1] == properties["x_names"].shape[0] for x in data)

        return DataLoader(data, properties=deepcopy(properties))


class MOSEI(SEWA):
    def get_cache_path(self) -> Path:
        return Path("cache") / f"MOSEI_{self.dimension}_{self.name}_{self.ifold}.pickle"

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
        loader.iterator = dict_to_batched_data(data, batch_size=2048, shuffle=True)
        return loader


class MOSI(MOSEI):
    def get_cache_path(self) -> Path:
        return Path("cache") / f"MOSI_{self.dimension}_{self.name}_{self.ifold}.pickle"


class IEMOCAP(SEWA):
    def get_cache_path(self) -> Path:
        return (
            Path("cache") / f"IEMOCAP_{self.dimension}_{self.name}_{self.ifold}.pickle"
        )

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
    def get_cache_path(self) -> Path:
        return Path("cache") / f"Test_{self.dimension}_{self.name}_{self.ifold}.pickle"


class Instagram(Test):
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

    def get_cache_path(self) -> Path:
        return (
            Path("cache")
            / f"Instagram_{self.dimension}_{self.name}_{self.ifold}.pickle"
        )

    def get_loader(self) -> DataLoader:
        loader = super().get_loader()
        loader.properties["y_names"] = np.array(self.dimensions)
        for batch in loader.iterator:
            batch["y"][0] = batch["y"][0].astype(int)
        return loader


class PANAM(SEWA):
    def get_cache_path(self) -> Path:
        return Path("cache") / f"PANAM_{self.dimension}_{self.name}_{self.ifold}.pickle"
