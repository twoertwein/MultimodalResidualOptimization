#!/usr/bin/env python3
import argparse
from functools import partial
from pathlib import Path
from typing import Any, Final

import torch
from python_tools.generic import namespace_as_string
from python_tools.ml import metrics, neural
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.default.neural_models import MLPModel
from python_tools.ml.default.transformations import (
    DefaultTransformations,
    revert_transform,
    set_transform,
)
from python_tools.ml.evaluator import evaluator

from dataloader import IEMOCAP, MOSEI, MOSI, PANAM, SEWA, Instagram, Test
from routing import MMRouting

CCC_DIMENSION = ("valence", "arousal")
MAE_DIMENSION = (
    # MOSEI
    "sentiment",
    "polarity",
    "happiness",
    # MOSI sentiment
    "mosi",
    # Test
    "uni",
    "bi",
    # IEMOCAP
    "Valence",
    "Arousal",
)
CLASS_DIMENSION = ("constructs", "intent")


class RoutingTFN(neural.LossModule):
    # each uni/bi/tri has its own TFN
    # embedding from TFNs need linear projection to have same size
    # routing creates new embedding
    # linear layer for prediction
    def __init__(
        self,
        *,
        input_size: int = -1,
        output_size: int = -1,
        layer_sizes: tuple[int, ...],
        input_sizes: tuple[int, ...],
        final_activation: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            loss_function=kwargs.get("loss_function", "MSELoss"),
            attenuation=kwargs.get("attenuation", ""),
            attenuation_lambda=kwargs.get("attenuation_lambda", 0.0),
            sample_weight=kwargs.get("sample_weight", None),
            training_validation=kwargs.get("training_validation", False),
        )
        input_sizes = tuple(x for x in input_sizes if x)
        assert input_size == sum(input_sizes)

        if len(input_sizes) == 2:
            tfns = [(input_sizes[0],), (input_sizes[1],), input_sizes]
        elif len(input_sizes) == 3:
            tfns = [
                (input_sizes[0],),
                (input_sizes[1],),
                (input_sizes[2],),
                (input_sizes[0], input_sizes[1]),
                (input_sizes[1], input_sizes[2]),
                (input_sizes[0], input_sizes[2]),
                input_sizes,
            ]
        else:
            raise AssertionError(input_sizes)
        # {MLP+TFN} routing, linear
        self.view_starts: Final = torch.cumsum(
            torch.LongTensor([0] + list(input_sizes)), dim=0
        )

        kwargs.pop("layers")
        # TFNs:
        self.tfns = torch.nn.ModuleList(
            [
                MLP_mult(
                    input_size=sum(input_sizes_),
                    output_size=layer_sizes[-1] + 1,
                    layer_sizes=layer_sizes[:-2] + (min(30, layer_sizes[-2]),),
                    layers=-1,
                    input_sizes=input_sizes_,
                    final_activation=kwargs["activation"],
                    save=False,
                    **kwargs,
                )
                for input_sizes_ in tfns
            ]
        )
        # Routing
        self.routing = MMRouting(
            in_capsules=len(tfns),
            input_size=layer_sizes[-1],
            out_capsules=output_size,
            output_size=5,
            iterations=10 if output_size > 1 else 1,
        )
        # prediction
        self.prediction = torch.nn.ModuleList(
            [torch.nn.Linear(5, 1) for i in range(output_size)]
        )
        self.jit_me = False

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: torch.Tensor | None = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # TFN + projection to capsule size
        x = torch.stack(
            # uni-modal
            [
                self.tfns[i](
                    x[:, start : self.view_starts[i + 1]], meta, y=y, dataset=dataset
                )[0]
                for i, start in enumerate(self.view_starts[:-1])
            ]
            # bi-modal
            + (
                [
                    self.tfns[3](
                        x[:, : self.view_starts[-1]], meta, y=y, dataset=dataset
                    )[0],
                    self.tfns[4](
                        x[:, self.view_starts[1] :], meta, y=y, dataset=dataset
                    )[0],
                    self.tfns[5](
                        torch.cat(
                            [x[:, : self.view_starts[1]], x[:, self.view_starts[-2] :]],
                            dim=1,
                        ),
                        meta,
                        y=y,
                        dataset=dataset,
                    )[0],
                ]
                if len(self.tfns) == 7
                else []
            )
            # tri-modal
            + [self.tfns[-1](x, meta, y=y, dataset=dataset)[0]],
            dim=1,
        )

        p_bi = torch.sigmoid(x[:, :, -1])
        meta["meta_p_i"] = p_bi
        # final prediction is a linear function: routing weights can be applied as
        # weighted sum!
        for imod in range(x.shape[1]):
            x_ = x[:, imod, :-1] * p_bi[:, imod, None]
            for ilabel, label_fun in enumerate(self.prediction):
                meta[f"meta_m{imod}_l{ilabel}"] = torch.einsum(
                    "bi, io, co-> bc",
                    x_,
                    self.routing.weights[imod, :, ilabel],
                    label_fun.weight,
                )

        # capsule
        r = self.routing(f_bid=x[:, :, :-1], p_bi=p_bi)[1]
        meta["meta_r_ij"] = r.view(r.shape[0], -1)
        meta["meta_embedding"] = meta["meta_r_ij"]
        # predict
        y_hat = torch.cat(
            [
                sum(
                    [
                        meta[f"meta_m{i}_l{l}"] * r[:, i, l, None]
                        for i in range(r.shape[1])
                    ],
                    start=self.prediction[l].bias,
                )
                for l in range(r.shape[2])
            ],
            axis=1,
        )

        return y_hat, meta


class MLP_mult(neural.LossModule):
    # multiplicative interactions before the last layer
    def __init__(
        self,
        *,
        input_size: int = -1,
        output_size: int = -1,
        layer_sizes: tuple[int, ...],
        input_sizes: tuple[int, ...],
        final_activation: dict[str, Any],
        residual: bool = False,
        save: bool = True,
        only_tri: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            loss_function=kwargs.get("loss_function", "MSELoss"),
            attenuation=kwargs.get("attenuation", ""),
            attenuation_lambda=kwargs.get("attenuation_lambda", 0.0),
            sample_weight=kwargs.get("sample_weight", None),
            training_validation=kwargs.get("training_validation", False),
        )
        self.residual: Final = residual
        self.save: Final = save
        self.only_tri: Final = only_tri
        input_sizes = tuple(x for x in input_sizes if x)
        assert input_size == sum(input_sizes)
        if only_tri:
            assert not residual

        self.view_starts: Final = torch.cumsum(
            torch.LongTensor([0] + list(input_sizes)), dim=0
        )

        kwargs.pop("layers")
        self.before = torch.nn.ModuleList(
            [
                neural.MLP(
                    input_size=size,
                    output_size=layer_sizes[-1] if layer_sizes else size,
                    layer_sizes=layer_sizes[:-1] if layer_sizes else None,
                    layers=-1,
                    final_activation=kwargs["activation"],
                    **kwargs,
                )
                for size in input_sizes
            ]
        )
        # determine combined size and uni/bi/tri parts
        size = layer_sizes[-1] if layer_sizes else input_sizes[0]
        if not layer_sizes:
            assert all(size == input_sizes[0] for size in input_sizes)
        combined = neural.interaction_terms(
            [torch.ones(1, size) * value for _, value in zip(input_sizes, [2, 3, 7])],
            append_one=True,
        )
        self.uni = (combined[0] == 2) | (combined[0] == 3) | (combined[0] == 7)
        self.bi = (combined[0] == 6) | (combined[0] == 14) | (combined[0] == 21)
        self.tri = combined[0] == 42
        if only_tri:
            self.uni[:] = False
            self.bi[:] = False
            self.tri[:] = True
        assert self.uni.sum() + self.bi.sum() + self.tri.sum() == combined.shape[1]

        self.mult = neural.InteractionModel(
            input_sizes=tuple([size] * len(input_sizes)),
            append_one=True,
        )

        kwargs["layers"] = 0
        self.after = torch.nn.ModuleList(
            [
                neural.MLP(
                    input_size=int(index.sum().item()),
                    output_size=output_size,
                    final_activation=final_activation,
                    **kwargs,
                )
                for index in (self.uni, self.bi, self.tri)
                if index.any()
            ]
        )
        self.jit_me = False

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: torch.Tensor | None = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # get uni-modal representations
        x = torch.cat(
            [
                self.before[i](
                    x[:, start : self.view_starts[i + 1]],
                    meta,
                    y=y,
                    dataset=dataset,
                )[0]
                for i, start in enumerate(self.view_starts[:-1])
            ],
            dim=1,
        )
        # TFN
        x = self.mult(x, meta, y=y, dataset=dataset)[0]
        index_uni = self.uni
        index_bi = self.bi
        index_tri = self.tri
        if self.save:
            meta["meta_tfn_embedding"] = x
        # predictions from each feature set
        y_hat = 0
        excluded = tuple(x for x in self.exclude_parameters_prefix if "before" not in x)
        for i, (model, index) in enumerate(
            zip(self.after, (index_uni, index_bi, index_tri))
        ):
            y_, meta = model(x[:, index], meta, y=y, dataset=dataset)
            if self.save:
                meta[f"meta_y_hat_{i}"] = y_
            if self.only_tri and not index.any():
                y_ = 0
                if self.save:
                    meta[f"meta_y_hat_{i}"] = meta[f"meta_y_hat_{i}"] * 0
            y_hat = y_hat + y_

            # uni: ("after.1", "after.2") or ("after.1",)
            if i == 0 and excluded in (
                ("after.1", "after.2"),
                ("after.1",),
            ):
                break
            # bi: ("after.0", "after.2") or ("after.0")
            elif i == 1 and excluded in (
                ("after.0", "after.2"),
                ("after.0",),
                # joint
                ("after.2",),
            ):
                break

        return y_hat, meta

    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert loss is None
        loss = 0

        if not self.residual or self.exclude_parameters_prefix:
            return super().loss(scores, ground_truth, meta, take_mean=take_mean)

        uni = super().loss(meta["meta_y_hat_0"], ground_truth, meta)
        previous = meta["meta_y_hat_0"].detach()
        bi = super().loss(meta["meta_y_hat_1"] + previous, ground_truth, meta)
        previous = previous + meta["meta_y_hat_1"].detach()
        tri = 0.0
        if "meta_y_hat_2" in meta:
            tri = super().loss(meta["meta_y_hat_2"] + previous, ground_truth, meta)

        return loss + uni + bi + tri


class MLP_parallel(neural.LossModule):
    # additive baseline
    def __init__(
        self,
        *,
        input_size: int = -1,
        input_sizes: tuple[int, ...],
        **kwargs,
    ) -> None:
        super().__init__(
            loss_function=kwargs.get("loss_function", "MSELoss"),
            attenuation=kwargs.get("attenuation", ""),
            attenuation_lambda=kwargs.get("attenuation_lambda", 0.0),
            sample_weight=kwargs.get("sample_weight", None),
            training_validation=kwargs.get("training_validation", False),
        )
        input_sizes = tuple(x for x in input_sizes if x)
        assert input_size == sum(input_sizes)

        self.view_starts: Final = torch.cumsum(
            torch.LongTensor([0] + list(input_sizes)), dim=0
        )

        fun = neural.MLP
        self.before = torch.nn.ModuleList(
            [fun(input_size=size, **kwargs) for size in input_sizes]
        )

        self.jit_me = False

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: torch.Tensor | None = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        xs = []

        for i, start in enumerate(self.view_starts[:-1]):
            x_, meta = self.before[i](
                x[:, start : self.view_starts[i + 1]], meta, y=y, dataset=dataset
            )
            xs.append(x_)
            meta[f"meta_tfn_embedding_{i}"] = meta["meta_embedding"]
        x = torch.stack(xs, dim=-1)
        meta["meta_tfn_embedding"] = torch.cat(
            [
                meta[f"meta_tfn_embedding_{i}"]
                for i in range(self.view_starts.shape[0] - 1)
            ],
            dim=1,
        )
        meta["meta_embedding"] = x.view(x.shape[0], -1)
        meta["meta_parallel_y"] = meta["meta_embedding"]

        return x.sum(dim=-1), meta


class MLP_detached_residual(neural.LossModule):
    # uni-modalS + additive bi-modalS on residual + tri-modal on residual
    def __init__(
        self,
        *,
        input_size: int = -1,
        input_sizes: tuple[int, ...],
        naive_loss: bool = False,
        only_tri: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            loss_function=kwargs.get("loss_function", "MSELoss"),
            attenuation=kwargs.get("attenuation", ""),
            attenuation_lambda=kwargs.get("attenuation_lambda", 0.0),
            sample_weight=kwargs.get("sample_weight", None),
            training_validation=kwargs.get("training_validation", False),
        )
        input_sizes = tuple(x for x in input_sizes if x)
        assert input_size == sum(input_sizes)

        self.naive_loss: Final = naive_loss
        self.only_tri: Final = only_tri
        self.view_starts: Final = torch.cumsum(
            torch.LongTensor([0] + list(input_sizes)), dim=0
        )
        if only_tri:
            assert naive_loss

        self.jit_me = False
        self.uni_modals = MLP_parallel(
            input_size=input_size,
            input_sizes=input_sizes,
            **kwargs,
        )

        fun = neural.MLP

        self.tri_modal = fun(input_size=input_size, **kwargs)
        if len(input_sizes) == 3:
            self.bi_a_modal = fun(input_size=input_sizes[0] + input_sizes[1], **kwargs)

            self.bi_a_index = torch.zeros(input_size, dtype=bool)
            self.bi_a_index[: input_sizes[0] + input_sizes[1]] = True
            self.bi_b_modal = fun(input_size=input_sizes[1] + input_sizes[2], **kwargs)

            self.bi_b_index = torch.zeros(input_size, dtype=bool)
            self.bi_b_index[-input_sizes[1] - input_sizes[2] :] = True
            self.bi_c_modal = fun(input_size=input_sizes[0] + input_sizes[2], **kwargs)
            self.bi_c_index = torch.zeros(input_size, dtype=bool)
            self.bi_c_index[: input_sizes[0]] = True
            self.bi_c_index[-input_sizes[-1] :] = True

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: torch.Tensor | None = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_uni, meta = self.uni_modals(x, meta, y=y, dataset=dataset)
        meta["meta_uni_y"] = y_uni

        y_bis = 0
        if self.view_starts.shape[0] == 4 and self.exclude_parameters_prefix != (
            "bi_",
            "tri_",
        ):
            y_bi_a, meta = self.bi_a_modal(
                x[:, self.bi_a_index], meta, y=y, dataset=dataset
            )
            meta["meta_bi_y_a"] = y_bi_a

            y_bi_b, meta = self.bi_b_modal(
                x[:, self.bi_b_index], meta, y=y, dataset=dataset
            )
            meta["meta_bi_y_b"] = y_bi_b

            y_bi_c, meta = self.bi_c_modal(
                x[:, self.bi_c_index], meta, y=y, dataset=dataset
            )
            meta["meta_bi_y_c"] = y_bi_c
            y_bis = y_bi_a + y_bi_b + y_bi_c
            meta["meta_bi_y"] = y_bis

        y_tri = 0
        if "tri_" not in self.exclude_parameters_prefix:
            y_tri, meta = self.tri_modal(x, meta, y=y, dataset=dataset)
            meta["meta_tri_y"] = y_tri
        if self.only_tri:
            y_uni = 0
            y_bis = 0
            meta["meta_uni_y"] = meta["meta_uni_y"] * 0
            if "meta_bi_y" in meta:
                meta["meta_bi_y"] = meta["meta_bi_y"] * 0
        return y_uni + y_bis + y_tri, meta

    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert loss is None
        loss = 0
        if self.naive_loss or self.exclude_parameters_prefix:
            return loss + super().loss(scores, ground_truth, meta, take_mean=take_mean)

        uni = super().loss(meta["meta_uni_y"], ground_truth, meta)
        previous = meta["meta_uni_y"].detach()
        bi = 0
        if "meta_bi_y" in meta:
            bi = super().loss(meta["meta_bi_y"] + previous, ground_truth, meta)
            previous = previous + meta["meta_bi_y"].detach()
        tri = super().loss(meta["meta_tri_y"] + previous, ground_truth, meta)

        return loss + uni + bi + tri


def get_modalities(xs: list[str]) -> dict[str, list[str]]:
    return {
        "vision": [
            x
            for x in xs
            if x.startswith("openface")
            or x == "duchenne_smile_ratio"
            or x.startswith("detr")
            or x.startswith("resnet")
            or x.startswith("afar")
        ],
        "acoustic": [
            x
            for x in xs
            if x.startswith("opensmile")
            or x.startswith("covarep")
            or x.startswith("volume")
        ],
        "language": [x for x in xs if x.startswith("roberta") or x.startswith("liwc")],
    }


def train(
    partitions: dict[int, dict[str, DataLoader]], folder: Path, args: argparse.Namespace
) -> None:
    metric = "ccc"
    metric_fun = metrics.interval_metrics
    params = {
        "interval": True,
        "metric_max": True,
        "y_names": partitions[0]["training"].properties["y_names"].copy(),
    }

    grid_search = {
        "epochs": [10_000],
        "early_stop": [300],
        "lr": [0.001, 0.0001, 0.00001],
        "dropout": [0.0],
        "layers": [0],
        "layer_sizes": [(5,), (10,), (20, 10), (10, 20)],
        "activation": [{"name": "ReLU"}],
        "attenuation": [""],
        "sample_weight": [True],
        "final_activation": [{"name": "linear"}],
        "minmax": [False, True],
        "weight_decay": [1e-4, 1e-3, 1e-2],
    }

    if args.dimension in ("uni", "bi", "tri"):
        grid_search["epochs"] = [1_000]
        grid_search["minmax"] = [False]
        grid_search["weight_decay"] = [0.0]
        grid_search["layer_sizes"] = [(5,), (10,), (10, 10)]
        metric = "mse"
        params["metric_max"] = False
    elif args.dimension in MAE_DIMENSION:
        grid_search["lr"].extend([0.005, 0.01, 0.05])
        grid_search["layer_sizes"].extend([(100, 20, 10), (100, 100, 10)])
        grid_search["weight_decay"].extend([0.0])
        metric = "mae"
        params["metric_max"] = False
        grid_search["loss_function"] = ["L1Loss"]
    elif args.dimension in CLASS_DIMENSION:
        grid_search["lr"].extend([0.005, 0.01, 0.05])
        if args.dimension != "intent":
            grid_search["layer_sizes"].extend([(100, 20, 10), (100, 100, 10)])
        grid_search["weight_decay"].extend([0.0])
        grid_search["sample_weight"] = [False]
        grid_search["minmax"] = [False]
        params["interval"] = False
        params["nominal"] = True
        params["y_names"] = partitions[0]["training"].properties["y_names"].copy()
        metric_fun = metrics.nominal_metrics
        metric = "brier_score"
        params["metric_max"] = False
    if args.routing:
        # need more layers
        grid_search["layer_sizes"] = [
            x if len(x) > 1 else x + (5,) for x in grid_search["layer_sizes"]
        ]

    model = MLPModel(device="cuda", **params)

    x_names = partitions[0]["training"].properties["x_names"]
    modalities = get_modalities(x_names.tolist())
    assert partitions[0]["training"].properties["x_names"].tolist() == sum(
        modalities.values(), []
    )

    # add feature names
    new_names = ["input_sizes"]
    if args.routing:
        # routing:
        grid_search["model_class"] = [RoutingTFN]
    elif not args.mult:
        # early fusion
        grid_search["model_class"] = [MLP_detached_residual]
        if args.joint:
            # naive joint loss
            grid_search["naive_loss"] = [True]
            new_names += ["naive_loss"]
            if args.tri:
                grid_search["only_tri"] = [True]
                new_names += ["only_tri"]
        if args.stepwise:
            # freeze bi+tri, uni+tri, bi+tri
            grid_search["exclude_parameters_prefixes"] = [
                (("bi_", "tri_"), ("uni_", "tri_"), ("uni_", "bi_"))
            ]
            if args.dimension == "intent" or (
                args.dimension in ("arousal",) and args.fs
            ):
                # only two modalities
                grid_search["exclude_parameters_prefixes"] = [(("tri_",), ("uni_",))]
    else:
        # tensor fusion
        grid_search["model_class"] = [MLP_mult]
        if args.joint:
            # Joint
            if args.tri:
                # Tri
                grid_search["only_tri"] = [True]
                new_names += ["only_tri"]
        elif args.res_det:
            # MRO
            grid_search["residual"] = [True]
            new_names += ["residual"]
        if args.stepwise and not args.joint:
            # sMRO
            # freeze bi+tri, uni+tri, bi+tri
            grid_search["exclude_parameters_prefixes"] = [
                (
                    ("after.1", "after.2"),
                    ("after.0", "after.2"),
                    ("after.0", "after.1"),
                )
            ]
            if args.dimension == "intent" or (
                args.dimension in ("arousal",) and args.fs
            ):
                # only two modalities
                grid_search["exclude_parameters_prefixes"] = [
                    (("after.1",), ("after.0",))
                ]

    grid_search["input_sizes"] = [
        (
            tuple(modalities["vision"]),
            tuple(modalities["acoustic"]),
            tuple(modalities["language"]),
        )
    ]
    model.forward_names = tuple(list(model.forward_names) + new_names)

    model.parameters.update(grid_search)
    models, parameters, model_transform = model.get_models()

    apply_transformation = partial(
        combine_transformations, model_transform=model_transform
    )

    transform = DefaultTransformations(**params)
    transforms = tuple([{"feature_selection": args.fs} for _ in range(len(partitions))])

    print(folder, len(parameters))
    for key, value in model.parameters.items():
        if len(value) > 1:
            print(len(value), key, value)

    evaluator(
        models=models,
        partitions=partitions,
        parameters=parameters,
        folder=folder,
        metric_fun=partial(
            metric_fun,
            clustering=args.dimension in CCC_DIMENSION
            or args.dimension == "constructs",
            names=tuple(params["y_names"].tolist()),
        ),
        metric=metric,
        metric_max=params["metric_max"],
        learn_transform=transform.define_transform,
        apply_transform=apply_transformation,
        revert_transform=revert_transform,
        transform_parameter=transforms,
        workers=args.workers,
        parallel="local",
        memory_limit=None,
    )


def combine_transformations(data, transform, model_transform=None):
    if data.properties["y_names"][0] in ("uni", "bi"):
        transform["x"]["mean"][:] = 0
        transform["x"]["std"][:] = 1
        transform["y"]["mean"][:] = 0
        transform["y"]["std"][:] = 1
    data = set_transform(data, transform)
    data.add_transform(model_transform, optimizable=True)
    return data


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimension",
        choices=list(CCC_DIMENSION) + list(MAE_DIMENSION) + list(CLASS_DIMENSION),
        default="valence",
    )
    parser.add_argument("--routing", action="store_const", const=True, default=False)
    parser.add_argument("--fs", action="store_const", const=True, default=False)
    parser.add_argument("--mult", action="store_const", const=True, default=False)
    parser.add_argument("--res_det", action="store_const", const=True, default=False)
    parser.add_argument("--joint", action="store_const", const=True, default=False)
    parser.add_argument("--stepwise", action="store_const", const=True, default=False)
    parser.add_argument("--tri", action="store_const", const=True, default=False)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if args.dimension in ("uni", "bi", "tri") and not args.routing:
        if args.fs:
            exit(code=0)
        if args.dimension in ("bi", "tri") and not args.mult:
            exit(code=0)
        if args.dimension == "uni" and args.mult:
            exit(code=0)

    # choose dataloader
    folder = Path("experiments") / namespace_as_string(args, exclude=("workers",))
    loader_class = SEWA
    folds = 5
    if args.dimension == "mosi":
        loader_class = MOSI
    elif args.dimension[0] != args.dimension[0].lower():
        loader_class = IEMOCAP
    elif args.dimension in ("uni", "bi"):
        loader_class = Test
    elif args.dimension in MAE_DIMENSION:
        loader_class = MOSEI
    elif args.dimension == "constructs":
        loader_class = PANAM
    elif args.dimension == "intent":
        loader_class = Instagram

    data = {
        i: {
            name: loader_class(
                ifold=i, name=name, dimension=args.dimension
            ).get_loader()
            for name in ("training", "validation", "test")
        }
        for i in range(folds)
    }

    train(data, folder, args)
