from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr

from create_indirect_study import create_elan_study


def raw_survey(path: Path | pd.DataFrame) -> pd.DataFrame:
    # load and re-code
    data = path
    if isinstance(path, Path):
        data = pd.read_csv(path, skiprows=[1, 2])
    data = (
        data.rename(
            columns={
                "StartDate": "begin",
                "EndDate": "end",
                "Duration (in seconds)": "duration",
                "ResponseId": "id",
            }
        )
        .drop(
            columns=[
                "RecordedDate",
                "DistributionChannel",
                "UserLanguage",
                "video1",
                "video2",
                "audio",
                "audio2",
                "ratings",
            ]
        )
        .set_index("id")
    )
    data = data.loc[data["Progress"] == 100]
    data = data.loc[data["invite"].notna()]
    data = data.loc[data["PROLIFIC_PID"] != "61fdad7b810bc5eede833d47"]
    data = data.loc[data["pid"].str.len() == 24]
    data = data.set_index("PROLIFIC_PID")
    return data


def get_mean(path: Path):
    data = raw_survey(path)
    data = data.loc[:, [x for x in data.columns if x.endswith("1")]]

    # z-norm
    data = (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]

    # weight: correlation with the mean
    mean = data.mean()
    correlations = np.zeros(data.shape[0])
    for irater, rater in enumerate(data.index):
        rated = data.loc[rater]
        index = rated.notna()
        correlations[irater] = pearsonr(mean[index], rated[index])[0]
    correlations = correlations.clip(min=0.0)
    weights = data.notna() * correlations[:, None]
    weights = weights / weights.sum(axis=0)
    return (data * weights).sum(axis=0)


def add_meta_data(path: Path) -> pd.DataFrame:
    data = get_mean(path)
    iemocap = create_elan_study()
    for prefix in ("T", "A", "V", "TA", "VA", "TV", "TAF", "TAV"):
        columns = [f"{prefix}{i}_1" for i in range(100)]
        iemocap[f"{prefix}"] = data.loc[columns].values
    return iemocap


def load(
    path: Path | pd.DataFrame,
    n: int = 6,
) -> pd.DataFrame:
    data = raw_survey(path)
    data = data.loc[:, [x for x in data.columns if x[-1] in ("1", "2") or x == "id"]]

    # find rater subset
    ratings = data.loc[:, [x for x in data.columns if x.endswith("_1")]]

    # prepare for R
    ratings.reset_index(drop=True).T.to_csv("ratings_icc.csv")

    # calculate ICC for each set
    ratings.columns = [x.split("_", 1)[0] for x in ratings.columns]
    fields = ["T", "V", "A", "TA", "TAF", "TV", "VA", "TAV"]
    for key in fields:
        # exclude current fields
        subset = ratings.loc[ratings.index.isin(keep)]

        rating_long = pd.wide_to_long(
            subset.reset_index(), fields, i="PROLIFIC_PID", j="segment"
        )
        result = pg.intraclass_corr(
            data=rating_long.reset_index(),
            targets="segment",
            raters="PROLIFIC_PID",
            ratings=key,
        )
        print(f"\n{key}")
        print(result.loc[[1, 2, 4, 5], :])
