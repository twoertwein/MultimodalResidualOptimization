from pathlib import Path

import numpy as np
from python_tools import caching
from scipy.stats import pearsonr
from statsmodels.formula.api import ols

from utils import find_correspondence, find_test, get_components
from read_qualtrics import add_meta_data

if __name__ == "__main__":
    for valence in (True, False):
        print()
        print(valence)
        if valence:
            data = add_meta_data(Path("qualtrics/ValenceRatings.csv"))
        else:
            data = add_meta_data(Path("qualtrics/ArousalRatings.csv"))
        ratings = ["T", "A", "V", "TA", "TAF", "VA", "TV", "TAV"]
        data.loc[:, ratings] = (
            data.loc[:, ratings] - data.loc[:, ratings].mean()
        ) / data.loc[:, ratings].std()

        # only uni-modal
        model = ols("TAV ~ T + V + A + 0", data=data).fit()
        print(model.summary())

        # |bi+tri| ~ |uni - y|
        data["res_abs"] = np.abs(model.resid)

        if valence:
            file = find_test("Valence")
        else:
            file = find_test("Arousal")
        data["bi_y_delta"] = 0
        data["tri_y_delta"] = 0
        data["y_hat"] = 0

        for fold, datum in enumerate(caching.read_pickle(file)):
            y_uni, y_bi, y_tri, datum = get_components(file, fold, datum)
            matches = find_correspondence(data, datum)
            index_a, index_b = np.nonzero(matches)
            data.loc[index_a, "bi_y_delta"] = datum["bi_y_delta"][index_b, 0]
            data.loc[index_a, "tri_y_delta"] = datum["tri_y_delta"][index_b, 0]
            data.loc[index_a, "y_hat"] = y_tri[index_b, 0]
        print(
            pearsonr(data["res_abs"], (data["bi_y_delta"] + data["tri_y_delta"]).abs())
        )
