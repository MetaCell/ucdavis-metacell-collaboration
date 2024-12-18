"""
export figures based on previous analysis

env: environments/generic.yml
"""

# %% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from tqdm.auto import tqdm

IN_PATH = "./intermediate/classify_cells"
INT_PATH = "./intermediate/export"
FIG_PATH = "./figs/export/"
PARAM_ZTHRES = 2
PARAM_EVTS = {
    "baseline": (0, 400),
    "tone": (0, 400),
    "trace": (0, 400),
    "shock": (0, 40),
    "post-shock-1": (0, 400),
    "post-shock-2": (0, 400),
    "post-shock-3": (0, 400),
    "post-shock-4": (0, 400),
    "post-shock-5": (0, 400),
}
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(INT_PATH, exist_ok=True)


def reset_uid(df):
    uid_map = {u: i for i, u in enumerate(df["unit_id"].unique())}
    df["uid"] = df["unit_id"].map(uid_map).astype("category")
    return df.set_index("uid")


def combine_act_cell(act_df, cdf, by="resp"):
    return (
        act_df.merge(
            cdf[["unit_id", by]], on="unit_id", how="right", validate="many_to_one"
        )
        .groupby(by, observed=True)
        .apply(reset_uid, include_groups=False)
        .reset_index()
        .sort_values(["animal", "session"])
        .sort_values(
            "evt", key=lambda evts: [list(PARAM_EVTS.keys()).index(e) for e in evts]
        )
        .sort_values([by, "uid", "frame"])
    )


def agg_by_trial_unit(act_df, cdf, by_col="resp"):
    by_trial = []
    by_unit = []
    for by, by_df in cdf.groupby(by_col, observed=True):
        dat_df = combine_act_cell(act_df, by_df)
        assert len(dat_df) <= len(act_df)
        by_unit.append(
            dat_df.groupby(
                [
                    "cls_var",
                    "group",
                    "animal",
                    "session",
                    "unit_id",
                    "uid",
                    "resp",
                    "evt",
                    "frame",
                ],
                observed=True,
                sort=False,
            )["value"]
            .mean()
            .reset_index()
        )
        by_trial.append(
            dat_df.groupby(
                [
                    "cls_var",
                    "group",
                    "animal",
                    "session",
                    "trial",
                    "resp",
                    "evt",
                    "frame",
                ],
                observed=True,
                sort=False,
            )["value"]
            .mean()
            .reset_index()
        )
    by_trial, by_unit = pd.concat(by_trial, ignore_index=True), pd.concat(
        by_unit, ignore_index=True
    )
    return by_trial, by_unit


def plot_agg_curve(dat_df, fm_map, by="resp", hue="trial"):
    g = sns.FacetGrid(
        dat_df,
        row=by,
        margin_titles=True,
        hue="group",
        aspect=3.5,
        sharex=True,
        sharey=False,
    )
    g.map_dataframe(
        concat_lineplot,
        fm_map=fm_map,
        estimator="mean",
        errorbar="se",
        lw=2,
        err_kws={"alpha": 0.3},
    )
    g.add_legend()
    g.set_axis_labels(x_var="frame", y_var="fluorescence")
    return g.figure


def concat_lineplot(data, fm_map, color=None, **kwargs):
    ax = plt.gca()
    dat_plt = data.merge(fm_map, on=["evt", "frame"], how="left").dropna()
    sns.lineplot(dat_plt, x="fm_plt", y="value", **kwargs, ax=ax)
    for ievt, (evt, map_df) in enumerate(fm_map.groupby("evt", sort=False)):
        fm0, fm1 = map_df["fm_plt"].min(), map_df["fm_plt"].max()
        alpha = 0.1 if ievt % 2 == 0 else 0
        ax.axvspan(fm0, fm1, facecolor="grey", alpha=alpha)
        ax.text(
            x=(fm0 + fm1) / 2,
            y=1,
            s=evt,
            ha="center",
            va="top",
            rotation="vertical" if fm1 - fm0 < 100 else "horizontal",
            transform=blended_transform_factory(ax.transData, ax.transAxes),
        )


# %% aggregate data
cell_df = pd.read_feather(os.path.join(IN_PATH, "cell_df.feat"))
cell_cls_df = pd.read_feather(os.path.join(IN_PATH, "cell_cls_df.feat"))
act_zs = pd.read_feather(os.path.join(IN_PATH, "act_zs.feat"))
cell_df_plt = cell_df[cell_df["resp"] == "activated"].copy()
cell_df_plt["resp"] = (
    cell_df_plt["evt"].astype(str) + "-" + cell_df_plt["resp"].astype(str)
).astype("category")
cell_df_plt = cell_df_plt.sort_values(
    ["cls_var", "animal", "session", "resp", "zval"]
).set_index(["cls_var", "animal", "session"])
cell_cls_df_plt = (
    cell_cls_df[cell_cls_df["cls"] != "non-responsive"]
    .sort_values(["cls_var", "animal", "session", "cls"])
    .rename(columns={"cls": "resp"})
    .set_index(["cls_var", "animal", "session"])
)
evt_dict = {
    "single_evt": cell_df_plt,
    "compound_evt": cell_cls_df_plt,
}
fig_path = os.path.join(FIG_PATH, "raster")
os.makedirs(fig_path, exist_ok=True)
for evt_type, evt_dat in evt_dict.items():
    by_unit_df = []
    by_trial_df = []
    for (cls_var, grp, anm, ss), act_df in tqdm(
        act_zs.groupby(["cls_var", "group", "animal", "session"], observed=True)
    ):
        try:
            cdf = evt_dat.loc[cls_var, anm, ss]
        except KeyError:
            print("missing events for {}, anm {} ss {}".format(cls_var, anm, ss))
            continue
        by_trial, by_unit = agg_by_trial_unit(act_df, cdf)
        by_unit_df.append(by_unit)
        by_trial_df.append(by_trial)
    by_unit_df = pd.concat(by_unit_df, ignore_index=True, copy=False)
    by_trial_df = pd.concat(by_trial_df, ignore_index=True, copy=False)
    by_unit_df.to_feather(os.path.join(INT_PATH, "by_unit-{}.feat".format(evt_type)))
    by_trial_df.to_feather(os.path.join(INT_PATH, "by_trial-{}.feat".format(evt_type)))


# %% plot agg lines
fig_path = os.path.join(FIG_PATH, "lines_agg")
os.makedirs(fig_path, exist_ok=True)
fm_map_df = None
for evt, evt_rg in PARAM_EVTS.items():
    if fm_map_df is not None:
        last_fm = fm_map_df["fm_plt"].max()
    else:
        last_fm = 0
    map_new = pd.DataFrame(
        {
            "evt": evt,
            "frame": np.arange(*evt_rg),
            "fm_plt": np.arange(*evt_rg) + last_fm + 1,
        }
    )
    fm_map_df = pd.concat([fm_map_df, map_new])
for evt_type, by_type in itt.product(
    ["single_evt", "compound_evt"], ["by_unit", "by_trial"]
):
    dat_df = pd.read_feather(
        os.path.join(INT_PATH, "{}-{}.feat".format(by_type, evt_type))
    )
    for cls_var, ddf in dat_df.groupby("cls_var", observed=True):
        fig = plot_agg_curve(ddf, fm_map=fm_map_df)
        fig.savefig(
            os.path.join(fig_path, "{}-{}-{}.svg".format(evt_type, by_type, cls_var))
        )
        plt.close(fig)
