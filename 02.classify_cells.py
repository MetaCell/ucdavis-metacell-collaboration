"""
classify cells based on activities around events

env: environments/generic.yml
"""

# %% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr

from routine.io import load_datasets
from routine.plotting import imshow

IN_EVT = "./intermediate/events_act/"
INT_PATH = "./intermediate/classify_cells"
PARAM_ZTHRES = 2
PARAM_EVT_ORD = [
    "baseline",
    "tone",
    "trace",
    "shock",
    "post-shock-1",
    "post-shock-2",
    "post-shock-3",
    "post-shock-4",
    "post-shock-5",
]
PARAM_CLS = [
    set(c)
    for r in range(3, 0, -1)
    for c in itt.combinations(["tone", "trace", "shock"], r)
] + [
    set(c)
    for c in [
        "post-shock-1",
        "post-shock-2",
        "post-shock-3",
        "post-shock-4",
        "post-shock-5",
        "baseline",
    ]
]
FIG_PATH = "./figs/cell_classification/"
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(INT_PATH, exist_ok=True)


def zs_part(df, evt_col="evt", base_lab="baseline", val_col="value", std_thres=1e-4):
    df_base = df[df[evt_col] == base_lab]
    base_mean, base_std = df_base[val_col].mean(), df_base[val_col].std()
    if base_std > std_thres:
        df[val_col] = (df[val_col] - base_mean) / base_std
    else:
        df[val_col] = np.nan
    return df.set_index(evt_col)


def thres_resp(zval):
    if zval > PARAM_ZTHRES:
        return "activated"
    elif zval < -PARAM_ZTHRES:
        return "inhibited"
    else:
        return "non-responsive"


def classify_cell(cdf, evt_col="evt", zval_col="zval"):
    act_set = set(cdf.loc[cdf[zval_col] > PARAM_ZTHRES, evt_col])
    for test_set in PARAM_CLS:
        if test_set <= act_set:
            suffix = " only" if len(test_set) == 1 else ""
            return "+".join(test_set) + suffix
    return "non-responsive"


# %% compute zscore and classify cells
cell_df_all = []
cell_cls_df_all = []
act_zs_all = []
for (anm, ss), (gpio, ts, ps_ds) in load_datasets():
    for cls_var in ["C", "S"]:
        act_ds = xr.open_dataset(os.path.join(IN_EVT, "{}-{}.nc".format(anm, ss)))
        act_df = act_ds[cls_var].rename("value").to_dataframe().dropna().reset_index()
        act_df["cls_var"] = cls_var
        act_zs = (
            act_df.groupby(["cls_var", "animal", "session", "unit_id", "trial"])
            .apply(zs_part, include_groups=False)
            .reset_index()
        )
        cell_df = (
            act_zs.groupby(["cls_var", "animal", "session", "unit_id", "evt"])["value"]
            .mean()
            .rename("zval")
            .reset_index()
        )
        cell_df["resp"] = cell_df["zval"].map(thres_resp)
        cell_cls_df = (
            cell_df.groupby(["cls_var", "animal", "session", "unit_id"])
            .apply(classify_cell, include_groups=False)
            .rename("cls")
            .reset_index()
        )
        cell_cls_df_all.append(cell_cls_df)
        cell_df_all.append(cell_df)
        act_zs_all.append(act_zs)
cell_df = pd.concat(cell_df_all, ignore_index=True).astype(
    {
        "cls_var": "category",
        "animal": "category",
        "session": "category",
        "unit_id": "category",
        "evt": "category",
    }
)
cell_cls_df = pd.concat(cell_cls_df_all, ignore_index=True).astype(
    {
        "cls_var": "category",
        "animal": "category",
        "session": "category",
        "unit_id": "category",
        "cls": "category",
    }
)
act_zs = pd.concat(act_zs_all, ignore_index=True).astype(
    {
        "cls_var": "category",
        "animal": "category",
        "session": "category",
        "unit_id": "category",
        "evt": "category",
    }
)
cell_df.to_feather(os.path.join(INT_PATH, "cell_df.feat"))
cell_cls_df.to_feather(os.path.join(INT_PATH, "cell_cls_df.feat"))
act_zs.to_feather(os.path.join(INT_PATH, "act_zs.feat"))


# %% plot cell counts for single events
fig_path = os.path.join(FIG_PATH, "cell_counts")
os.makedirs(fig_path, exist_ok=True)
cell_df_agg = (
    cell_df.groupby(["animal", "session", "evt", "resp", "cls_var"], observed=True)[
        "unit_id"
    ]
    .count()
    .rename("count")
    .reset_index()
    .sort_values("evt", key=lambda evts: [PARAM_EVT_ORD.index(e) for e in evts])
)
for (anm, ss, cls_var), subdf in cell_df_agg.groupby(
    ["animal", "session", "cls_var"], observed=True
):
    fig = px.pie(subdf, names="resp", values="count", facet_col="evt", facet_col_wrap=3)
    fig.write_html(
        os.path.join(fig_path, "{}-{}-by{}-single_evt.html".format(anm, ss, cls_var))
    )

# %% plot cell counts for compound events
fig_path = os.path.join(FIG_PATH, "cell_counts")
os.makedirs(fig_path, exist_ok=True)
cell_cls_agg = (
    cell_cls_df.groupby(["cls_var", "animal", "session", "cls"], observed=True)[
        "unit_id"
    ]
    .count()
    .rename("count")
    .reset_index()
)
for (anm, ss, cls_var), subdf in cell_cls_agg.groupby(
    ["animal", "session", "cls_var"], observed=True
):
    fig = px.pie(
        subdf, names="cls", values="count", facet_col="session", facet_col_wrap=3
    )
    fig.write_html(
        os.path.join(fig_path, "{}-{}-by{}-compound_evt.html".format(anm, ss, cls_var))
    )


# %% plot rasters
def reset_uid(df):
    uid_map = {u: i for i, u in enumerate(df["unit_id"].unique())}
    df["uid"] = df["unit_id"].map(uid_map).astype("category")
    return df.set_index("uid")


def combine_act_cell(act_df, cdf, by="resp"):
    return (
        act_df.merge(cdf[["unit_id", by]], on="unit_id", how="right")
        .groupby(by, observed=True)
        .apply(reset_uid, include_groups=False)
        .reset_index()
        .sort_values(["animal", "session"])
        .sort_values("evt", key=lambda evts: [PARAM_EVT_ORD.index(e) for e in evts])
        .sort_values([by, "uid", "frame"])
    )


def agg_by_trial_unit(act_df, cdf, by="resp"):
    by_trial = []
    by_unit = []
    for by, by_df in cdf.groupby(by, observed=True):
        dat_df = combine_act_cell(act_df, by_df)
        assert len(dat_df) <= len(act_df)
        by_unit.append(
            dat_df.groupby(
                [
                    "cls_var",
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
                ["cls_var", "animal", "session", "trial", "resp", "evt", "frame"],
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


def plot_raster(dat_df, by="resp", y="uid"):
    fig = imshow(
        dat_df,
        facet_row=by,
        facet_col="evt",
        x="frame",
        y=y,
        z="value",
        colorscale="viridis",
        zmin=0,
        zmax=dat_df["value"].quantile(0.99),
        showscale=False,
        subplot_args={
            "shared_xaxes": "columns",
            "shared_yaxes": "rows",
            "horizontal_spacing": 0.004,
            "vertical_spacing": 0.006,
        },
    )
    fig.update_layout({"height": 1600, "hoverlabel.namelength": -1})
    return fig


def plot_agg_curve(dat_df, by="resp", hue="trial", show_individual=False):
    g = sns.FacetGrid(
        dat_df,
        row=by,
        col="evt",
        margin_titles=True,
        sharex="row",
        sharey="row",
    )
    if show_individual:
        g.map_dataframe(
            sns.lineplot, x="frame", y="value", hue=hue, errorbar=None, alpha=0.5
        )
    g.map_dataframe(sns.lineplot, x="frame", y="value", estimator="mean", errorbar="se")
    g.add_legend()
    return g.figure


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
    for (cls_var, anm, ss), act_df in act_zs.groupby(
        ["cls_var", "animal", "session"], observed=True
    ):
        cdf = evt_dat.loc[cls_var, anm, ss]
        by_trial, by_unit = agg_by_trial_unit(act_df, cdf)
        dat_dict = {"by_unit": by_unit, "by_trial": by_trial}
        for act_type, dat in dat_dict.items():
            fpath = os.path.join(fig_path, act_type)
            os.makedirs(fpath, exist_ok=True)
            fig = plot_raster(dat, y="uid" if act_type == "by_unit" else "trial")
            fig.write_html(
                os.path.join(
                    fpath, "{}-{}-by{}-{}.html".format(anm, ss, cls_var, evt_type)
                )
            )
            fig_agg = plot_agg_curve(
                dat, show_individual=False if act_type == "by_unit" else True
            )
            fig_agg.savefig(
                os.path.join(
                    fpath, "{}-{}-by{}-{}.svg".format(anm, ss, cls_var, evt_type)
                )
            )
            plt.close(fig_agg)
    for cls_var, act_df in act_zs.groupby("cls_var", observed=True):
        cdf = evt_dat.loc[cls_var].reset_index()
        by_trial, by_unit = agg_by_trial_unit(act_df, cdf)
        dat_dict = {"by_unit": by_unit, "by_trial": by_trial}
        for act_type, dat in dat_dict.items():
            fpath = os.path.join(fig_path, act_type)
            os.makedirs(fpath, exist_ok=True)
            fig = plot_raster(dat, y="uid" if act_type == "by_unit" else "trial")
            fig.write_html(
                os.path.join(fpath, "all-all-by{}-{}.html".format(cls_var, evt_type))
            )
            fig_agg = plot_agg_curve(
                dat, show_individual=False if act_type == "by_unit" else True
            )
            fig_agg.savefig(
                os.path.join(fpath, "all-all-by{}-{}.svg".format(cls_var, evt_type))
            )
            plt.close(fig_agg)


# %% generate trial avg activity with cell class
def reset_uid(df):
    uid_map = {u: i for i, u in enumerate(df["unit_id"].unique())}
    df["uid"] = df["unit_id"].map(uid_map)
    return df


act_agg = (
    act_zs.groupby(
        ["cls_var", "animal", "session", "unit_id", "evt", "frame"], observed=True
    )["value"]
    .mean()
    .reset_index()
)
cell_cls_df_plt = cell_cls_df.sort_values(["cls_var", "animal", "session"]).set_index(
    ["cls_var", "animal", "session"]
)
act_cls = []
for (cls_var, anm, ss), act_df in act_agg.groupby(
    ["cls_var", "animal", "session"], observed=True
):
    cdf = cell_cls_df_plt.loc[cls_var, anm, ss]
    act_df = (
        act_df.merge(cdf, on="unit_id", how="left")
        .groupby("cls")
        .apply(reset_uid, include_groups=False)
        .reset_index()
        .sort_values("evt", key=lambda evts: [PARAM_EVT_ORD.index(e) for e in evts])
        .sort_values(["cls", "uid", "frame"])
    )
    act_cls.append(act_df)
act_cls = pd.concat(act_cls, ignore_index=True)


# %% plot raster of post shock activation
fig_path = os.path.join(FIG_PATH, "post_shock")
os.makedirs(fig_path, exist_ok=True)
for (cls, anm, ss), act_df in act_cls.groupby(
    ["cls_var", "animal", "session"], observed=True
):
    fig = imshow(
        act_df,
        facet_row="cls",
        facet_col="evt",
        x="frame",
        y="uid",
        z="value",
        colorscale="viridis",
        zmin=0,
        zmax=act_df["value"].quantile(0.98),
        showscale=False,
        subplot_args={
            "shared_xaxes": "columns",
            "shared_yaxes": "rows",
            "horizontal_spacing": 0.004,
            "vertical_spacing": 0.006,
        },
    )
    fig.update_layout({"height": 1600, "hoverlabel.namelength": -1})
    fig.write_html(os.path.join(fig_path, "{}-{}-by{}.html".format(anm, ss, cls)))

# %% plot aggregated post-shock activations based on cell class
fig_path = os.path.join(FIG_PATH, "post_shock")
os.makedirs(fig_path, exist_ok=True)
for (cls, anm), act_df in act_cls.groupby(["cls_var", "animal"], observed=True):
    plt_df = act_df[act_df["evt"].map(lambda e: e.startswith("post-shock"))]
    g = sns.FacetGrid(
        plt_df,
        row="session",
        col="cls",
        hue="evt",
        margin_titles=True,
        sharex="row",
        sharey="row",
    )
    g.map_dataframe(sns.lineplot, x="frame", y="value", estimator="mean", errorbar="se")
    g.add_legend()
    g.figure.savefig(os.path.join(fig_path, "{}-agg-by{}.svg".format(anm, cls)))
