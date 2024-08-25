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
cell_df = pd.concat(cell_df_all, ignore_index=True)
cell_cls_df = pd.concat(cell_cls_df_all, ignore_index=True)
act_zs = pd.concat(act_zs_all, ignore_index=True)
cell_df.to_feather(os.path.join(INT_PATH, "cell_df.feat"))
cell_cls_df.to_feather(os.path.join(INT_PATH, "cell_cls_df.feat"))
act_zs.to_feather(os.path.join(INT_PATH, "act_zs.feat"))


# %% plot cell counts for single events
fig_path = os.path.join(FIG_PATH, "cell_counts")
os.makedirs(fig_path, exist_ok=True)
cell_df_agg = (
    cell_df.groupby(["animal", "session", "evt", "resp", "cls_var"])["unit_id"]
    .count()
    .rename("count")
    .reset_index()
    .sort_values("evt", key=lambda evts: [PARAM_EVT_ORD.index(e) for e in evts])
)
for (anm, ss, cls_var), subdf in cell_df_agg.groupby(["animal", "session", "cls_var"]):
    fig = px.pie(subdf, names="resp", values="count", facet_col="evt", facet_col_wrap=3)
    fig.write_html(
        os.path.join(fig_path, "{}-{}-by{}-single_evt.html".format(anm, ss, cls_var))
    )

# %% plot cell counts for compound events
fig_path = os.path.join(FIG_PATH, "cell_counts")
os.makedirs(fig_path, exist_ok=True)
cell_cls_agg = (
    cell_cls_df.groupby(["cls_var", "animal", "session", "cls"])["unit_id"]
    .count()
    .rename("count")
    .reset_index()
)
for (anm, ss, cls_var), subdf in cell_cls_agg.groupby(["animal", "session", "cls_var"]):
    fig = px.pie(
        subdf, names="cls", values="count", facet_col="session", facet_col_wrap=3
    )
    fig.write_html(
        os.path.join(fig_path, "{}-{}-by{}-compound_evt.html".format(anm, ss, cls_var))
    )


# %% plot rasters
def plot_raster(act_df, cdf):
    dat_df = []
    for resp, resp_df in cdf.groupby("resp"):
        resp_df["uid"] = np.arange(len(resp_df))
        dat = act_df.merge(
            resp_df[["unit_id", "uid", "resp", "zval"]], on="unit_id", how="right"
        ).sort_values(["evt", "uid", "frame"])
        dat_df.append(dat)
    dat_df = (
        pd.concat(dat_df, ignore_index=True)
        .sort_values(["animal", "session"])
        .sort_values("evt", key=lambda evts: [PARAM_EVT_ORD.index(e) for e in evts])
        .sort_values(["resp", "uid", "frame"])
    )
    fig = imshow(
        dat_df,
        facet_row="resp",
        facet_col="evt",
        x="frame",
        y="uid",
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


cell_df_plt = cell_df[cell_df["resp"] == "activated"].copy()
cell_df_plt["resp"] = cell_df_plt["evt"] + "-" + cell_df_plt["resp"]
cell_df_plt = cell_df_plt.sort_values(
    ["cls_var", "animal", "session", "resp", "zval"]
).set_index(["cls_var", "animal", "session"])
act_agg = (
    act_zs.groupby(["cls_var", "animal", "session", "unit_id", "evt", "frame"])["value"]
    .mean()
    .reset_index()
)
fig_path = os.path.join(FIG_PATH, "raster")
os.makedirs(fig_path, exist_ok=True)
for (cls_var, anm, ss), act_df in act_agg.groupby(["cls_var", "animal", "session"]):
    cdf = cell_df_plt.loc[cls_var, anm, ss]
    fig = plot_raster(act_df, cdf)
    fig.write_html(
        os.path.join(fig_path, "{}-{}-by{}-single_evt.html".format(anm, ss, cls_var))
    )
for cls_var, act_df in act_agg.groupby("cls_var"):
    cdf = cell_df_plt.loc[cls_var].reset_index()
    fig = plot_raster(act_df, cdf)
    fig.write_html(
        os.path.join(fig_path, "all-all-by{}-single_evt.html".format(cls_var))
    )


# %% generate trial avg activity with cell class
def reset_uid(df):
    uid_map = {u: i for i, u in enumerate(df["unit_id"].unique())}
    df["uid"] = df["unit_id"].map(uid_map)
    return df


act_agg = (
    act_zs.groupby(["cls_var", "animal", "session", "unit_id", "evt", "frame"])["value"]
    .mean()
    .reset_index()
)
cell_cls_df_plt = cell_cls_df.sort_values(["cls_var", "animal", "session"]).set_index(
    ["cls_var", "animal", "session"]
)
act_cls = []
for (cls_var, anm, ss), act_df in act_agg.groupby(["cls_var", "animal", "session"]):
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
for (cls, anm, ss), act_df in act_cls.groupby(["cls_var", "animal", "session"]):
    fig = imshow(
        act_df,
        facet_row="cls",
        facet_col="evt",
        x="frame",
        y="uid",
        z="value",
        colorscale="viridis",
        zmin=0,
        zmax=dat_df["value"].quantile(0.98),
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
for (cls, anm), act_df in act_cls.groupby(["cls_var", "animal"]):
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
