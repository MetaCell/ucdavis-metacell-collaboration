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
import xarray as xr

from routine.io import load_datasets
from routine.plotting import imshow

IN_EVT = "./intermediate/events_act/"
PARAM_ZTHRES = 2
PARAM_EVT_DICT = {
    "baseline-0": "post-shock-1",
    "baseline-1": "post-shock-2",
    "baseline-2": "post-shock-3",
    "baseline-3": "post-shock-4",
    "baseline-4": "post-shock-5",
    "baseline-5": "baseline",
}
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
FIG_PATH = "./figs/cell_classification/"
os.makedirs(FIG_PATH, exist_ok=True)


def zs_part(df, evt_col="evt", base_lab="baseline-5", val_col="value"):
    df_base = df[df[evt_col] == base_lab]
    base_mean, base_std = df_base[val_col].mean(), df_base[val_col].std()
    df[val_col] = (df[val_col] - base_mean) / base_std
    return df.set_index(evt_col)


def classify_cell(zval):
    if zval > PARAM_ZTHRES:
        return "activated"
    elif zval < -PARAM_ZTHRES:
        return "inhibited"
    else:
        return "non-responsive"


# %% compute zscore and classify cells
cell_df_all = []
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
        cell_df["resp"] = cell_df["zval"].map(classify_cell)
        cell_df_all.append(cell_df)
        act_zs_all.append(act_zs)
cell_df = pd.concat(cell_df_all, ignore_index=True)
act_zs = pd.concat(act_zs_all, ignore_index=True)


# %% plot distribution of cell counts
fig_path = os.path.join(FIG_PATH, "cell_counts")
os.makedirs(fig_path, exist_ok=True)
cell_df_agg = (
    cell_df.groupby(["animal", "session", "evt", "resp", "cls_var"])["unit_id"]
    .count()
    .rename("count")
    .reset_index()
    .replace({"evt": PARAM_EVT_DICT})
    .sort_values("evt", key=lambda evts: [PARAM_EVT_ORD.index(e) for e in evts])
)
for (anm, ss, cls_var), subdf in cell_df_agg.groupby(["animal", "session", "cls_var"]):
    fig = px.pie(subdf, names="resp", values="count", facet_col="evt", facet_col_wrap=3)
    fig.write_html(os.path.join(fig_path, "{}-{}-by{}.html".format(anm, ss, cls_var)))

# %% plot rasters
cell_df_plt = (
    cell_df[cell_df["resp"] == "activated"].copy().replace({"evt": PARAM_EVT_DICT})
)
cell_df_plt["resp"] = cell_df_plt["evt"] + "-" + cell_df_plt["resp"]
cell_df_plt = cell_df_plt.sort_values(
    ["cls_var", "animal", "session", "resp", "zval"]
).set_index(["cls_var", "animal", "session"])
act_agg = (
    act_zs.replace(np.inf, np.nan)
    .dropna(subset=["value"])
    .groupby(["cls_var", "animal", "session", "unit_id", "evt", "frame"])["value"]
    .mean()
    .reset_index()
    .replace({"evt": PARAM_EVT_DICT})
)
fig_path = os.path.join(FIG_PATH, "raster")
os.makedirs(fig_path, exist_ok=True)
for (cls_var, anm, ss), act_df in act_agg.groupby(["cls_var", "animal", "session"]):
    cdf = cell_df_plt.loc[cls_var, anm, ss]
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
        zmax=dat_df["value"].quantile(0.95),
        showscale=False,
        subplot_args={
            "shared_xaxes": "columns",
            "shared_yaxes": "rows",
            "horizontal_spacing": 0.004,
            "vertical_spacing": 0.006,
        },
    )
    fig.update_layout({"height": 1600, "hoverlabel.namelength": -1})
    fig.write_html(os.path.join(fig_path, "{}-{}-by{}.html".format(anm, ss, cls_var)))
