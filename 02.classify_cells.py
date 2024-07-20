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
PARAM_CLS_VAR = "C"
PARAM_ZTHRES = 2
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
    act_ds = xr.open_dataset(os.path.join(IN_EVT, "{}-{}.nc".format(anm, ss)))
    act_df = act_ds[PARAM_CLS_VAR].rename("value").to_dataframe().dropna().reset_index()
    act_zs = (
        act_df.groupby(["animal", "session", "unit_id", "trial"])
        .apply(zs_part, include_groups=False)
        .reset_index()
    )
    cell_df = (
        act_zs.groupby(["animal", "session", "unit_id", "evt"])["value"]
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
    cell_df.groupby(["animal", "session", "evt", "resp"])["unit_id"]
    .count()
    .rename("count")
    .reset_index()
)
for (anm, ss), subdf in cell_df_agg.groupby(["animal", "session"]):
    fig = px.pie(subdf, names="resp", values="count", facet_col="evt", facet_col_wrap=3)
    fig.write_html(os.path.join(fig_path, "{}-{}.html".format(anm, ss)))

# %% plot rasters
cell_df_plt = cell_df[cell_df["resp"] == "activated"].copy()
cell_df_plt["resp"] = cell_df_plt["evt"] + "-" + cell_df_plt["resp"]
cell_df_plt = cell_df_plt.sort_values(["animal", "session", "resp", "zval"]).set_index(
    ["animal", "session"]
)
act_agg = (
    act_zs.groupby(["animal", "session", "unit_id", "evt", "frame"])["value"]
    .mean()
    .reset_index()
)
fig_path = os.path.join(FIG_PATH, "raster")
os.makedirs(fig_path, exist_ok=True)
for (anm, ss), act_df in act_agg.groupby(["animal", "session"]):
    cdf = cell_df_plt.loc[anm, ss]
    dat_df = []
    for resp, resp_df in cdf.groupby("resp"):
        resp_df["uid"] = np.arange(len(resp_df))
        dat = act_df.merge(
            resp_df[["unit_id", "uid", "resp", "zval"]], on="unit_id", how="right"
        ).sort_values(["evt", "uid", "frame"])
        dat_df.append(dat)
    dat_df = pd.concat(dat_df, ignore_index=True)
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
    fig.write_html(os.path.join(fig_path, "{}-{}.html".format(anm, ss)))
