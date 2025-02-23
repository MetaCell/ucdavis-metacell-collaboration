"""
functional connectivity analysis

env: environments/generic.yml
"""

# %% imports and definition
import itertools as itt
import os

import cf_xarray as cfxr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from sklearn.metrics import pairwise_distances

xr.set_options(keep_attrs=True)

IN_ACT = "./intermediate/classify_cells/act_zs.feat"
IN_CELL_LAB = "./intermediate/classify_cells/cell_cls_df.feat"
INT_PATH = "./intermediate/fc_analysis"
FIG_PATH = "./figs/fc_analysis"
PARAM_PRDS = {
    "stim": ["tone", "trace", "shock"],
    "post-shock": [
        "post-shock-1",
        "post-shock-2",
        "post-shock-3",
        "post-shock-4",
        "post-shock-5",
    ],
}

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% load and transform data
lab_id_cols = ["cls_var", "group", "session", "animal", "unit_id"]
grp_cols = ["cls_var", "group", "session"]
act = pd.read_feather(IN_ACT)
cell_lab = pd.read_feather(IN_CELL_LAB).sort_values(lab_id_cols).set_index(lab_id_cols)
outpath = os.path.join(INT_PATH, "prd_arr")
os.makedirs(outpath, exist_ok=True)
for (cls_var, grp, ss), cur_act in act.groupby(grp_cols, observed=True):
    for prd, prd_evts in PARAM_PRDS.items():
        act_sub = cur_act[cur_act["evt"].isin(prd_evts)]
        act_arr = []
        for u, ((anm, uid), act_unit) in enumerate(
            act_sub.groupby(["animal", "unit_id"], observed=True)
        ):
            ulab = cell_lab.loc[cls_var, grp, ss, anm, uid]["cls"]
            a = xr.DataArray(
                np.array(act_unit["value"]).reshape((1, -1)),
                dims=["u", "t"],
                coords={
                    "trial": ("t", np.array(act_unit["trial"])),
                    "evt": ("t", np.array(act_unit["evt"])),
                    "frame": ("t", np.array(act_unit["frame"])),
                    "cls_var": cls_var,
                    "group": grp,
                    "session": ss,
                    "prd": prd,
                    "animal": ("u", [anm]),
                    "unit_id": ("u", [uid]),
                    "cls": ("u", [ulab]),
                },
            ).set_index(t=["trial", "evt", "frame"], u=["cls", "animal", "unit_id"])
            act_arr.append(a)
        # TODO: fillna(0) since the data are z-scored against baseline
        # Think about better potential data source
        act_arr = cfxr.encode_multi_index_as_compress(
            xr.concat(act_arr, "u").fillna(0).rename("prd_arr").to_dataset()
        )
        act_arr.to_netcdf(
            os.path.join(outpath, "{}-{}-{}-{}.nc".format(cls_var, grp, ss, prd))
        )

# %% Compute corr matrix
inpath = os.path.join(INT_PATH, "prd_arr")
figpath = os.path.join(FIG_PATH, "corr")
os.makedirs(figpath, exist_ok=True)
corr_agg = []
for arr_file in os.listdir(inpath):
    # compute corr
    act_arr = cfxr.decode_compress_to_multi_index(
        xr.load_dataset(os.path.join(inpath, arr_file))
    )["prd_arr"].sortby(["cls", "animal", "unit_id"])
    cls_var, grp, ss, prd = (
        act_arr.coords["cls_var"].item(),
        act_arr.coords["group"].item(),
        act_arr.coords["session"].item(),
        act_arr.coords["prd"].item(),
    )
    crd = np.array(act_arr.coords["u"].to_pandas().astype(str))
    crd_cls, crd_anm = np.array(act_arr.coords["cls"]), np.array(
        act_arr.coords["animal"]
    )
    corr = 1 - xr.apply_ufunc(
        pairwise_distances,
        act_arr,
        input_core_dims=[["u", "t"]],
        output_core_dims=[["u0", "u1"]],
        kwargs={"metric": "correlation", "n_jobs": -1},
    ).assign_coords(
        cls0=("u0", crd_cls),
        cls1=("u1", crd_cls),
        anm0=("u0", crd_anm),
        anm1=("u1", crd_anm),
    ).set_index(
        u0=["cls0", "anm0"], u1=["cls1", "anm1"]
    )
    np.fill_diagonal(corr.data, np.nan)
    # plot corr
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=corr,
            x=crd,
            y=crd,
            colorscale="rdbu_r",
            zmid=0,
            colorbar={"title": "correlation"},
        )
    )
    labs, idxs = np.unique(crd_cls, return_index=True)
    for i in idxs:
        fig.add_hline(i, line_width=1, line_dash="dot", line_color="black", opacity=0.5)
        fig.add_vline(i, line_width=1, line_dash="dot", line_color="black", opacity=0.5)
    fig.update_layout(yaxis={"scaleanchor": "x"}, height=1200)
    fig.write_html(
        os.path.join(figpath, "{}-{}-{}-{}.html".format(cls_var, grp, ss, prd))
    )
    # aggregate corr
    corr_pd = corr.rename("corr").to_series().reset_index()
    for (c0, c1, a0, a1), corr_sub in corr_pd.groupby(
        ["cls0", "cls1", "anm0", "anm1"], observed=True
    ):
        if c0 == c1 and a0 == a1:
            cval = pd.Series(
                {
                    "cls_var": cls_var,
                    "group": grp,
                    "session": ss,
                    "prd": prd,
                    "cls": c0,
                    "animal": a0,
                    "corr": corr_sub["corr"].mean(),
                }
            )
            corr_agg.append(cval)
corr_agg = pd.concat(corr_agg, axis="columns").T

# %% plot aggregated corr
g = sns.FacetGrid(corr_agg, row="cls_var", col="prd", aspect=1.5)
g.map_dataframe(
    sns.barplot, x="cls", y="corr", hue="group", palette="tab10", errorbar="se"
)
g.add_legend()
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=45)
g.tight_layout()
g.figure.savefig(os.path.join(figpath, "summary.svg"))
