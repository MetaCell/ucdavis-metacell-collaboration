import colorcet as cc
import cv2
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import cm

from .place_cell import df2arr
from .utilities import norm

hv.extension("bokeh")


def plotA_contour(A: xr.DataArray, im: xr.DataArray, cmap=None, im_opts=None):
    im = hv.Image(im, ["width", "height"])
    if im_opts is not None:
        im = im.opts(**im_opts)
    im = im * hv.Path([])
    for uid in A.coords["unit_id"].values:
        curA = (np.array(A.sel(unit_id=uid)) > 0).astype(np.uint8)
        try:
            cnt = cv2.findContours(curA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][
                0
            ].squeeze()
        except IndexError:
            continue
        if cnt.ndim > 1:
            pth = hv.Path(cnt.squeeze())
            if cmap is not None:
                pth = pth.opts(color=cmap[uid])
            im = im * pth
    return im


def plot_overlap(
    im_green: np.ndarray, im_red: np.ndarray, brt_offset=0, return_raw=False
):
    im_red = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(
            np.array(im_red)
        )
        + brt_offset,
        0,
        1,
    )
    im_green = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(
            np.array(im_green)
        )
        + brt_offset,
        0,
        1,
    )
    ovly = np.clip(im_red + im_green, 0, 1)
    if return_raw:
        return ovly, im_green, im_red
    else:
        return ovly


def plot_hd_cells(
    plt_df: pd.DataFrame,
    act: xr.DataArray,
    fr: pd.DataFrame,
    title_cols=None,
    marker_size=1.5,
    figwidth=10,
    ncols=5,
):
    ncell = len(plt_df)
    ncols = min(ncell, ncols)
    nrows = int(np.ceil(ncell / ncols)) * 2
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figwidth, figwidth / ncols * nrows),
        subplot_kw={"projection": "polar"},
    )
    aname = act.name
    for i, mrow in plt_df.reset_index().iterrows():
        uid = mrow["unit_id"]
        if title_cols is not None:
            tt = ", ".join(["{}: {:.2f}".format(t, mrow[t]) for t in title_cols])
        else:
            tt = ""
        irow = int(i / ncols) * 2
        icol = i % ncols
        cur_act = act.sel(unit_id=uid).squeeze().to_dataframe().reset_index()
        act_nz = cur_act[cur_act[aname] > 0].copy()
        act_nz[aname] = norm(act_nz[aname], q=(0.05, 0.95))
        axs[irow, icol].plot(cur_act["hd"], cur_act["frame"], color="tab:blue", lw=0.1)
        axs[irow, icol].scatter(
            act_nz["hd"],
            act_nz["frame"],
            color="tab:red",
            s=act_nz[aname] * marker_size,
            alpha=act_nz[aname],
            zorder=2,
        )
        axs[irow, icol].set_yticklabels([])
        if tt:
            axs[irow, icol].set_title(", ".join(["uid: {}".format(int(uid)), tt]))
        axs[irow + 1, icol].plot(
            fr.loc[uid, "hd"], fr.loc[uid, "fr_norm"], color="tab:blue", lw=1
        )
        axs[irow + 1, icol].fill_between(
            fr.loc[uid, "hd"], fr.loc[uid, "fr_norm"], color="tab:blue", alpha=0.6
        )
        axs[irow + 1, icol].set_yticklabels([])
    fig.tight_layout(pad=0.7)
    return fig


def plot_plc_cells(
    plt_df: pd.DataFrame,
    act: xr.DataArray,
    fr: pd.DataFrame,
    title_cols=None,
    marker_size=1.5,
    figwidth=10,
    ncols=5,
):
    ncell = len(plt_df)
    ncols = min(ncell, ncols)
    nrows = int(np.ceil(ncell / ncols)) * 2
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figwidth, figwidth / ncols * nrows)
    )
    aname = act.name
    for i, mrow in plt_df.reset_index().iterrows():
        uid = mrow["unit_id"]
        if title_cols is not None:
            tt = ", ".join(["{}: {:.2f}".format(t, mrow[t]) for t in title_cols])
        else:
            tt = ""
        irow = int(i / ncols) * 2
        icol = i % ncols
        cur_act = act.sel(unit_id=uid).squeeze().to_dataframe().reset_index()
        act_nz = cur_act[cur_act[aname] > 0].copy()
        act_nz[aname] = norm(act_nz[aname], q=(0.05, 0.95))
        if tt:
            axs[irow, icol].set_title(", ".join(["uid: {}".format(int(uid)), tt]))
        axs[irow, icol].plot(cur_act["x"], cur_act["y"], color="tab:gray", lw=0.1)
        axs[irow, icol].scatter(
            x=act_nz["x"],
            y=act_nz["y"],
            s=act_nz[aname] * marker_size,
            c=act_nz["frame"],
            alpha=act_nz[aname],
            zorder=2,
        )
        axs[irow, icol].set_aspect("equal", adjustable="box")
        axs[irow, icol].get_xaxis().set_visible(False)
        axs[irow, icol].get_yaxis().set_visible(False)
        axs[irow + 1, icol].imshow(df2arr(fr.loc[uid], prob_col="fr_norm").T[::-1, :])
        axs[irow + 1, icol].contour(
            (df2arr(fr.loc[uid], prob_col="fr_lab") == 0).T[::-1, :],
            colors="tab:red",
            linestyles="solid",
            linewidths=0.5,
        )
        axs[irow + 1, icol].get_xaxis().set_visible(False)
        axs[irow + 1, icol].get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.7)
    return fig


def plot_trace(data, x, y, color=None, label=None, **kwargs):
    ax = plt.gca()
    ax = sns.lineplot(data=data, x=x, y=y, color="w", lw=4, **kwargs)
    ax = ax.fill_between(data[x], 0, data[y], color=color, alpha=0.5, zorder=2)
    ax = sns.lineplot(data=data, x=x, y=y, color=color, lw=1, **kwargs)
