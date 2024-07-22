import copy
import itertools as itt

import colorcet as cc
import cv2
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import cm
from plotly import graph_objects as go
from plotly.colors import convert_colors_to_same_type, unlabel_rgb
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from .utilities import enumerated_product, norm

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
            cnt_scale = np.zeros_like(cnt)
            cnt_scale[:, 0] = A.coords["width"][cnt[:, 0]]
            cnt_scale[:, 1] = A.coords["height"][cnt[:, 1]]
            pth = hv.Path(cnt_scale.squeeze())
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


def facet_plotly(
    data: pd.DataFrame,
    facet_row: str = None,
    facet_col: str = None,
    title_dim: str = None,
    margin_titles: bool = True,
    specs: dict = None,
    col_wrap: int = None,
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    row_crd = data[facet_row].unique()
    col_crd = data[facet_col].unique()
    layout_ls = []
    iiter = 0
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        dat_sub = data[(data[facet_row] == r) & (data[facet_col] == c)]
        if not len(dat_sub) > 0:
            continue
        if title_dim is not None:
            title = dat_sub[title_dim].unique().item()
        else:
            if facet_row == "DUMMY_FACET_ROW":
                title = "{}={}".format(facet_col, c)
            elif facet_col == "DUMMY_FACET_COL":
                title = "{}={}".format(facet_row, r)
            else:
                title = "{}={}; {}={}".format(facet_row, r, facet_col, c)
        if col_wrap is not None:
            ir = iiter // col_wrap
            ic = iiter % col_wrap
            iiter += 1
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": title}
        )
    layout = pd.DataFrame(layout_ls).set_index(["row_label", "col_label"])
    if col_wrap is not None:
        nrow, ncol = int(layout["row"].max() + 1), int(layout["col"].max() + 1)
    else:
        nrow, ncol = len(row_crd), len(col_crd)
    if specs is not None:
        specs = np.full((nrow, ncol), specs).tolist()
    if margin_titles:
        fig = make_subplots(
            rows=nrow,
            cols=ncol,
            row_titles=row_crd.tolist(),
            column_titles=col_crd.tolist(),
            specs=specs,
            **kwargs,
        )
    else:
        fig = make_subplots(
            rows=nrow,
            cols=ncol,
            subplot_titles=layout["title"].values,
            specs=specs,
            **kwargs,
        )
    return fig, layout


def handle_single_facet(data, facet_row, facet_col):
    data = data.copy()
    if facet_row is None:
        data["DUMMY_FACET_ROW"] = ""
        facet_row = "DUMMY_FACET_ROW"
    if facet_col is None:
        data["DUMMY_FACET_COL"] = ""
        facet_col = "DUMMY_FACET_COL"
    return data, facet_row, facet_col


def scatter_3d(
    data,
    facet_row,
    facet_col,
    legend_dim: str = None,
    title_dim: str = None,
    col_wrap: int = None,
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(
        data,
        facet_row,
        facet_col,
        title_dim,
        col_wrap=col_wrap,
        specs={"type": "scene"},
    )
    if legend_dim is not None:
        show_legend = {l: True for l in data[legend_dim].unique()}
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        if legend_dim is not None:
            for llab, subdf in facet_df.groupby(legend_dim, observed=True):
                show_leg = show_legend[llab]
                cur_args = transform_arguments(subdf, kwargs)
                trace = go.Scatter3d(
                    name=llab, legendgroup=llab, showlegend=show_leg, **cur_args
                )
                if show_leg:
                    show_legend[llab] = False
                fig.add_trace(trace, row=ir + 1, col=ic + 1)
        else:
            cur_args = transform_arguments(facet_df, kwargs)
            trace = go.Scatter3d(showlegend=False, **cur_args)
            fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def transform_arguments(data: pd.DataFrame, arguments: dict):
    arg_ret = copy.deepcopy(arguments)
    if arg_ret.get("x"):
        arg_ret["x"] = data[arg_ret["x"]].values
    if arg_ret.get("y"):
        arg_ret["y"] = data[arg_ret["y"]].values
    if arg_ret.get("z"):
        arg_ret["z"] = data[arg_ret["z"]].values
    if arg_ret.get("customdata"):
        arg_ret["customdata"] = data[arg_ret["customdata"]].values
    if arg_ret.get("error_x"):
        x_err = data[arg_ret["error_x"]].values
        if np.nansum(x_err) > 0:
            arg_ret["error_x"] = {"array": x_err}
        else:
            del arg_ret["error_x"]
    if arg_ret.get("error_y"):
        y_err = data[arg_ret["error_y"]].values
        if np.nansum(y_err) > 0:
            arg_ret["error_y"] = {"array": y_err}
        else:
            del arg_ret["error_y"]
    if arg_ret.get("text"):
        try:
            arg_ret["text"] = data[arg_ret["text"]].values
        except KeyError:
            del arg_ret["text"]
    mkopts = arg_ret.get("marker")
    if mkopts:
        if mkopts.get("color"):
            try:
                color = data[mkopts["color"]].values
                if len(np.unique(color)) == 1:
                    mkopts["color"] = np.unique(color).item()
                else:
                    mkopts["color"] = color
            except KeyError:
                pass
        if mkopts.get("size"):
            try:
                size = data[mkopts["size"]].values
                if len(np.unique(size)) == 1:
                    mkopts["size"] = np.unique(size).item()
                else:
                    mkopts["size"] = size
            except KeyError:
                pass
        if mkopts.get("symbol"):
            try:
                symb = data[mkopts["symbol"]].values
                if len(np.unique(symb)) == 1:
                    mkopts["symbol"] = np.unique(symb).item()
                else:
                    mkopts["symbol"] = symb
            except KeyError:
                pass
    return arg_ret


def scatter_agg(
    data,
    x,
    y,
    facet_row,
    facet_col,
    col_wrap: int = None,
    legend_dim: str = None,
    title_dim: str = None,
    show_point_legend=False,
    subplot_args: dict = dict(),
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, col_wrap=col_wrap, **subplot_args
    )
    grp_dims = [facet_row, facet_col, x]
    idx_dims = [facet_row, facet_col]
    if legend_dim is not None:
        grp_dims.append(legend_dim)
        idx_dims.append(legend_dim)
        show_legend = {l: True for l in data[legend_dim].unique()}
    data_agg = (
        data.groupby(grp_dims)[y]
        .agg(["mean", "sem"])
        .reset_index()
        .merge(data, on=grp_dims)
        .set_index(idx_dims)
        .sort_index()
    )
    kwargs["x"] = x
    kwargs["y"] = y
    kwargs_agg = copy.deepcopy(kwargs)
    kwargs_agg["y"] = "mean"
    kwargs_agg["error_y"] = "sem"
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        trace_ls = []
        if legend_dim is not None:
            for llab, subdf in facet_df.groupby(legend_dim, observed=True):
                show_leg = show_legend[llab]
                cur_args = transform_arguments(subdf, kwargs)
                ndata = subdf.groupby(x)[y].count().max()
                if ndata > 1:
                    strp = go.Box(
                        name=llab,
                        legendgroup=llab,
                        showlegend=show_leg and show_point_legend,
                        boxpoints="all",
                        fillcolor="rgba(255,255,255,0)",
                        hoveron="points",
                        line={"color": "rgba(255,255,255,0)"},
                        pointpos=0,
                        opacity=0.4,
                        **cur_args,
                    )
                    trace_ls.append(strp)
                subdf_agg = data_agg.loc[rlab, clab, llab].reset_index()
                cur_args = transform_arguments(subdf_agg, kwargs_agg)
                ln = go.Scatter(
                    name=llab,
                    legendgroup=llab,
                    showlegend=show_leg,
                    mode="lines",
                    **cur_args,
                )
                trace_ls.append(ln)
                if show_leg:
                    show_legend[llab] = False
        else:
            cur_args = transform_arguments(facet_df, kwargs)
            ndata = facet_df.groupby(x)[y].count().max()
            if ndata > 1:
                strp = go.Box(
                    boxpoints="all",
                    fillcolor="rgba(255,255,255,0)",
                    hoveron="points",
                    line={"color": "rgba(255,255,255,0)"},
                    pointpos=0,
                    opacity=0.4,
                    **cur_args,
                )
                trace_ls.append(strp)
            facet_df_agg = data_agg.loc[rlab, clab].reset_index()
            cur_args = transform_arguments(facet_df_agg, kwargs_agg)
            ln = go.Scatter(mode="lines", **cur_args)
            trace_ls.append(ln)
        for t in trace_ls:
            fig.add_trace(t, row=ir + 1, col=ic + 1)
    return fig


def imshow(
    data,
    facet_row,
    facet_col,
    margin_titles=True,
    title_dim: str = None,
    subplot_args: dict = dict(),
    equal_aspect=True,
    **kwargs,
):
    # data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    if equal_aspect:
        xcol, ycol = kwargs["x"], kwargs["y"]
        row_h = data.groupby(facet_row, sort=False)[ycol].nunique()
        col_w = data.groupby(facet_col, sort=False)[xcol].nunique()
        subplot_args["row_heights"] = row_h.tolist()
        subplot_args["column_widths"] = col_w.tolist()
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, margin_titles, **subplot_args
    )
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        cur_args = transform_arguments(facet_df, kwargs)
        trace = go.Heatmap(name="<br>".join([rlab, clab]), **cur_args)
        fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def map_gofunc(
    data,
    func,
    facet_row,
    facet_col,
    margin_titles=True,
    title_dim: str = None,
    subplot_args: dict = dict(),
    **kwargs,
):
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, margin_titles, **subplot_args
    )
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        cur_args = transform_arguments(facet_df, kwargs)
        trace = func(name="<br>".join([rlab, clab]), **cur_args)
        fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def map_colors(a, cc=qualitative.D3, return_colors=False):
    cmap = dict()
    for x, c in zip(a, itt.cycle(cc)):
        cmap[x] = convert_colors_to_same_type(c)[0][0]
    if return_colors:
        return a.map(cmap)
    else:
        return cmap


def add_color_opacity(rgb, alpha):
    r, g, b = unlabel_rgb(rgb)
    return "rgba({},{},{},{})".format(r, g, b, alpha)
