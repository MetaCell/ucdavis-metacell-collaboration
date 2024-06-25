import itertools as itt
import os
import re
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

# from svgutils.compose import SVG, Panel, Text
# from lxml import etree


def norm(a, q=(0, 1)):
    amin, amax = np.nanquantile(a, q[0]), np.nanquantile(a, q[1])
    diff = amax - amin
    if diff > 0:
        return np.clip((a - amin) / diff, 0, 1)
    else:
        return np.full_like(a, np.nan)


def norm_xr(x: xr.DataArray, q=1):
    xmin = x.min().compute().values
    if q < 1:
        xmax = x.compute().quantile(q).compute().values
    else:
        xmax = x.max().compute().values
    diff = xmax - xmin
    if diff > 0:
        return ((x - xmin) / diff).clip(0, 1)
    else:
        return x


def walklevel(path, depth=1):
    if depth < 0:
        for root, dirs, files in os.walk(path):
            yield root, dirs[:], files
        return
    elif depth == 0:
        return
    base_depth = path.rstrip(os.path.sep).count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs[:], files
        cur_depth = root.count(os.path.sep)
        if base_depth + depth <= cur_depth:
            del dirs[:]


def subset_sessions(r, anm_pat, ss_pat):
    anm_match = re.search(anm_pat, r["animal"])
    ss_match = re.search(ss_pat, r["session"])
    return bool(anm_match) and bool(ss_match)


def xr_reset_index(ds, dim):
    if type(dim) is not list:
        dim = [dim]
    for d in dim:
        ds = ds.assign_coords({d: np.arange(ds.sizes[d])})
    return ds


def xr_nditer(arr, dims):
    for key in itt.product(*[arr.coords[d].values for d in dims]):
        asub = arr.sel({d: k for d, k in zip(dims, key)}).squeeze()
        for d in asub.dims:
            asub = asub.dropna(d, how="all")
        if len(asub) > 0:
            yield (key, asub)


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def resample_motion(motion, nsmp=None, f_org=None, f_new=None):
    if f_org is None:
        f_org = np.linspace(0, 1, motion.shape[0], endpoint=True)
    if f_new is None:
        f_new = np.linspace(0, 1, nsmp, endpoint=True)
    motion_ret = np.zeros((len(f_new), 2))
    for i in range(2):
        motion_ret[:, i] = interp1d(
            f_org, motion[:, i], bounds_error=False, fill_value="extrapolate"
        )(f_new)
    return motion_ret


def df_set_metadata(dfs: list, meta: dict):
    if type(dfs) is not list:
        return_list = False
        dfs = [dfs]
    else:
        return_list = True
    for df in dfs:
        for k, v in meta.items():
            df[k] = v
    if return_list:
        return dfs
    else:
        return dfs[0]


def df_map_values(dfs: list, mappings: dict):
    if type(dfs) is not list:
        return_list = False
        dfs = [dfs]
    else:
        return_list = True
    for df in dfs:
        for col, mp in mappings.items():
            df[col] = df[mp[0]].map(mp[1])
    if return_list:
        return dfs
    else:
        return dfs[0]


def nan_corr(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    mask = np.logical_and(~np.isnan(a), ~np.isnan(b))
    a, b = a[mask], b[mask]
    a = a - np.mean(a)
    b = b - np.mean(b)
    ssum = (a**2).sum() * (b**2).sum()
    if ssum > 0:
        return (a * b).sum() / np.sqrt(ssum)
    else:
        return np.nan


def corr_mat(a: np.ndarray, b: np.ndarray, agg_axis=0):
    a = a - a.mean(axis=agg_axis, keepdims=True)
    b = b - b.mean(axis=agg_axis, keepdims=True)
    with np.errstate(divide="ignore"):
        return (a * b).sum(axis=agg_axis) / np.sqrt(
            (a**2).sum(axis=agg_axis) * (b**2).sum(axis=agg_axis)
        )


def unique_seg(x: np.ndarray):
    _, x_cat = np.unique(x, return_inverse=True)
    d = np.diff(x_cat, prepend=x_cat[0]) != 0
    return np.cumsum(d)


def thres_gmm(a: np.ndarray, com=-1, pos_thres=0.5) -> np.ndarray:
    ret = np.zeros_like(a)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(a.reshape(-1, 1))
    idg = np.argsort(gmm.means_.reshape(-1))[com]
    s = (gmm.predict(a.reshape(-1, 1)) == idg).reshape(-1)
    if s.sum() / len(s) < pos_thres:
        ret = np.where(s, a, 0)
    return ret


def xrconcat_recursive(var, dims) -> xr.Dataset:
    if len(dims) > 1:
        if type(var) is dict:
            var_dict = var
        elif type(var) is list:
            var_dict = {tuple([np.array(v[d]).item() for d in dims]): v for v in var}
        else:
            raise NotImplementedError("type {} not supported".format(type(var)))
        try:
            var_dict = {k: v.to_dataset() for k, v in var_dict.items()}
        except AttributeError:
            pass
        data = np.empty(len(var_dict), dtype=object)
        for iv, ds in enumerate(var_dict.values()):
            data[iv] = ds
        index = pd.MultiIndex.from_tuples(list(var_dict.keys()), names=dims)
        var_ps = pd.Series(data=data, index=index)
        xr_ls = []
        for idx, v in var_ps.groupby(level=dims[0]):
            v.index = v.index.droplevel(dims[0])
            xarr = xrconcat_recursive(v.to_dict(), dims[1:])
            xr_ls.append(xarr)
        return xr.concat(xr_ls, dim=dims[0])
    else:
        if type(var) is dict:
            var = list(var.values())
        return xr.concat(var, dim=dims[0])


# def make_svg_panel(label, im_path, param_text, im_scale=1, fix_mpl=True, sh=None):
#     im = SVG(im_path, fix_mpl=fix_mpl)
#     if im_scale != 1:
#         im = im.scale(im_scale)
#     lab = Text(label, **param_text)
#     tsize = param_text["size"]
#     if sh is None:
#         x_sh, y_sh = 0.6 * tsize, 1 * tsize
#     else:
#         x_sh, y_sh = sh
#     pan = Panel(im.move(x=x_sh, y=y_sh), lab.move(x=0, y=tsize))
#     pan.height = im.height * im_scale + y_sh
#     pan.width = im.width * im_scale + x_sh
#     return pan


# def svg_unique_id(fname, out_name=None):
#     doc = etree.parse(fname)
#     rt = doc.getroot()
#     nmap = rt.nsmap
#     id_eles = doc.findall("//path[@id]", nmap)
#     ids = list(set([e.attrib["id"] for e in id_eles]))
#     id_dict = {i: str(uuid4())[:8] for i in ids}
#     attr_name = "{{{}}}href".format(nmap["xlink"])
#     for ele in id_eles:
#         ele.attrib["id"] = id_dict[ele.attrib["id"]]
#     for ele in doc.findall("//use[@xlink:href]", nmap):
#         try:
#             ele.attrib[attr_name] = "#" + id_dict[ele.attrib[attr_name][1:]]
#         except KeyError:
#             pass
#     if out_name is None:
#         out_name = fname
#     doc.write(out_name, pretty_print=True)
