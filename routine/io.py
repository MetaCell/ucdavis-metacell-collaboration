import functools as fct
import os
import re
from typing import Callable, Optional, Union

import dask as da
import dask.array as darr
import numpy as np
import xarray as xr
from minian.utilities import custom_arr_optimize, load_avi_lazy, load_tif_lazy
from natsort import natsorted

from .isx import Movie


def load_videos(
    vpath: str,
    pattern=r"msCam[0-9]+\.avi$",
    dtype: Union[str, type] = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Load multiple videos in a folder and return a `xr.DataArray`.

    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together and return a `xr.DataArray`
    representation of the concatenated videos. The videos are sorted by
    filenames with :func:`natsort.natsorted` before concatenation. Optionally
    the data can be downsampled, and the user can pass in a custom callable to
    post-process the result.

    Parameters
    ----------
    vpath : str
        The path containing the videos to load.
    pattern : regexp, optional
        The regexp matching the filenames of the videso. By default
        `r"msCam[0-9]+\.avi$"`, which can be interpreted as filenames starting
        with "msCam" followed by at least a number, and then followed by ".avi".
    dtype : Union[str, type], optional
        Datatype of the resulting DataArray, by default `np.float64`.
    downsample : dict, optional
        A dictionary mapping dimension names to an integer downsampling factor.
        The dimension names should be one of "height", "width" or "frame". By
        default `None`.
    downsample_strategy : str, optional
        How the downsampling should be done. Only used if `downsample` is not
        `None`. Either `"subset"` where data points are taken at an interval
        specified in `downsample`, or `"mean"` where mean will be taken over
        data within each interval. By default `"subset"`.
    post_process : Callable, optional
        An user-supplied custom function to post-process the resulting array.
        Four arguments will be passed to the function: the resulting DataArray
        `varr`, the input path `vpath`, the list of matched video filenames
        `vlist`, and the list of DataArray before concatenation `varr_list`. The
        function should output another valide DataArray. In other words, the
        function should have signature `f(varr: xr.DataArray, vpath: str, vlist:
        List[str], varr_list: List[xr.DataArray]) -> xr.DataArray`. By default
        `None`

    Returns
    -------
    varr : xr.DataArray
        The resulting array representation of the input movie. Should have
        dimensions ("frame", "height", "width").

    Raises
    ------
    FileNotFoundError
        if no files under `vpath` match the pattern `pattern`
    ValueError
        if the matched files does not have extension ".avi", ".mkv" or ".tif"
    NotImplementedError
        if `downsample_strategy` is not "subset" or "mean"
    """
    vpath = os.path.normpath(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in (".avi", ".mkv"):
        movie_load_func = load_avi_lazy
    elif file_extension == ".tif":
        movie_load_func = load_tif_lazy
    elif file_extension == ".isxd":
        movie_load_func = load_isxd_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError("unrecognized downsampling strategy")
    varr = varr.rename("fluorescence")
    if post_process:
        varr = post_process(varr, vpath, vlist, varr_list)
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])
    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]
    return varr


def load_isxd_lazy(fname: str, chunksize=100) -> darr.array:
    """
    Lazy load an avi video.

    This function construct a single delayed task for loading the video as a
    whole.

    Parameters
    ----------
    fname : str
        The filename of the video to load.

    Returns
    -------
    arr : darr.array
        The array representation of the video.
    """
    try:
        mov = Movie.read(fname)
    except:
        raise ValueError(fname)
    nfm = mov.timing.num_samples
    h, w = mov.spacing.num_pixels
    fidxs = np.array_split(np.arange(nfm), int(np.ceil(nfm / chunksize)))
    fmread = da.delayed(load_isxd_chunk)
    fms = [
        da.array.from_delayed(
            fmread(fname, fid), dtype=mov.data_type, shape=(len(fid), h, w)
        )
        for fid in fidxs
    ]
    return da.array.concatenate(fms, axis=0)


def load_isxd_chunk(fname: str, fidx: np.array):
    mov = Movie.read(fname)
    return np.stack([mov.get_frame_data(int(f)) for f in fidx], axis=0)
