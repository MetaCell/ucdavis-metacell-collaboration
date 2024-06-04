import os

import numpy as np
import xarray as xr
from minian.initialization import local_max_roll, max_proj_frame
from minian.utilities import save_minian


def seeds_init(
    varr: xr.DataArray,
    wnd_size=500,
    method="rolling",
    stp_size=200,
    nchunk=100,
    max_wnd=10,
    diff_thres=2,
):
    """
    Generate over-complete set of seeds by finding local maxima across frames.

    This function computes the maximum intensity projection of a subset of
    frames and finds the local maxima. The subsetting use either a rolling
    window or random sampling of frames. `wnd_size` `stp_size` and `nchunk`
    controls different aspects of the subsetting. `max_wnd` and `diff_thres`
    controls how local maxima are computed. The set of all local maxima found in
    this process constitutes  an overly-complete set of seeds, representing
    putative locations of cells.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "frame", "height" and "width".
    wnd_size : int, optional
        Number of frames in each chunk, for which a max projection will be
        calculated. By default `500`.
    method : str, optional
        Either `"rolling"` or `"random"`. Controls whether to use rolling window
        or random sampling of frames to construct chunks. By default
        `"rolling"`.
    stp_size : int, optional
        Number of frames between the center of each chunk when stepping through
        the data with rolling windows. Only used if `method is "rolling"`. By
        default `200`.
    nchunk : int, optional
        Number of chunks to sample randomly. Only used if `method is "random"`.
        By default `100`.
    max_wnd : int, optional
        Radius (in pixels) of the disk window used for computing local maxima.
        Local maximas are defined as pixels with maximum intensity in such a
        window. By default `10`.
    diff_thres : int, optional
        Intensity threshold for the difference between local maxima and its
        neighbours. Any local maxima that is not birghter than its neighbor
        (defined by the same disk window) by `diff_thres` intensity values will
        be filtered out. By default `2`.

    Returns
    -------
    seeds : pd.DataFrame
        Seeds dataframe with each seed as a row. Has column "height" and "width"
        which are location of the seeds. Also has column "seeds" which is an
        integer showing how many chunks where the seed is considered a local
        maxima.
    """
    int_path = os.environ["MINIAN_INTERMEDIATE"]
    print("constructing chunks")
    idx_fm = varr.coords["frame"]
    nfm = len(idx_fm)
    if method == "rolling":
        nstp = np.ceil(nfm / stp_size) + 1
        centers = np.linspace(0, nfm - 1, int(nstp))
        hwnd = np.ceil(wnd_size / 2)
        max_idx = list(
            map(
                lambda c: slice(
                    int(np.floor(c - hwnd).clip(0)), int(np.ceil(c + hwnd))
                ),
                centers,
            )
        )
    elif method == "random":
        max_idx = [np.random.randint(0, nfm - 1, wnd_size) for _ in range(nchunk)]
    print("computing max projections")
    res = [max_proj_frame(varr, cur_idx) for cur_idx in max_idx]
    max_res = xr.concat(res, "sample")
    max_res = save_minian(max_res.rename("max_res"), int_path, overwrite=True)
    print("calculating local maximum")
    loc_max = xr.apply_ufunc(
        local_max_roll,
        max_res,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint8],
        kwargs=dict(k0=max_wnd[0], k1=max_wnd[1], diff=diff_thres),
    ).sum("sample")
    seeds = (
        loc_max.where(loc_max > 0).rename("seeds").to_dataframe().dropna().reset_index()
    )
    return seeds[["height", "width", "seeds"]]
