import numpy as np
import xarray as xr
from scipy.signal import correlate, correlation_lags, find_peaks, periodogram


def detect_stripes(x, thresh=15):
    x = x[:, 0]
    f, Pxx_spec = periodogram(x, 1)
    peaks = find_peaks(np.sqrt(Pxx_spec)[: int(len(Pxx_spec) / 5)], height=thresh)[0]
    return not any(f[peaks] > 0.03)


def label_good_frames(varr, **kwargs):
    good_frames = xr.apply_ufunc(
        detect_stripes,
        varr,
        input_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        kwargs=kwargs,
    )
    good_frames = good_frames.compute()
    return good_frames


def determine_buffers(
    frame, ref_frame, exp_buffer_size=8076, int_thres=20, search_range=16
):
    # find anchors
    fm_flt = frame.flatten().astype(float)
    ref_flt = ref_frame.flatten().astype(float)
    fm_size = len(fm_flt)
    fm_diff = np.abs(np.diff(fm_flt))
    diff_fm = np.abs(fm_flt - ref_flt)
    diff_fm_diff = np.diff(diff_fm)
    np.place(diff_fm_diff, (np.arange(len(diff_fm_diff)) % 600) == 0, 0)
    anchors_initial = np.where(diff_fm_diff > int_thres)[0]
    anchors = []
    for ibuf, buf in enumerate(np.diff(anchors_initial)):
        if buf == exp_buffer_size * 2:
            anchors.append(anchors_initial[ibuf])
    anchors = np.unique(anchors)
    # construct initial buffers
    buffers = np.arange(anchors[0], 0, -2 * exp_buffer_size)[::-1]
    for ibuf, buf in enumerate(anchors):
        try:
            end = anchors[ibuf + 1] - search_range
        except IndexError:
            end = fm_size
        buffers = np.append(buffers, np.arange(buf, end, 2 * exp_buffer_size))
    buffers = np.unique(buffers) + 1
    buffers = np.stack([buffers, buffers + exp_buffer_size], axis=-1)
    # refine buffers
    for idxbuf, buf in np.ndenumerate(buffers):
        idx = np.arange(
            max(buf - search_range, 0), min(buf + search_range, fm_size - 2)
        )
        diffs = fm_diff[idx]
        if len(idx) > 0 and ((max(diffs) - min(diffs)) > 0):
            buffers[idxbuf] = idx[np.argmax(diffs)] + 1
    return buffers


def label_buffer(frame, buffers, include=False):
    fm_flt = frame.flatten()
    if include:
        fm_ret = np.zeros_like(fm_flt)
    else:
        fm_ret = fm_flt.copy()
    for ibuf, buf in enumerate(buffers):
        a, b = buf
        if include:
            fm_ret[a:b] = fm_flt[a:b]
        else:
            fm_ret[a:b] = 0
    return fm_ret.reshape(frame.shape)


def fix_frame(
    frame,
    ref_frame,
    buffers,
    same_sh_diff=False,
    prev_buffer=False,
    interpolate=False,
    shifts_initial=None,
):
    period = frame.shape[1]
    fm_flatten = frame.flatten().astype(float)
    fm_ref_flatten = ref_frame.flatten().astype(float)
    shifts = np.zeros(buffers.shape[0], dtype=int)
    pads = np.zeros(buffers.shape[0], dtype=int)
    fm_fix = fm_flatten.copy()
    col_idxs = np.where(np.arange(len(fm_flatten)) % period == 0)[0]
    for ibuf, buf in enumerate(buffers):
        a, b = buf
        fm_fix[a:b] = 0
        cur_ref = fm_ref_flatten[a:b]
        if shifts_initial is not None:
            b = min(a + shifts_initial[ibuf], b)
        cur_buf = fm_flatten[a:b]
        cur_cols = col_idxs[(col_idxs > a) & (col_idxs < b)] - a
        # cur_buf = pad_buffer(cur_buf, cur_cols, 1)
        pad, sh = estimate_pad_shift(cur_buf, cur_ref, cur_cols)
        # if shifts_initial is not None:
        #     sh = sh + shifts_initial[ibuf]
        shifts[ibuf] = sh
        pads[ibuf] = pad
    if same_sh_diff:
        sh_diff = np.diff(shifts)
        d_val, d_counts = np.unique(sh_diff, return_counts=True)
        d = d_val[np.argmax(d_counts)]
        anchor_sh = np.where(sh_diff == d)[0]
        first_sh_idx, last_sh_idx = np.min(anchor_sh), np.max(anchor_sh)
        first_sh, last_sh = shifts[first_sh_idx], shifts[last_sh_idx]
        shifts[0:first_sh_idx] = np.arange(first_sh - d * first_sh_idx, first_sh, d)
        shifts[last_sh_idx:] = np.arange(
            last_sh, last_sh + d * (len(shifts) - last_sh_idx), d
        )
    for ish, (sh, buf, pad) in enumerate(zip(shifts, buffers, pads)):
        a, b = buf
        cur_buf = fm_flatten[a:b]
        cur_cols = col_idxs[(col_idxs > a) & (col_idxs < b)] - a
        cur_cols_new = col_idxs - sh
        cur_cols_new = cur_cols_new[(cur_cols_new > a) & (cur_cols_new < b)] - a
        cur_buf = pad_buffer(
            cur_buf, cur_cols, pad, interpolate=False, delidx=cur_cols_new
        )
        cur_ref = fm_ref_flatten[a:b]
        cur_new, cur_mask = shift_buffer(fm_fix[a:b], cur_buf, sh)
        if (ish > 0) and prev_buffer:
            idx_missing = np.where(~cur_mask)[0]
            c, d = idx_missing.min(), idx_missing.max()
            len_missing = d - c
            if len_missing > 0:
                j, k = buffers[ish - 1]
                assert (k - j) > len_missing
                j = k - len_missing
                prev_buf = fm_flatten[j:k]
                cur_cols = col_idxs[(col_idxs > j) & (col_idxs < k)] - j
                pad_sub, sh_sub = estimate_pad_shift(prev_buf, cur_ref[c:d], cur_cols)
                cur_cols_new = col_idxs - sh_sub
                cur_cols_new = cur_cols_new[(cur_cols_new > j) & (cur_cols_new < k)] - j
                prev_buf = pad_buffer(
                    prev_buf, cur_cols, pad_sub, interpolate=False, delidx=cur_cols_new
                )
                # sh_sub = estimate_buffer_shift(prev_buf, cur_ref[c:d])
                cur_new, mask_new = shift_buffer(cur_new, prev_buf, sh_sub + c)
                cur_mask = np.logical_or(cur_mask, mask_new)
        if interpolate:
            idx_filled = np.where(cur_mask)[0]
            idx_interp = np.where(~cur_mask)[0]
            cur_new[idx_interp] = np.interp(idx_interp, idx_filled, cur_new[idx_filled])
        fm_fix[a:b] = cur_new
    return fm_fix.reshape(frame.shape).astype(frame.dtype), shifts


def estimate_pad_shift(buf, buf_ref, pidx, pad_max=20):
    shifts = np.zeros(pad_max, dtype=int)
    corrs = np.zeros(pad_max)
    for pad in range(pad_max):
        last_sh = None
        niter = 0
        while niter < 50:
            try:
                if last_sh is None:
                    buf_pad = pad_buffer(buf, pidx, pad)
                else:
                    delidx = pidx - last_sh
                    delidx = delidx[(delidx >= 0) & (delidx < len(buf))]
                    buf_pad = pad_buffer(buf, pidx, pad, delidx=delidx)
            except ValueError:
                sh = 0
                cor = 0
                break
            sh, cor = estimate_buffer_shift(buf_pad, buf_ref)
            niter += 1
            if last_sh == sh:
                break
            else:
                last_sh = sh
        shifts[pad] = sh
        corrs[pad] = cor
    pad_max = np.argmax(corrs)
    return pad_max, shifts[pad_max]


def pad_buffer(buffer, pidx, pad, interpolate=True, delidx=None):
    buffer = buffer.copy()
    if not pad > 0:
        return buffer
    if delidx is not None and len(delidx) > 0:
        delidxs = np.concatenate([np.arange(max(d - pad, 0), d) for d in delidx])
        buffer[delidxs] = np.nan
    for ii, idx in enumerate(pidx):
        if delidx is None:
            try:
                nxt = pidx[ii + 1]
            except IndexError:
                nxt = len(buffer)
            cur_buffer = buffer[idx:nxt]
            delidxs = np.argpartition(cur_buffer, pad)[:pad] + idx
            buffer = np.delete(buffer, delidxs)
        else:
            idx = idx + ii * pad
        if interpolate:
            try:
                pad_buf = np.linspace(buffer[idx], buffer[idx + 1], pad)
            except IndexError:
                pad_buf = np.repeat(buffer[idx], pad)
        else:
            pad_buf = np.zeros(pad)
        buffer = np.insert(buffer, idx, pad_buf)
    return buffer[~np.isnan(buffer)]


def estimate_buffer_shift(buf, buf_ref):
    std_ref = np.std(buf_ref)
    std_buf = np.std(buf)
    if std_ref > 0 and std_buf > 0:
        ref_norm = (buf_ref - buf_ref.mean()) / std_ref
        buf_norm = (buf - buf.mean()) / std_buf
        lags = correlation_lags(len(buf), len(buf_ref), mode="full")
        corr = correlate(buf_norm, ref_norm, mode="full", method="fft")
        imax = np.argmax(corr)
        sh = -lags[imax]
        c = corr[imax]
    else:
        sh = 0
        c = 0
    return sh, c


def shift_buffer(buf_org, buf_repl, sh):
    buf_new = buf_org.copy()
    mask = np.zeros_like(buf_org, dtype=bool)
    range_repl = np.arange(max(0, sh), min(len(buf_repl) + sh, len(buf_new)))
    buf_new[range_repl] = buf_repl[range_repl - sh]
    mask[range_repl] = 1
    return buf_new, mask


def notch_filt_fft(x, pos, wnd):
    x_fft = np.fft.rfft(x)
    x_fft[pos - wnd : pos + wnd + 1] = 0
    return np.fft.irfft(x_fft)


def ripple_correction(varr, dim="width", **kwargs):
    return xr.apply_ufunc(
        notch_filt_fft,
        varr,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=varr.dtype,
    )
