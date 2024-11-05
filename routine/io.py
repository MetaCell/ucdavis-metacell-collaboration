import os
import warnings

import pandas as pd
import xarray as xr
from scipy.ndimage import label

SSMAP = "./sessions.csv"
DPATH = "./data"
PS_PATH = "./intermediate/processed"
IDX = ["animal", "session"]
DUR_MAP = {(1, 3): "shock", (4, 6): "start/end", (19, 21): "tone"}


def load_datasets(
    dpath=DPATH, ssmap=SSMAP, processed_path=PS_PATH, idx=IDX, meta_only=False
):
    ssmap = pd.read_csv(ssmap).set_index(idx)
    for idxs, ss_dat in ssmap.iterrows():
        if meta_only:
            yield idxs
        dat_path = ss_dat["data"]
        try:
            gpio = pd.read_csv(os.path.join(dpath, dat_path, "gpio.csv")).rename(
                columns={
                    "Time (s)": "ts",
                    " Channel Name": "channel",
                    " Value": "value",
                }
            )
        except FileNotFoundError:
            warnings.warn("Missing GPIO file. Skipping {}".format(dat_path))
            continue
        try:
            ts = (
                pd.read_csv(
                    os.path.join(dpath, dat_path, "traces.csv"), header=1, usecols=[0]
                )
                .squeeze()
                .rename("ts")
            )
        except FileNotFoundError:
            warnings.warn("Missing Traces file. Skipping {}".format(dat_path))
            continue
        if processed_path is not None:
            try:
                ps_ds = xr.open_dataset(
                    os.path.join(processed_path, "-".join(idxs) + ".nc")
                )
            except FileNotFoundError:
                warnings.warn("Missing Processed data. Skipping {}".format(idxs))
                continue
            ps_ds = ps_ds.assign_coords(ts=("frame", ts.values))
        yield idxs, (gpio, ts, ps_ds)


def parse_gpio(gpio, chn_name=" GPIO-1", dig_thres=3e4, dur_map=DUR_MAP):
    ts = (
        gpio[gpio["channel"] == chn_name]
        .sort_values("ts")
        .drop(columns="channel")
        .reset_index()
    )
    ts["dig"] = ts["value"] > dig_thres
    evt_lab, nevt = label(ts["dig"])
    ts["evt_lab"] = evt_lab
    dur_idx = pd.Series(
        dur_map.values(),
        index=pd.IntervalIndex.from_tuples(dur_map.keys(), closed="both"),
    )
    evts = []
    for ievt, evt_df in ts.groupby("evt_lab"):
        if ievt == 0:
            continue
        t_start = evt_df["ts"].min()
        dur = evt_df["ts"].max() - t_start
        evt = dur_idx.loc[dur]
        evts.append(
            pd.DataFrame([{"ievt": ievt, "ts": t_start, "duration": dur, "evt": evt}])
        )
    return pd.concat(evts, ignore_index=True)
