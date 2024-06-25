"""
script to run extract activities around events

env: environments/generic.yml
"""

# %% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from routine.evts_act import construct_periods, extract_acts
from routine.io import load_datasets, parse_gpio

OUT_PATH = "./intermediate/events_act/"
FIG_FOLDER = "./figs/events_act/"
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


# %% extract events
for (anm, ss), (gpio, ts, ps_ds) in load_datasets():
    evts = parse_gpio(gpio)
    prds = construct_periods(evts)
    act_ds = []
    for varname in ["C", "S"]:
        act_ds.append(extract_acts(ps_ds[varname], prds))
    act_ds = xr.merge(act_ds)
    act_ds.to_netcdf(os.path.join(OUT_PATH, "{}-{}.nc".format(anm, ss)))

# %% plotting
plt_ncell = 8
for dsfile in os.listdir(OUT_PATH):
    act_ds = xr.open_dataset(os.path.join(OUT_PATH, dsfile))
    anm, ss = os.path.splitext(dsfile)[0].split("-")
    fig_path = os.path.join(FIG_FOLDER, "{}-{}".format(anm, ss))
    os.makedirs(fig_path, exist_ok=True)
    uid_segs = np.array_split(
        act_ds.coords["unit_id"], int(act_ds.sizes["unit_id"] / plt_ncell)
    )
    for varname, iuid in itt.product(act_ds.var(), range(len(uid_segs))):
        uids = uid_segs[iuid]
        dat = (
            act_ds[varname]
            .sel(unit_id=uids)
            .squeeze()
            .to_dataframe()
            .reset_index()
            .dropna()
        )
        g = sns.FacetGrid(
            dat,
            row="unit_id",
            col="evt",
            hue="evt",
            sharey="row",
            sharex="col",
            margin_titles=True,
            height=1.5,
            aspect=1.2,
        )
        g.map_dataframe(sns.lineplot, x="frame", y=varname, errorbar="se")
        g.figure.savefig(
            os.path.join(fig_path, "{}-{}.svg".format(varname, iuid)),
            dpi=500,
            bbox_inches="tight",
        )
        plt.close(g.figure)
