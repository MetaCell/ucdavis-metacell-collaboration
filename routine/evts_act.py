import numpy as np
import pandas as pd

from .utilities import xrconcat_recursive


def construct_periods(evts):
    evts = evts[evts["evt"].isin(["tone", "shock"])].reset_index(drop=True)
    prds = []
    for idx in np.arange(0, len(evts), 2):
        trial = int(idx / 2)
        tone = evts.loc[idx]
        shk = evts.loc[idx + 1]
        prds.append(
            pd.DataFrame(
                [
                    {
                        "evt": "baseline-{}".format(i),
                        "trial": trial,
                        "start": tone["ts"] - (6 - i) * 20,
                        "end": tone["ts"] - (5 - i) * 20,
                    }
                    for i in range(6)
                ]
            )
        )
        prds.append(
            pd.DataFrame(
                [
                    {
                        "evt": "tone",
                        "trial": trial,
                        "start": tone["ts"],
                        "end": tone["ts"] + tone["duration"],
                    },
                    {
                        "evt": "trace",
                        "trial": trial,
                        "start": tone["ts"] + tone["duration"],
                        "end": shk["ts"],
                    },
                    {
                        "evt": "shock",
                        "trial": trial,
                        "start": shk["ts"],
                        "end": shk["ts"] + shk["duration"],
                    },
                ]
            )
        )
    return pd.concat(prds, ignore_index=True)


def extract_acts(var, prds):
    var = var.set_index({"frame": "ts"})
    out_arr = []
    for _, prd in prds.iterrows():
        varsub = (
            var.sel(frame=slice(prd["start"], prd["end"]))
            .assign_coords(evt=prd["evt"], trial=prd["trial"])
            .expand_dims(["evt", "trial"])
        )
        varsub = varsub.assign_coords(frame=np.arange(varsub.sizes["frame"]))
        out_arr.append(varsub)
    return xrconcat_recursive(out_arr, ["trial", "evt"])
