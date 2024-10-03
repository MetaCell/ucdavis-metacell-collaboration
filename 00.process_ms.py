"""
script to run minian pipeline on datasets

env: environments/generic.yml
"""

from ds_utils.minian_pipeline.process import minian_process_batch

if __name__ == "__main__":
    minian_process_batch(
        ss_csv="./sessions.csv",
        dat_path="./data",
        param_path="./process/params/",
        out_path="./intermediate/processed/",
        fig_path="./figs/processed/",
        err_path="./process/errs",
        int_path="./process/minian_int/",
        param_col="params",
        worker_path="./process/dask-worker-space",
        skip_existing=True,
        raise_err=False,
    )
