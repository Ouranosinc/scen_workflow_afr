# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Statistics functions.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import numpy as np
import os
import pandas as pd
import utils
import xarray as xr


def run(mode = 1):

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.

    Parameters
    ----------
    mode : int
        Mode 1 is for climate variables, whereas 2 is for climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List emission scenarios.
    rcps = ["ref"]
    for rcp in cfg.rcps:
        rcps.append(rcp)

    # Scenarios.
    utils.log("=")
    if mode == 1:
        utils.log("Step #7a  Calculation of statistics for climate scenarios.")
    else:
        utils.log("Step #7b  Calculation of statistics for climate indices.")

    # Loop through stations.
    for stn in cfg.stns:

        # Loop through variables (or indices).
        if mode == 1:
            vars = cfg.variables_cordex
        else:
            vars = cfg.idx_names
        for var in vars:

            # Containers.
            stn_list = []
            var_list = []
            rcp_list = []
            hor_list = []
            val_list = []

            # Loop through emission scenarios.
            for rcp in rcps:

                # Select years.
                if rcp == "ref":
                    hors = [cfg.per_ref]
                else:
                    hors = cfg.per_hors

                # Select paths of simulations files.
                if mode == 1:
                    if rcp == "ref":
                        p_list = [cfg.get_p_obs(stn, var)]
                    else:
                        p_list = glob.glob(cfg.get_d_sim(stn, cfg.cat_qqmap, var) + "*_" + rcp + "*.nc")
                else:
                    p_list = [cfg.get_d_sim(stn, cfg.cat_qqmap, var) + var + "_" + rcp + "_mean_4qqmap.nc"]

                # Loop through horizons.
                for hor in hors:

                    # Loop through simulations.
                    val_sum = 0
                    n_sum   = 0
                    for p in p_list:
                        if not(os.path.exists(p)):
                            continue

                        # Load dataset.
                        ds = xr.open_dataset(p)

                        # Determine first and last years.
                        if rcp == "ref":
                            year_1 = int(str(ds.time.values[0])[0:4])
                            year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
                        else:
                            year_1 = hor[0]
                            year_n = hor[1]

                        # Calculate statistics.
                        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
                        ds = ds.sel(time=slice(years_str[0], years_str[1]))
                        if var == cfg.var_cordex_pr:
                            if mode == 1:
                                val = float(ds[var].sum() / float(year_n - year_1 + 1))
                            else:
                                val = float(ds.to_array().values.sum() / float(year_n - year_1 + 1))
                        else:
                            if mode == 1:
                                val = float(ds[var].mean())
                            else:
                                val = ds.to_array().values.mean()
                        if (mode == 1) and (rcp != "ref") and (var != cfg.var_cordex_pr):
                            val = val - 273.15
                        val_sum = val + val_sum
                        n_sum = n_sum + 1

                    # Add row.
                    if n_sum > 0:
                        stn_list.append(stn)
                        var_list.append(var)
                        rcp_list.append(rcp)
                        hor_list.append(str(hor[0]) + "-" + str(hor[1]))
                        val_list.append(round(val_sum / n_sum, 6))

            # Save results.
            if len(stn_list) > 0:

                # Build pandas dataframe.
                dict = {"stn": stn_list, "var": var_list, "rcp": rcp_list, "hor": hor_list, "val": val_list}
                df = pd.DataFrame(dict)

                # Save file.
                fn = var + "_" + stn + ".csv"
                d  = cfg.get_d_sim(stn, cfg.cat_stats, var)
                if not(os.path.isdir(d)):
                    os.makedirs(d)
                p = d + fn
                df.to_csv(p)
                if os.path.exists(p):
                    utils.log("Statistics file created/updated: " + fn, True)
