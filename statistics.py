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


def calc_stat(data_type, freq, stn, var, rcp, hor, stat, q=-1):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate quantiles.

    Parameters
    ----------
    data_type : str
        Dataset type: {cfg.obs, cfg.cat_sim}
    freq : str
        Frequency: cfg.freq_D=daily; cfg.freq_YS=annual
    stn : str
        Station.
    var : str
        Variable.
    rcp : str
        Emission scenario: {cfg.rcp_26, cfg.rcp_45, cfg_rcp_85, "*"}
    hor : [int]
        Horizon: ex: [1981, 2010]
        If None is specified, the complete time range is considered.
    stat : str
        Statistic: {cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_quantil"}
    q : float
        Quantile: value between 0 and 1.

    Returns
    -------
    ds_q : xr.DataSet
        Dataset containing quantiles.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List files.
    if data_type == cfg.cat_obs:
        p_sim_list = [cfg.get_p_obs(stn, var)]
    else:
        if var in cfg.variables_cordex:
            d = cfg.get_d_sim(stn, cfg.cat_qqmap, var)
        else:
            d = cfg.get_d_sim(stn, cfg.cat_idx, var)
        p_sim_list = glob.glob(d + "*_" + rcp + ".nc")

    # Exit if there is not file corresponding to the criteria.
    if (len(p_sim_list) == 0) or \
       ((len(p_sim_list) > 0) and not(os.path.isdir(os.path.dirname(p_sim_list[0])))):
        return None

    # List days.
    ds = xr.open_dataset(p_sim_list[0])
    lon = ds["lon"]
    lat = ds["lat"]
    n_sim = len(p_sim_list)

    # First and last years.
    year_1 = int(str(ds.time.values[0])[0:4])
    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
    if not(hor is None):
        year_1 = max(year_1, hor[0])
        year_n = min(year_n, hor[1])
    if freq == cfg.freq_D:
        mult = 365
    elif freq == cfg.freq_YS:
        mult = 1
    n_time = (year_n - year_1 + 1) * mult

    # Collect values from simulations.
    arr_vals = []
    for i_sim in range(n_sim):

        # Load dataset.
        ds = xr.open_dataset(p_sim_list[i_sim])
        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
        ds = ds.sel(time=slice(years_str[0], years_str[1]))

        # Records values.
        if "lon" in str(ds.dims):
            arr_vals.append(ds.squeeze(["lat", "lon"])[var].values)
        else:
            arr_vals.append(ds[var].values)
        if n_sim == 1:
            arr_vals = [arr_vals]

    # Calculate statistics.
    arr_vals = np.transpose(arr_vals)
    arr_stat = []
    for i_time in range(n_time):

        if (stat == cfg.stat_min) or (q == 0):
            val_stat = np.min(arr_vals[i_time])
        elif (stat == cfg.stat_max) or (q == 1):
            val_stat = np.max(arr_vals[i_time])
        elif stat == cfg.stat_mean:
            val_stat = np.mean(arr_vals[i_time])
        elif stat == cfg.stat_quantile:
            val_stat = np.quantile(arr_vals[i_time], q)
        elif stat == cfg.stat_sum:
            val_stat = np.sum()
        arr_stat.append(val_stat)

    # Build dataset.
    da_stat = xr.DataArray(np.array(arr_stat), name=var, coords=[("time", np.arange(n_time))])
    ds_stat = da_stat.to_dataset()
    ds_stat = ds_stat.expand_dims(lon=1, lat=1)
    if not("lon" in ds_stat.dims):
        ds_stat["lon"] = lon
        ds_stat["lat"] = lat
    ds_stat["time"] = utils.reset_calendar(ds_stat, year_1, year_n, freq)

    return ds_stat


def run(cat):

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.

    Parameters
    ----------
    cat : str
        Category: cfg.cat_scen is for climate scenarios or cfg.cat_idx for climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List emission scenarios.
    rcps = [cfg.rcp_ref] + cfg.rcps

    # Data frequency.
    if cat == cfg.cat_scen:
        freq = cfg.freq_D
    elif  cat == cfg.cat_idx:
        freq = cfg.freq_YS

    # Scenarios.
    utils.log("=")
    if cat == cfg.cat_scen:
        utils.log("Step #7a  Calculation of statistics for climate scenarios.")
    else:
        utils.log("Step #7b  Calculation of statistics for climate indices.")

    # Loop through stations.
    for stn in cfg.stns:

        # Loop through variables (or indices).
        if cat == cfg.cat_scen:
            vars = cfg.variables_cordex
        else:
            vars = cfg.idx_names
        for var in vars:

            # Containers.
            stn_list  = []
            var_list  = []
            rcp_list  = []
            hor_list  = []
            stat_list = []
            q_list    = []
            val_list  = []

            msg = "Processing: station = " + stn + "; variable = " + var
            utils.log(msg, True)

            # Loop through emission scenarios.
            for rcp in rcps:

                # Select years.
                if rcp == cfg.rcp_ref:
                    hors = [cfg.per_ref]
                    data_type = cfg.cat_obs
                    d = os.path.dirname(cfg.get_p_obs(stn, var))
                else:
                    hors = cfg.per_hors
                    if cat == cfg.cat_scen:
                        data_type = cfg.cat_scen
                        d = cfg.get_d_sim(stn, cfg.cat_qqmap, var)
                    elif cat == cfg.cat_idx:
                        data_type = cfg.cat_idx
                        d = cfg.get_d_sim(stn, cfg.cat_idx, var)

                if not(os.path.isdir(d)):
                    continue

                # Loop through statistics.
                if data_type == cfg.cat_obs:
                    stats = [cfg.stat_mean]
                else:
                    stats = [cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_quantile]
                for stat in stats:
                    stat_quantiles = cfg.stat_quantiles
                    if stat != cfg.stat_quantile:
                        stat_quantiles = [-1]
                    for q in stat_quantiles:
                        if ((q <= 0) or (q >= 1)) and (q != -1):
                            continue

                        # Calculate statistics.
                        hor = [min(min(hors)), max(max(hors))]
                        ds_stat = calc_stat(data_type, freq, stn, var, rcp, hor, stat, q)
                        if ds_stat is None:
                            continue

                        # Loop through horizons.
                        for hor in hors:

                            # Extract value.
                            year_1 = max(hor[0], int(str(ds_stat.time.values[0])[0:4]))
                            year_n = min(hor[1], int(str(ds_stat.time.values[len(ds_stat.time.values) - 1])[0:4]))
                            years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
                            val = float(ds_stat.sel(time=slice(years_str[0], years_str[1]))[var].mean())
                            if (cat == cfg.cat_scen) and (rcp != cfg.rcp_ref) and (var != cfg.var_cordex_pr):
                                val = val - 273.15

                            # Add row.
                            stn_list.append(stn)
                            var_list.append(var)
                            rcp_list.append(rcp)
                            hor_list.append(str(hor[0]) + "-" + str(hor[1]))
                            if data_type == cfg.cat_obs:
                                stat = "none"
                            stat_list.append(stat)
                            q_list.append(str(q))
                            val_list.append(round(val, 6))

            # Save results.
            if len(stn_list) > 0:

                # Build pandas dataframe.
                dict = {"stn": stn_list, "var": var_list, "rcp": rcp_list, "hor": hor_list, "stat": stat_list,
                        "q": q_list, "val": val_list}
                df = pd.DataFrame(dict)

                # Save file.
                fn = var + "_" + stn + ".csv"
                d  = cfg.get_d_sim(stn, cfg.cat_stat, var)
                if not(os.path.isdir(d)):
                    os.makedirs(d)
                p = d + fn
                df.to_csv(p)
                if os.path.exists(p):
                    utils.log("Statistics file created/updated: " + fn, True)
