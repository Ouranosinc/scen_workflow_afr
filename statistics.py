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
import logging
import numpy as np
import os
import pandas as pd
import utils
import xarray as xr


def calc_stat(data_type, freq_in, freq_out, stn, var_or_idx, rcp, hor, stat, q=-1):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate quantiles.

    Parameters
    ----------
    data_type : str
        Dataset type: {cfg.obs, cfg.cat_sim}
    freq_in : str
        Frequency (input): cfg.freq_D=daily; cfg.freq_YS=annual
    freq_out : str, optional
        Frequency (output): cfg.freq_D=daily; cfg.freq_YS=annual
    stn : str
        Station.
    var_or_idx : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).
    rcp : str
        Emission scenario: {cfg.rcp_26, cfg.rcp_45, cfg_rcp_85, "*"}
    hor : [int]
        Horizon: ex: [1981, 2010]
        If None is specified, the complete time range is considered.
    stat : str
        Statistic: {cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_quantil"}
    q : float, optional
        Quantile: value between 0 and 1.

    Returns
    -------
    ds_q : xr.DataSet
        Dataset containing quantiles.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List files.
    if data_type == cfg.cat_obs:
        p_sim_list = [cfg.get_p_obs(stn, var_or_idx)]
    else:
        if var_or_idx in cfg.variables_cordex:
            d = cfg.get_d_sim(stn, cfg.cat_qqmap, var_or_idx)
        else:
            d = cfg.get_d_sim(stn, cfg.cat_idx, var_or_idx)
        p_sim_list = glob.glob(d + "*_" + rcp + ".nc")

    # Exit if there is not file corresponding to the criteria.
    if (len(p_sim_list) == 0) or \
       ((len(p_sim_list) > 0) and not(os.path.isdir(os.path.dirname(p_sim_list[0])))):
        return None

    # List days.
    ds = utils.open_netcdf(p_sim_list[0])
    lon = ds[cfg.dim_lon]
    lat = ds[cfg.dim_lat]
    if cfg.attrs_units in ds[var_or_idx].attrs:
        units = ds[var_or_idx].attrs[cfg.attrs_units]
    else:
        units = 1
    n_sim = len(p_sim_list)

    # First and last years.
    year_1 = int(str(ds.time.values[0])[0:4])
    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
    if not(hor is None):
        year_1 = max(year_1, hor[0])
        year_n = min(year_n, hor[1])
    n_time = (year_n - year_1 + 1) * (365 if freq_in == cfg.freq_D else 1)

    # Collect values from simulations.
    arr_vals = []
    for i_sim in range(n_sim):

        # Load dataset.
        ds = utils.open_netcdf(p_sim_list[i_sim])
        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
        ds = ds.sel(time=slice(years_str[0], years_str[1]))

        # Records values.
        # Simulation data is assumed to be complete.
        if ds[var_or_idx].size == n_time:
            if cfg.dim_lon in str(ds.dims):
                vals = ds.squeeze([cfg.dim_lat, cfg.dim_lon])[var_or_idx].values.tolist()
            else:
                vals = ds[var_or_idx].values.tolist()
            arr_vals.append(vals)
        # Observation data can be incomplete. This explains the day-by-day copy performed below. There is probably a
        # nicer and more efficient way to do this.
        else:
            vals = np.empty(n_time)
            vals[:] = np.nan
            vals = vals.tolist()
            for i_year in range(year_1, year_n + 1):
                for i_month in range(1, 13):
                    for i_day in range(1, 32):
                        date_str = str(i_year) + "-" + str(i_month).zfill(2) + "-" + str(i_day).zfill(2)
                        try:
                            ds_i = ds.sel(time=slice(date_str, date_str))
                            if ds_i[var_or_idx].size != 0:
                                day_of_year = ds_i[var_or_idx].time.dt.dayofyear.values[0]
                                val = ds_i[var_or_idx].values[0][0][0]
                                vals[(i_year - year_1) * 365 + day_of_year - 1] = val
                        except Exception as e:
                            logging.exception(e)
            if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                vals = [i * cfg.spd for i in vals]
            arr_vals.append(vals)

    # Transpose.
    arr_vals_t = []

    # Collapse to yearly frequency.
    if (freq_in == cfg.freq_D) and (freq_out == cfg.freq_YS):
        for i_sim in range(n_sim):
            vals_sim = []
            for i_year in range(0, year_n - year_1 + 1):
                vals_year = arr_vals[i_sim][(365 * i_year):(365 * (i_year + 1))]
                if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                    val_year = np.nansum(vals_year)
                else:
                    val_year = np.nanmean(vals_year)
                vals_sim.append(val_year)
            arr_vals_t.append(vals_sim)
        arr_vals = arr_vals_t
        n_time = year_n - year_1 + 1

    # Transpose.
    arr_vals_t = []
    for i_time in range(n_time):
        vals = []
        for i_sim in range(n_sim):
            vals.append(arr_vals[i_sim][i_time])
        arr_vals_t.append(vals)

    # Calculate statistics.
    arr_stat = []
    for i_time in range(n_time):

        val_stat = None
        if (stat == cfg.stat_min) or (q == 0):
            val_stat = np.min(arr_vals_t[i_time])
        elif (stat == cfg.stat_max) or (q == 1):
            val_stat = np.max(arr_vals_t[i_time])
        elif stat == cfg.stat_mean:
            val_stat = np.mean(arr_vals_t[i_time])
        elif stat == cfg.stat_quantile:
            val_stat = np.quantile(arr_vals_t[i_time], q)
        elif stat == cfg.stat_sum:
            val_stat = np.sum(arr_vals_t[i_time])
        arr_stat.append(val_stat)

    # Build dataset.
    da_stat = xr.DataArray(np.array(arr_stat), name=var_or_idx, coords=[(cfg.dim_time, np.arange(n_time))])
    ds_stat = da_stat.to_dataset()
    ds_stat = ds_stat.expand_dims(lon=1, lat=1)

    # Adjust coordinates and time.
    if not(cfg.dim_lon in ds_stat.dims):
        ds_stat[cfg.dim_lon] = lon
        ds_stat[cfg.dim_lat] = lat
    ds_stat[cfg.dim_time] = utils.reset_calendar(ds_stat, year_1, year_n, freq_out)

    # Adjust units.
    ds_stat[var_or_idx].attrs[cfg.attrs_units] = units

    return ds_stat


def calc_stats(cat):

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
    freq = cfg.freq_D if cat == cfg.cat_scen else cfg.freq_YS

    # Scenarios.
    utils.log("=")
    if cat == cfg.cat_scen:
        utils.log("Step #7a  Calculation of statistics for climate scenarios.")
    else:
        utils.log("Step #7b  Calculation of statistics for climate indices.")

    # Loop through stations.
    for stn in cfg.stns:

        # Loop through variables (or indices).
        vars_or_idxs = cfg.variables_cordex if cat == cfg.cat_scen else cfg.idx_names
        for var_or_idx in vars_or_idxs:

            # Containers.
            stn_list        = []
            var_or_idx_list = []
            rcp_list        = []
            hor_list        = []
            stat_list       = []
            q_list          = []
            val_list        = []

            msg = "Processing: station = " + stn + "; " + ("variable" if cat  == cfg.cat_scen else "index") + \
                  " = " + var_or_idx
            utils.log(msg, True)

            # Loop through emission scenarios.
            for rcp in rcps:

                # Select years.
                if rcp == cfg.rcp_ref:
                    hors = [cfg.per_ref]
                    cat_rcp = cfg.cat_obs
                    d = os.path.dirname(cfg.get_p_obs(stn, var_or_idx))
                else:
                    hors = cfg.per_hors
                    if cat == cfg.cat_scen:
                        cat_rcp = cfg.cat_scen
                        d = cfg.get_d_sim(stn, cfg.cat_qqmap, var_or_idx)
                    else:
                        cat_rcp = cfg.cat_idx
                        d = cfg.get_d_sim(stn, cfg.cat_idx, var_or_idx)

                if not os.path.isdir(d):
                    utils.log("This combination does not exist.", True)
                    continue

                # Loop through statistics.
                stats = [cfg.stat_mean]
                if cat_rcp != cfg.cat_obs:
                    stats = stats + [cfg.stat_min, cfg.stat_max, cfg.stat_quantile]
                for stat in stats:
                    stat_quantiles = cfg.stat_quantiles
                    if stat != cfg.stat_quantile:
                        stat_quantiles = [-1]
                    for q in stat_quantiles:
                        if ((q <= 0) or (q >= 1)) and (q != -1):
                            continue

                        # Calculate statistics.
                        hor = [min(min(hors)), max(max(hors))]
                        ds_stat = calc_stat(cat_rcp, freq, cfg.freq_YS, stn, var_or_idx, rcp, hor, stat, q)
                        if ds_stat is None:
                            continue

                        # Loop through horizons.
                        for hor in hors:

                            # Extract value.
                            year_1 = max(hor[0], int(str(ds_stat.time.values[0])[0:4]))
                            year_n = min(hor[1], int(str(ds_stat.time.values[len(ds_stat.time.values) - 1])[0:4]))
                            years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
                            if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                                val = float(ds_stat.sel(time=slice(years_str[0], years_str[1]))[var_or_idx].sum()) /\
                                      (year_n - year_1 + 1)
                            else:
                                val = float(ds_stat.sel(time=slice(years_str[0], years_str[1]))[var_or_idx].mean())

                            # Convert units.
                            if (cat == cfg.cat_scen) and (rcp != cfg.rcp_ref) and \
                               (var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]):
                                val = val - cfg.d_KC
                            elif var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                                val = val * cfg.spd

                            # Add row.
                            stn_list.append(stn)
                            var_or_idx_list.append(var_or_idx)
                            rcp_list.append(rcp)
                            hor_list.append(str(hor[0]) + "-" + str(hor[1]))
                            if cat_rcp == cfg.cat_obs:
                                stat = "none"
                            stat_list.append(stat)
                            q_list.append(str(q))
                            val_list.append(round(val, 6))

            # Save results.
            if len(stn_list) > 0:

                # Build pandas dataframe.
                dict = {"stn": stn_list, ("var" if cat == cfg.cat_scen else "idx"): var_or_idx_list, "rcp": rcp_list,
                        "hor": hor_list, "stat": stat_list, "q": q_list, "val": val_list}
                df = pd.DataFrame(dict)

                # Save file.
                fn = var_or_idx + "_" + stn + ".csv"
                d  = cfg.get_d_sim(stn, cfg.cat_stat, var_or_idx)
                if not(os.path.isdir(d)):
                    os.makedirs(d)
                p = d + fn
                df.to_csv(p)
                if os.path.exists(p):
                    utils.log("Statistics file created/updated: " + fn, True)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    msg = "Step #7   Calculation of statistics is "
    if cfg.opt_stat:

        msg = msg + "running"
        utils.log(msg)

        # Scenarios.
        calc_stats(cfg.cat_scen)

        # Indices.
        calc_stats(cfg.cat_idx)

    else:

        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)
