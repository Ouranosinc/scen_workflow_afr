# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Statistics functions.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import functools
import glob
import multiprocessing
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
    if (cfg.dim_lon in ds.variables) or (cfg.dim_lon in ds.dims):
        lon = ds[cfg.dim_lon]
        lat = ds[cfg.dim_lat]
    elif (cfg.dim_rlon in ds.variables) or (cfg.dim_rlon in ds.dims):
        lon = ds[cfg.dim_rlon]
        lat = ds[cfg.dim_rlat]
    else:
        lon = ds[cfg.dim_longitude]
        lat = ds[cfg.dim_latitude]
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

        # Select specific location.
        if cfg.opt_ra:
            ds = utils.subset_center(ds)

        # Records values.
        # Simulation data is assumed to be complete.
        if ds[var_or_idx].size == n_time:
            if (cfg.dim_lon in ds.dims) or (cfg.dim_rlon in ds.dims) or (cfg.dim_longitude in ds.dims):
                vals = ds.squeeze()[var_or_idx].values.tolist()
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
                        except:
                            pass
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
    utils.log("-")
    if cat == cfg.cat_scen:
        utils.log("Step #7a  Calculation of statistics for climate scenarios.")
    else:
        utils.log("Step #7b  Calculation of statistics for climate indices.")

    # Loop through stations.
    stns = (cfg.stns if not cfg.opt_ra else [cfg.obs_src])
    for stn in stns:

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
                    continue

                utils.log("Processing: '" + stn + "', '" + var_or_idx + "', '" + rcp + "'", True)

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
                dict_pd = {"stn": stn_list, ("var" if cat == cfg.cat_scen else "idx"): var_or_idx_list, "rcp": rcp_list,
                           "hor": hor_list, "stat": stat_list, "q": q_list, "val": val_list}
                df = pd.DataFrame(dict_pd)

                # Save file.
                fn = var_or_idx + "_" + stn + ".csv"
                p  = cfg.get_d_sim(stn, cfg.cat_stat, var_or_idx) + fn
                utils.save_csv(df, p)


def conv_nc_csv():

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert NetCDF to CSV files.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Loop through categories.
        cat_list = [cfg.cat_obs, cfg.cat_raw, cfg.cat_regrid, cfg.cat_qmf, cfg.cat_qqmap, cfg.cat_idx]
        for cat in cat_list:

            # Loop through variables or indices.
            var_or_idx_list = cfg.variables_cordex if cat != cfg.cat_idx else cfg.idx_names
            for var_or_idx in var_or_idx_list:

                # List NetCDF files.
                p_list = list(glob.glob(cfg.get_d_sim(stn, cat, var_or_idx) + "*.nc"))
                n_files = len(p_list)
                if n_files == 0:
                    continue
                p_list.sort()

                utils.log("Processing: '" + stn + "', '" + cat + "', '" + var_or_idx + "'", True)

                # Scalar processing mode.
                if cfg.n_proc == 1:
                    for i_file in range(n_files):
                        conv_nc_csv_single(p_list, var_or_idx, i_file)

                # Parallel processing mode.
                else:

                    # Loop until all files have been converted.
                    while True:

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before = len(list(glob.glob(cfg.get_d_sim(stn, cat, var_or_idx) + "*.csv")))

                        try:
                            utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=cfg.n_proc)
                            func = functools.partial(conv_nc_csv_single, p_list, var_or_idx)
                            pool.map(func, list(range(n_files)))
                            pool.close()
                            pool.join()
                            utils.log("Fork ended.", True)
                        except Exception as e:
                            utils.log(str(e))
                            pass

                        # Calculate the number of files processed (after conversion).
                        n_files_proc_after = len(list(glob.glob(cfg.get_d_sim(stn, cat, var_or_idx) + "*.csv")))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_files_proc_before == n_files_proc_after):
                            break


def conv_nc_csv_single(p_list, var_or_idx, i_file):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a single NetCDF to CSV file.

    Parameters
    ----------
    p_list : [str]
        List of paths.
    var_or_idx : str
        Variable or index.
    i_file : int
        Rank of file in 'p_list'.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Paths.
    p = p_list[i_file]
    p_csv = p.replace("/" + var_or_idx + "/", "/" + var_or_idx + "_csv/").replace(".nc", ".csv")
    if os.path.exists(p_csv):
        return()

    # Open dataset.
    ds = xr.open_dataset(p)

    # Extract time.
    time_list = list(ds.time.values)
    n_time = len(time_list)
    for i in range(n_time):
        if not var_or_idx in cfg.idx_names:
            time_list[i] = str(time_list[i])[0:10]
        else:
            time_list[i] = str(time_list[i])[0:4]

    # Extract longitude and latitude.
    # Calculate average values (only if the analysis is based on observations at a station).
    lon_list = None
    lat_list = None
    if cfg.opt_ra:
        lon_list = ds.lon.values
        lat_list = ds.lat.values

    # Extract values.
    # Calculate average values (only if the analysis is based on observations at a station).
    val_list = list(ds[var_or_idx].values)
    if not cfg.opt_ra:
        if var_or_idx not in cfg.idx_names:
            for i in range(n_time):
                val_list[i] = val_list[i].mean()
        else:
            val_list = list(val_list[0][0])

    # Convert values to more practical units (if required).
    if (var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]) and\
       (ds[var_or_idx].attrs[cfg.attrs_units] == "kg m-2 s-1"):
        for i in range(n_time):
            val_list[i] = val_list[i] * cfg.spd
    elif (var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
         (ds[var_or_idx].attrs[cfg.attrs_units] == "K"):
        for i in range(n_time):
            val_list[i] = val_list[i] - cfg.d_KC

    # Build pandas dataframe.
    if not cfg.opt_ra:
        dict_pd = {cfg.dim_time: time_list, var_or_idx: val_list}
    else:
        dict_pd = {cfg.dim_time: time_list, cfg.dim_lon: lon_list, cfg.dim_lat: lat_list, var_or_idx: val_list}
    df = pd.DataFrame(dict_pd)

    # Save CSV file.
    utils.save_csv(df, p_csv)


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
        # TODO.remove.comment: calc_stats(cfg.cat_scen)

        # Indices.
        calc_stats(cfg.cat_idx)

    else:

        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)

    utils.log("-")
    msg = "Step #7c  Conversion of NetCDF to CSV files is "
    if cfg.opt_conv_nc_csv:

        msg = msg + "running"
        utils.log(msg)
        conv_nc_csv()

    else:

        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)