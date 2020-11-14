# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Statistics functions.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import functools
import glob
import multiprocessing
import numpy as np
import os
import pandas as pd
import plot
import utils
import xarray as xr
from typing import Union


def calc_stat(data_type: str, freq_in: str, freq_out: str, stn: str, var_or_idx: str, rcp: str, hor: [int], stat: str,
              q: float = -1) -> Union[xr.Dataset, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate quantiles.

    Parameters
    ----------
    data_type : str
        Dataset type: {cfg.obs, cfg.cat_scen}
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
    --------------------------------------------------------------------------------------------------------------------
    """

    # List files.
    if data_type == cfg.cat_obs:
        if var_or_idx in cfg.variables_cordex:
            p_sim_list = [cfg.get_d_scen(stn, cfg.cat_obs, var_or_idx) + var_or_idx + "_" + stn + ".nc"]
        else:
            p_sim_list = [cfg.get_d_idx(stn, var_or_idx) + var_or_idx + "_ref.nc"]
    else:
        if var_or_idx in cfg.variables_cordex:
            d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
        else:
            d = cfg.get_d_scen(stn, cfg.cat_idx, var_or_idx)
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

        # Select years and adjust units.
        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
        ds = ds.sel(time=slice(years_str[0], years_str[1]))
        if cfg.attrs_units in ds[var_or_idx].attrs:
            if (var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
               (ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K):
                    ds = ds - cfg.d_KC
                    ds[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_C
            elif var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                ds = ds * cfg.spd

        # Select control point.
        if cfg.opt_ra:
            if cfg.d_bounds == "":
                ds = utils.subset_ctrl_pt(ds)
            else:
                ds = utils.squeeze_lon_lat(ds)

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
            np_vals = np.empty(n_time)
            np_vals[:] = np.nan
            vals = np_vals.tolist()
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


def calc_stats(cat: str):

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
                    if cat == cfg.cat_scen:
                        d = os.path.dirname(cfg.get_p_obs(stn, var_or_idx))
                    else:
                        d = cfg.get_d_scen(stn, cfg.cat_idx, var_or_idx)
                else:
                    hors = cfg.per_hors
                    if cat == cfg.cat_scen:
                        cat_rcp = cfg.cat_scen
                        d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
                    else:
                        cat_rcp = cfg.cat_idx
                        d = cfg.get_d_scen(stn, cfg.cat_idx, var_or_idx)

                if not os.path.isdir(d):
                    continue

                utils.log("Processing (stats): '" + stn + "', '" + var_or_idx + "', '" + rcp + "'", True)

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
                p  = cfg.get_d_scen(stn, cfg.cat_stat, var_or_idx) + fn
                utils.save_csv(df, p)


def conv_nc_csv(cat: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert NetCDF to CSV files.

    Parameters
    ----------
    cat : str
        Category: cfg.cat_scen is for climate scenarios or cfg.cat_idx for climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Loop through categories.

        cat_list = [cfg.cat_obs, cfg.cat_raw, cfg.cat_regrid, cfg.cat_qqmap]
        if cat == cfg.cat_idx:
            cat_list = [cfg.cat_idx]
        for cat in cat_list:

            # Loop through variables or indices.
            var_or_idx_list = cfg.variables_cordex if cat != cfg.cat_idx else cfg.idx_names
            for var_or_idx in var_or_idx_list:

                # List NetCDF files.
                p_list = list(glob.glob(cfg.get_d_scen(stn, cat, var_or_idx) + "*.nc"))
                n_files = len(p_list)
                if n_files == 0:
                    continue
                p_list.sort()

                utils.log("Processing: '" + stn + "', '" + var_or_idx + "'", True)

                # Scalar processing mode.
                if cfg.n_proc == 1:
                    for i_file in range(n_files):
                        conv_nc_csv_single(p_list, var_or_idx, i_file)

                # Parallel processing mode.
                else:

                    # Loop until all files have been converted.
                    while True:

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before = len(list(glob.glob(cfg.get_d_scen(stn, cat, var_or_idx) + "*.csv")))

                        try:
                            utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=min(cfg.n_proc, len(p_list)))
                            func = functools.partial(conv_nc_csv_single, p_list, var_or_idx)
                            pool.map(func, list(range(n_files)))
                            pool.close()
                            pool.join()
                            utils.log("Fork ended.", True)
                        except Exception as e:
                            utils.log(str(e))
                            pass

                        # Calculate the number of files processed (after conversion).
                        n_files_proc_after = len(list(glob.glob(cfg.get_d_scen(stn, cat, var_or_idx) + "*.csv")))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_files_proc_before == n_files_proc_after):
                            break


def conv_nc_csv_single(p_list: [str], var_or_idx: str, i_file: int):

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
    if os.path.exists(p_csv) and (not cfg.opt_force_overwrite):
        return()

    # Open dataset.
    ds = xr.open_dataset(p)

    # Extract time.
    time_list = list(ds.time.values)
    n_time = len(time_list)
    for i in range(n_time):
        if var_or_idx not in cfg.idx_names:
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
            if cfg.rcp_ref in p:
                for i in range(n_time):
                    val_list[i] = val_list[i][0][0]
            else:
                val_list = list(val_list[0][0])

    # Convert values to more practical units (if required).
    if (var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]) and\
       (ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kgm2s1):
        for i in range(n_time):
            val_list[i] = val_list[i] * cfg.spd
    elif (var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
         (ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K):
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


def calc_ts(cat: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Plot time series for individual simulations.

    Parameters
    ----------
    cat : str
        Category: cfg.cat_scen is for climate scenarios or cfg.cat_idx for climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Emission scenarios.
    rcps = [cfg.rcp_ref] + cfg.rcps

    # Minimum and maximum values along the y-axis
    ylim = []

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Loop through variables.
        vars_or_idxs = cfg.variables_cordex if cat == cfg.cat_scen else cfg.idx_names
        for i_var_or_idx in range(len(vars_or_idxs)):
            var_or_idx = vars_or_idxs[i_var_or_idx]
            threshs = [] if cat == cfg.cat_scen else cfg.idx_threshs

            utils.log("Processing (time series): '" + stn + "', '" + var_or_idx + "'", True)

            p_csv = cfg.get_d_scen(stn, cfg.cat_stat, var_or_idx) + var_or_idx + "_" + stn + "_ts.csv"
            if os.path.exists(p_csv) and (not cfg.opt_force_overwrite):
                continue

            # Loop through emission scenarios.
            ds_ref = None
            ds_rcp_26, ds_rcp_26_grp = [], []
            ds_rcp_45, ds_rcp_45_grp = [], []
            ds_rcp_85, ds_rcp_85_grp = [], []
            for rcp in rcps:

                # List files.
                if rcp == cfg.rcp_ref:
                    if var_or_idx in cfg.variables_cordex:
                        p_sim_list = [cfg.get_d_scen(stn, cfg.cat_obs, var_or_idx) + var_or_idx + "_" + stn + ".nc"]
                    else:
                        p_sim_list = [cfg.get_d_idx(stn, var_or_idx) + var_or_idx + "_ref.nc"]
                else:
                    if var_or_idx in cfg.variables_cordex:
                        d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
                    else:
                        d = cfg.get_d_idx(stn, var_or_idx)
                    p_sim_list = glob.glob(d + "*_" + rcp + ".nc")

                # Exit if there is no file corresponding to the criteria.
                if (len(p_sim_list) == 0) or \
                   ((len(p_sim_list) > 0) and not(os.path.isdir(os.path.dirname(p_sim_list[0])))):
                    continue

                # Loop through simulation files.
                for i_sim in range(len(p_sim_list)):

                    # Load dataset.
                    ds = utils.open_netcdf(p_sim_list[i_sim]).squeeze()

                    # Select control point.
                    if cfg.opt_ra:
                        if cfg.d_bounds == "":
                            ds = utils.subset_ctrl_pt(ds)
                        else:
                            ds = utils.squeeze_lon_lat(ds, var_or_idx)

                    # First and last years.
                    year_1 = int(str(ds.time.values[0])[0:4])
                    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
                    if rcp == cfg.rcp_ref:
                        year_1 = max(year_1, cfg.per_ref[0])
                        year_n = min(year_n, cfg.per_ref[1])
                    else:
                        year_1 = max(year_1, cfg.per_ref[0])
                        year_n = min(year_n, cfg.per_fut[1])

                    # Select years.
                    years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
                    ds = ds.sel(time=slice(years_str[0], years_str[1]))

                    # Remember units.
                    units = ds[var_or_idx].attrs[cfg.attrs_units] if cat == cfg.cat_scen else ds.attrs[cfg.attrs_units]
                    if units == "degree_C":
                        units = cfg.unit_C

                    # Calculate statistics.
                    # TODO: Include coordinates in the generated dataset.
                    years = ds.groupby(ds.time.dt.year).groups.keys()
                    if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                        ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
                    else:
                        ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
                    n_time = len(ds[cfg.dim_time].values)
                    da = xr.DataArray(np.array(ds[var_or_idx].values), name=var_or_idx,
                                      coords=[(cfg.dim_time, np.arange(n_time))])
                    ds = da.to_dataset()
                    ds[cfg.dim_time] = utils.reset_calendar_list(years)
                    ds[var_or_idx].attrs[cfg.attrs_units] = units

                    # Convert units.
                    if var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                        if ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K:
                            ds = ds - cfg.d_KC
                            ds[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_C
                    elif var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                        if ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kgm2s1:
                            ds = ds * cfg.spd
                            ds[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_mm

                    # Calculate minimum and maximum values along the y-axis.
                    if not ylim:
                        ylim = [min(ds[var_or_idx].values), max(ds[var_or_idx].values)]
                    else:
                        ylim = [min(ylim[0], min(ds[var_or_idx].values)),
                                max(ylim[1], max(ds[var_or_idx].values))]

                    # Append to list of datasets.
                    if rcp == cfg.rcp_ref:
                        ds_ref = ds
                    elif rcp == cfg.rcp_26:
                        ds_rcp_26.append(ds)
                    elif rcp == cfg.rcp_45:
                        ds_rcp_45.append(ds)
                    elif rcp == cfg.rcp_85:
                        ds_rcp_85.append(ds)

                # Group by RCP.
                if rcp != cfg.rcp_ref:
                    if rcp == cfg.rcp_26:
                        ds_rcp_26_grp = calc_stat_mean_min_max(ds_rcp_26, var_or_idx)
                    elif rcp == cfg.rcp_45:
                        ds_rcp_45_grp = calc_stat_mean_min_max(ds_rcp_45, var_or_idx)
                    elif rcp == cfg.rcp_85:
                        ds_rcp_85_grp = calc_stat_mean_min_max(ds_rcp_85, var_or_idx)

            if (ds_ref is not None) or (ds_rcp_26 != []) or (ds_rcp_45 != []) or (ds_rcp_85 != []):

                # Generate statistics.
                if cfg.opt_save_csv:

                    # Extract years.
                    years = []
                    if cfg.rcp_26 in rcps:
                        years = utils.extract_years(ds_rcp_26_grp[0])
                    elif cfg.rcp_45 in rcps:
                        years = utils.extract_years(ds_rcp_45_grp[0])
                    elif cfg.rcp_85 in rcps:
                        years = utils.extract_years(ds_rcp_85_grp[0])

                    # Initialize pandas dataframe.
                    dict_pd = {"year": years}
                    df = pd.DataFrame(dict_pd)
                    df[cfg.rcp_ref] = None

                    # Add values.
                    for rcp in rcps:
                        if rcp == cfg.rcp_ref:
                            years = utils.extract_years(ds_ref)
                            vals = ds_ref[var_or_idx].values
                            for i in range(len(vals)):
                                df[cfg.rcp_ref][df["year"] == years[i]] = vals[i]
                        elif rcp == cfg.rcp_26:
                            df[cfg.rcp_26 + "_min"] = ds_rcp_26_grp[0][var_or_idx].values
                            df[cfg.rcp_26 + "_moy"] = ds_rcp_26_grp[1][var_or_idx].values
                            df[cfg.rcp_26 + "_max"] = ds_rcp_26_grp[2][var_or_idx].values
                        elif rcp == cfg.rcp_45:
                            df[cfg.rcp_45 + "_min"] = ds_rcp_45_grp[0][var_or_idx].values
                            df[cfg.rcp_45 + "_moy"] = ds_rcp_45_grp[1][var_or_idx].values
                            df[cfg.rcp_45 + "_max"] = ds_rcp_45_grp[2][var_or_idx].values
                        elif rcp == cfg.rcp_85:
                            df[cfg.rcp_85 + "_min"] = ds_rcp_85_grp[0][var_or_idx].values
                            df[cfg.rcp_85 + "_moy"] = ds_rcp_85_grp[1][var_or_idx].values
                            df[cfg.rcp_85 + "_max"] = ds_rcp_85_grp[2][var_or_idx].values

                    # Save file.
                    utils.save_csv(df, p_csv)

                # Generate plots.
                if cfg.opt_plot:

                    # Time series with simulations grouped by RCP scenario.
                    cat_fig = cfg.cat_fig + "/" + cat + "/" + var_or_idx + "/"
                    p_fig = cfg.get_d_scen(stn, cat_fig, "") + var_or_idx + "_" + stn + "_rcp.png"
                    plot.plot_ts(ds_ref, ds_rcp_26_grp, ds_rcp_45_grp, ds_rcp_85_grp, stn.capitalize(), var_or_idx,
                        threshs, rcps, ylim, p_fig, 1)

                    # Time series showing individual simulations.
                    p_fig = p_fig.replace("_rcp.png", "_sim.png")
                    plot.plot_ts(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(), var_or_idx,
                        threshs, rcps, ylim, p_fig, 2)


def calc_stat_mean_min_max(ds_list: [xr.Dataset], var_or_idx: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate mean, minimum and maximum values within a group of datasets.
    TODO: Include coordinates in the returned datasets.

    Parameters
    ----------
    ds_list : [xr.Dataset]
        Array of datasets from a given group.
    var_or_idx : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).

    Returns
    -------
    ds_mean_min_max : [xr.Dataset]
        Array of datasets with mean, minimum and maximum values.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_mean_min_max = []

    # Get years, units and coordinates.
    units = ds_list[0][var_or_idx].attrs[cfg.attrs_units]
    year_1 = int(str(ds_list[0].time.values[0])[0:4])
    year_n = int(str(ds_list[0].time.values[len(ds_list[0].time.values) - 1])[0:4])
    n_time = year_n - year_1 + 1

    # Calculate statistics.
    arr_vals_mean = []
    arr_vals_min = []
    arr_vals_max = []
    for i_time in range(n_time):
        vals = []
        for ds in ds_list:
            vals.append(float(ds[var_or_idx][i_time].values))
        arr_vals_mean.append(np.array(vals).mean())
        arr_vals_min.append(min(np.array(vals)))
        arr_vals_max.append(max(np.array(vals)))

    # Build datasets.
    for i in range(1, 4):

        # Select values.
        if i == 1:
            arr_vals = arr_vals_mean
        elif i == 2:
            arr_vals = arr_vals_min
        else:
            arr_vals = arr_vals_max

        # Build dataset.
        da = xr.DataArray(np.array(arr_vals), name=var_or_idx, coords=[(cfg.dim_time, np.arange(n_time))])
        ds = da.to_dataset()
        ds[cfg.dim_time] = utils.reset_calendar(ds, year_1, year_n, cfg.freq_YS)
        ds[var_or_idx].attrs[cfg.attrs_units] = units

        ds_mean_min_max.append(ds)

    return ds_mean_min_max
