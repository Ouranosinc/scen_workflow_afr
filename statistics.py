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
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import plot
import utils
import warnings
import xarray as xr
from pandas.core.common import SettingWithCopyWarning
from scipy.interpolate import griddata
from streamlit import caching
from typing import Union, List


def calc_stat(data_type: str, freq_in: str, freq_out: str, stn: str, var_or_idx_code: str, rcp: str, hor: [int],
              use_bounds: bool, stat: str, q: float = -1) -> Union[xr.Dataset, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate quantiles.

    Parameters
    ----------
    data_type : str
        Dataset type: {cfg.obs, cfg.cat_scen}
    freq_in : str
        Frequency (input): cfg.freq_D=daily; cfg.freq_YS=annual
    freq_out : str
        Frequency (output): cfg.freq_D=daily; cfg.freq_YS=annual
    stn : str
        Station.
    var_or_idx_code : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index code (ex: cfg.idx_txdaysabove).
    rcp : str
        Emission scenario: {cfg.rcp_26, cfg.rcp_45, cfg_rcp_85, cfg_rcp_xx}
    hor : [int]
        Horizon: ex: [1981, 2010]
        If None is specified, the complete time range is considered.
    use_bounds : bool
        If True, use cfg.d_bounds.
    stat : str
        Statistic: {cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_quantile"}
    q : float, optional
        Quantile: value between 0 and 1.
    --------------------------------------------------------------------------------------------------------------------
    """

    cat = cfg.cat_scen if var_or_idx_code in cfg.variables_cordex else cfg.cat_idx
    var_or_idx = var_or_idx_code if cat == cfg.cat_scen else cfg.extract_idx(var_or_idx_code)

    # List files.
    if data_type == cfg.cat_obs:
        if var_or_idx in cfg.variables_cordex:
            p_sim_list = [cfg.get_d_scen(stn, cfg.cat_obs, var_or_idx) + var_or_idx + "_" + stn + cfg.f_ext_nc]
        else:
            p_sim_list = [cfg.get_d_idx(stn, var_or_idx_code) + var_or_idx + "_ref" + cfg.f_ext_nc]
    else:
        if var_or_idx in cfg.variables_cordex:
            d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
        else:
            d = cfg.get_d_idx(stn, var_or_idx_code)
        p_sim_list = glob.glob(d + "*_" + ("*rcp*" if rcp == cfg.rcp_xx else rcp) + cfg.f_ext_nc)

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

        if cfg.opt_ra:

            # Statistics are calculated at a point.
            if cfg.d_bounds == "":
                ds = utils.subset_ctrl_pt(ds)

            # Statistics are calculated on a surface.
            else:

                # Clip to geographic boundaries.
                if use_bounds and (cfg.d_bounds != ""):
                    ds = utils.subset_shape(ds)

                # Calculate mean value.
                ds = utils.squeeze_lon_lat(ds)

        # Records values.
        # Simulation data is assumed to be complete.
        if ds[var_or_idx].time.size == n_time:
            if (cfg.dim_lon in ds.dims) or (cfg.dim_rlon in ds.dims) or (cfg.dim_longitude in ds.dims):
                vals = ds.squeeze()[var_or_idx].values.tolist()
            else:
                vals = ds[var_or_idx].values.tolist()
            arr_vals.append(vals)
        # Observation data can be incomplete. This explains the day-by-day copy performed below. There is probably a
        # nicer and more efficient way to do this.
        # TODO: The algorithm is not efficient in the following lines. Fix this.
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
                da_vals = xr.DataArray(np.array(vals_year))
                vals_year = list(da_vals[np.isnan(da_vals.values) == False].values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                        val_year = np.nansum(vals_year)
                    else:
                        val_year = np.nanmean(vals_year)
                vals_sim.append(val_year)
            if (var_or_idx != cfg.var_cordex_pr) | (sum(vals_sim) > 0):
                arr_vals_t.append(vals_sim)
        arr_vals = arr_vals_t
        n_time = year_n - year_1 + 1
    n_sim = len(arr_vals)

    # Transpose array.
    arr_vals_t = []
    for i_time in range(n_time):
        vals = []
        for i_sim in range(n_sim):
            if (var_or_idx != cfg.idx_rainqty) or ((var_or_idx == cfg.idx_rainqty) and (max(arr_vals[i_sim]) > 0)):
                vals.append(arr_vals[i_sim][i_time])
        arr_vals_t.append(vals)

    # Calculate statistics.
    arr_stat = []
    for i_time in range(n_time):
        val_stat = None
        vals_t = arr_vals_t[i_time]
        if n_sim > 1:
            da_vals = xr.DataArray(np.array(vals_t))
            vals_t = list(da_vals[(np.isnan(da_vals.values) == False) &
                                  ((var_or_idx != cfg.var_cordex_pr) | (sum(da_vals.values) > 0))].values)
        if (stat == cfg.stat_min) or (q == 0):
            val_stat = np.min(vals_t)
        elif (stat == cfg.stat_max) or (q == 1):
            val_stat = np.max(vals_t)
        elif stat == cfg.stat_mean:
            val_stat = np.mean(vals_t)
        elif stat == cfg.stat_quantile:
            val_stat = np.quantile(vals_t, q)
        elif stat == cfg.stat_sum:
            val_stat = np.sum(vals_t)
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
    Calculate statistics.

    Parameters
    ----------
    cat : str
        Category: cfg.cat_scen is for climate scenarios or cfg.cat_idx for climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List emission scenarios.
    rcps = [cfg.rcp_ref] + cfg.rcps
    if len(rcps) > 2:
        rcps = rcps + [cfg.rcp_xx]

    # Data frequency.
    freq = cfg.freq_D if cat == cfg.cat_scen else cfg.freq_YS

    # Loop through stations.
    stns = (cfg.stns if not cfg.opt_ra else [cfg.obs_src])
    for stn in stns:

        # Loop through variables (or indices).
        var_or_idx_list = cfg.variables_cordex if cat == cfg.cat_scen else cfg.idx_names
        for i_var_or_idx in range(len(var_or_idx_list)):
            var_or_idx = var_or_idx_list[i_var_or_idx]
            var_or_idx_code = var_or_idx if cat == cfg.cat_scen else cfg.idx_codes[i_var_or_idx]

            # Skip iteration if the file already exists.
            p_csv =\
                cfg.get_d_scen(stn, cfg.cat_stat, cat + "/" + var_or_idx_code) + var_or_idx + "_" + stn + cfg.f_ext_csv
            if os.path.exists(p_csv) and (not cfg.opt_force_overwrite):
                continue

            # Containers.
            stn_list             = []
            rcp_list             = []
            hor_list             = []
            stat_list            = []
            q_list               = []
            val_list             = []

            # Loop through emission scenarios.
            for rcp in rcps:

                # Select years.
                if rcp == cfg.rcp_ref:
                    hors = [cfg.per_ref]
                    cat_rcp = cfg.cat_obs
                    if cat == cfg.cat_scen:
                        d = os.path.dirname(cfg.get_p_obs(stn, var_or_idx))
                    else:
                        d = cfg.get_d_idx(stn, var_or_idx_code)
                else:
                    hors = cfg.per_hors
                    if cat == cfg.cat_scen:
                        cat_rcp = cfg.cat_scen
                        d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
                    else:
                        cat_rcp = cfg.cat_idx
                        d = cfg.get_d_idx(stn, var_or_idx_code)

                if not os.path.isdir(d):
                    continue

                utils.log("Processing: '" + stn + "', '" + var_or_idx_code + "', '" + rcp + "'", True)

                # Loop through statistics.
                stats = [cfg.stat_mean]
                if cat_rcp != cfg.cat_obs:
                    stats = stats + [cfg.stat_min, cfg.stat_max, cfg.stat_quantile]
                for stat in stats:
                    stat_quantiles = cfg.opt_stat_quantiles
                    if stat != cfg.stat_quantile:
                        stat_quantiles = [-1]
                    for q in stat_quantiles:
                        if ((q <= 0) or (q >= 1)) and (q != -1):
                            continue

                        # Loop through horizons.
                        for hor in hors:

                            # Calculate statistics.
                            ds_stat =\
                                calc_stat(cat_rcp, freq, cfg.freq_YS, stn, var_or_idx_code, rcp, hor, True, stat, q)
                            if ds_stat is None:
                                continue

                            # Select period.
                            ds_stat_hor = utils.sel_period(ds_stat.squeeze(), hor)

                            # Extract value.
                            if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                                val = ds_stat_hor.sum() / (hor[1] - hor[0] + 1)
                            else:
                                val = ds_stat_hor.mean()
                            val = float(val[var_or_idx])

                            # Add row.
                            stn_list.append(stn)
                            rcp_list.append(rcp)
                            hor_list.append(str(hor[0]) + "-" + str(hor[1]))
                            if cat_rcp == cfg.cat_obs:
                                stat = "none"
                            stat_list.append(stat)
                            q_list.append(str(q))
                            val_list.append(round(val, 6))

                            # Clearing cache.
                            # This is an ugly patch. Otherwise, the value of 'val' is incorrect.
                            caching.clear_cache()

            # Save results.
            if len(stn_list) > 0:

                # Build pandas dataframe.
                dict_pd = {"stn": stn_list, ("var" if cat == cfg.cat_scen else "idx"): [var_or_idx] * len(stn_list),
                           "rcp": rcp_list, "hor": hor_list, "stat": stat_list, "q": q_list, "val": val_list}
                df = pd.DataFrame(dict_pd)

                # Save file.
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

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Loop through variables.
        var_or_idx_list = cfg.variables_cordex if cat == cfg.cat_scen else cfg.idx_names
        for i_var_or_idx in range(len(var_or_idx_list)):
            var_or_idx = var_or_idx_list[i_var_or_idx]
            var_or_idx_code = var_or_idx if cat == cfg.cat_scen else cfg.idx_codes[i_var_or_idx]

            # Minimum and maximum values along the y-axis
            ylim = []

            utils.log("Processing: '" + stn + "', '" + var_or_idx_code + "'", True)

            # Files to be created.
            p_csv = cfg.get_d_scen(stn, cfg.cat_fig + "/" + cat + "/time_series", var_or_idx_code + "_csv") + \
                var_or_idx + "_" + stn + cfg.f_ext_csv
            if not (((cat == cfg.cat_scen) and (cfg.opt_plot[0])) or ((cat == cfg.cat_idx) and (cfg.opt_plot[1]))) and\
                    (os.path.exists(p_csv) or cfg.opt_force_overwrite):
                continue

            # Loop through emission scenarios.
            ds_ref = None
            ds_rcp_26, ds_rcp_26_grp = [], []
            ds_rcp_45, ds_rcp_45_grp = [], []
            ds_rcp_85, ds_rcp_85_grp = [], []
            ds_rcp_xx, ds_rcp_xx_grp = [], []
            for rcp in rcps:

                # List files.
                if rcp == cfg.rcp_ref:
                    if var_or_idx in cfg.variables_cordex:
                        p_sim_list = [cfg.get_d_stn(var_or_idx) + var_or_idx + "_" + stn + cfg.f_ext_nc]
                    else:
                        p_sim_list = [cfg.get_d_idx(stn, var_or_idx_code) + var_or_idx + "_ref" + cfg.f_ext_nc]
                else:
                    if var_or_idx in cfg.variables_cordex:
                        d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
                    else:
                        d = cfg.get_d_idx(stn, var_or_idx_code)
                    p_sim_list = glob.glob(d + "*_" + ("*" if rcp == cfg.rcp_xx else rcp) + cfg.f_ext_nc)

                # Exit if there is no file corresponding to the criteria.
                if (len(p_sim_list) == 0) or \
                   ((len(p_sim_list) > 0) and not(os.path.isdir(os.path.dirname(p_sim_list[0])))):
                    continue

                # Loop through simulation files.
                for i_sim in range(len(p_sim_list)):

                    # Load dataset.
                    ds = utils.open_netcdf(p_sim_list[i_sim]).squeeze()
                    if (rcp == cfg.rcp_ref) and (var_or_idx in cfg.variables_cordex):
                        ds = utils.remove_feb29(ds)
                        ds = utils.sel_period(ds, cfg.per_ref)

                    # Records years and units.
                    years  = ds.groupby(ds.time.dt.year).groups.keys()
                    units = ds[var_or_idx].attrs[cfg.attrs_units] if cat == cfg.cat_scen else ds.attrs[cfg.attrs_units]
                    if units == "degree_C":
                        units = cfg.unit_C

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

                    # Calculate statistics.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                            ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
                        else:
                            ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
                    n_time = len(ds[cfg.dim_time].values)
                    da = xr.DataArray(np.array(ds[var_or_idx].values), name=var_or_idx,
                                      coords=[(cfg.dim_time, np.arange(n_time))])

                    # Create dataset.
                    ds = da.to_dataset()
                    ds[cfg.dim_time] = utils.reset_calendar_list(years)
                    ds[var_or_idx].attrs[cfg.attrs_units] = units

                    # Convert units.
                    if var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                        if ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K:
                            ds = ds - cfg.d_KC
                            ds[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_C
                    elif var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                        if ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kg_m2s1:
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
                    else:
                        ds_is_ok = True
                        if (var_or_idx == cfg.idx_rainqty) and (float(ds[var_or_idx].max().values) == 0):
                            ds_is_ok = False
                        if ds_is_ok:
                            if rcp == cfg.rcp_26:
                                ds_rcp_26.append(ds)
                            elif rcp == cfg.rcp_45:
                                ds_rcp_45.append(ds)
                            elif rcp == cfg.rcp_85:
                                ds_rcp_85.append(ds)
                            ds_rcp_xx.append(ds)

                # Group by RCP.
                if rcp != cfg.rcp_ref:
                    if rcp == cfg.rcp_26:
                        ds_rcp_26_grp = calc_stat_mean_min_max(ds_rcp_26, var_or_idx)
                    elif rcp == cfg.rcp_45:
                        ds_rcp_45_grp = calc_stat_mean_min_max(ds_rcp_45, var_or_idx)
                    elif rcp == cfg.rcp_85:
                        ds_rcp_85_grp = calc_stat_mean_min_max(ds_rcp_85, var_or_idx)
                    ds_rcp_xx_grp = calc_stat_mean_min_max(ds_rcp_xx, var_or_idx)

            if (ds_ref is not None) or (ds_rcp_26 != []) or (ds_rcp_45 != []) or (ds_rcp_85 != []):

                # Generate statistics.
                if cfg.opt_save_csv:

                    # Extract years.
                    years = []
                    if cfg.rcp_26 in rcps:
                        years = utils.extract_date_field(ds_rcp_26_grp[0], "year")
                    elif cfg.rcp_45 in rcps:
                        years = utils.extract_date_field(ds_rcp_45_grp[0], "year")
                    elif cfg.rcp_85 in rcps:
                        years = utils.extract_date_field(ds_rcp_85_grp[0], "year")

                    # Initialize pandas dataframe.
                    dict_pd = {"year": years}
                    df = pd.DataFrame(dict_pd)
                    df[cfg.rcp_ref] = None

                    # Add values.
                    for rcp in rcps:
                        if rcp == cfg.rcp_ref:
                            years = utils.extract_date_field(ds_ref, "year")
                            vals = ds_ref[var_or_idx].values
                            for i in range(len(vals)):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=SettingWithCopyWarning)
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
                        else:
                            df[cfg.rcp_xx + "_min"] = ds_rcp_xx_grp[0][var_or_idx].values
                            df[cfg.rcp_xx + "_moy"] = ds_rcp_xx_grp[1][var_or_idx].values
                            df[cfg.rcp_xx + "_max"] = ds_rcp_xx_grp[2][var_or_idx].values

                    # Save file.
                    utils.save_csv(df, p_csv)

                # Generate plots.
                if ((cat == cfg.cat_scen) and (cfg.opt_plot[0])) or ((cat == cfg.cat_idx) and (cfg.opt_plot[1])):

                    # Time series with simulations grouped by RCP scenario.
                    p_fig_rcp = cfg.get_d_scen(stn, cfg.cat_fig + "/" + cat + "/time_series", var_or_idx_code) + \
                                var_or_idx + "_" + stn + "_rcp" + cfg.f_ext_png
                    plot.plot_ts(ds_ref, ds_rcp_26_grp, ds_rcp_45_grp, ds_rcp_85_grp, stn.capitalize(),
                                 var_or_idx_code, rcps, ylim, p_fig_rcp, 1)

                    # Time series showing individual simulations.
                    p_fig_sim = p_fig_rcp.replace("_rcp" + cfg.f_ext_png, "_sim" + cfg.f_ext_png)
                    plot.plot_ts(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(),
                                 var_or_idx_code, rcps, ylim, p_fig_sim, 2)


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
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_txdaysabove).

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
            val = float(ds[var_or_idx][i_time].values)
            if not np.isnan(val):
                vals.append(val)
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


def calc_monthly(ds: xr.Dataset, var: str, freq: str) -> List[xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate monthly values (mean, min, max).

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    var: str
        Climate variable.
    freq: str
        Frequency.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Initialize array of results.
    ds_list = []

    # Extract data for the current month.
    if cfg.dim_rlon in ds.dims:
        da_m = ds[var].rename({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})
    elif cfg.dim_lon in ds.dims:
        da_m = ds[var].rename({cfg.dim_lon: cfg.dim_longitude, cfg.dim_lat: cfg.dim_latitude})
    else:
        da_m = ds[var]

    # Grouping frequency.
    freq_str = "time.month" if freq == cfg.freq_MS else "time.dayofyear"
    time_str = "M" if freq == cfg.freq_MS else "1D"

    # Summarize data per month.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_m = da_m.mean(dim={cfg.dim_longitude, cfg.dim_latitude})
        if freq != cfg.freq_MS:

            # Extract values.
            if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                da_mean = da_m.resample(time=time_str).sum().groupby(freq_str).mean()
                da_min  = da_m.resample(time=time_str).sum().groupby(freq_str).min()
                da_max  = da_m.resample(time=time_str).sum().groupby(freq_str).max()
            else:
                da_mean = da_m.resample(time=time_str).mean().groupby(freq_str).mean()
                da_min  = da_m.resample(time=time_str).mean().groupby(freq_str).min()
                da_max  = da_m.resample(time=time_str).mean().groupby(freq_str).max()

            # Create dataset
            da_mean.name = da_min.name = da_max.name = var
            for i in range(3):
                if i == 0:
                    ds_m = da_mean.to_dataset()
                elif i == 1:
                    ds_m = da_min.to_dataset()
                else:
                    ds_m = da_max.to_dataset()
                ds_m[var].attrs[cfg.attrs_units] = ds[var].attrs[cfg.attrs_units]
                ds_list.append(ds_m)

        else:

            # Extract values.
            arr_all = []
            for m in range(1, 13):
                if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                    vals_m = list(da_m[da_m["time.month"] == m].resample(time=cfg.freq_YS).sum().values)
                else:
                    vals_m = list(da_m[da_m["time.month"] == m].resample(time=cfg.freq_YS).mean().values)

                arr_all.append(vals_m)

            # Create dataset.
            dict_pd = {var: arr_all}
            df = pd.DataFrame(dict_pd)
            ds_list = df.to_xarray()
            ds_list.attrs[cfg.attrs_units] = ds[var].attrs[cfg.attrs_units]

    return ds_list


def calc_heatmap(var_or_idx_code: str, per_hor_forced: [int] = None):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate heat map.

    Parameters
    ----------
    var_or_idx_code: str
        Climate index code.
    per_hor_forced: [int], optional
        Only considering a single horizon, rather than taking all available horizons.
    --------------------------------------------------------------------------------------------------------------------
    """

    cat = cfg.cat_scen if var_or_idx_code in cfg.variables_cordex else cfg.cat_idx
    var_or_idx = var_or_idx_code if cat == cfg.cat_scen else cfg.extract_idx(var_or_idx_code)
    rcps = [cfg.rcp_ref, cfg.rcp_xx] + cfg.rcps

    if per_hor_forced is None:
        per_str = str(cfg.per_ref[0]) + "-" + str(cfg.per_hors[len(cfg.per_hors) - 1][1])
    else:
        per_str = str(per_hor_forced[0]) + "-" + str(per_hor_forced[1])

    msg = "Calculating maps (" + per_str + ")."
    utils.log(msg, True)

    # Prepare data -----------------------------------------------------------------------------------------------------

    # XY grid boundaries.
    grid_x = grid_y = None

    # Calculate the overall minimum and maximum values (considering all maps for the current 'var_or_idx').
    z_min = z_max = z_min_delta = z_max_delta = np.nan

    # Calculated datasets and item description.
    arr_ds_maps = []
    arr_items = []

    # Build arrays for statistics to calculate.
    arr_stat = [cfg.stat_mean]
    arr_q = [-1]
    if cfg.opt_map_quantiles is not None:
        arr_stat = arr_stat + ([cfg.stat_quantile] * len(cfg.opt_map_quantiles))
        arr_q = arr_q + cfg.opt_map_quantiles

    # Loop through statistics.
    for i in range(len(arr_stat)):
        stat = arr_stat[i]
        q = arr_q[i]

        # Loop through categories of emission scenarios.
        for j in range(len(rcps)):
            rcp = rcps[j]

            # Loop through horizons.
            if j == 0:
                per_hors = [cfg.per_ref]
            elif per_hor_forced is not None:
                per_hors = [cfg.per_ref, per_hor_forced]
            else:
                per_hors = cfg.per_hors
            for per_hor in per_hors:

                # Current map.
                ds_map = calc_heatmap_rcp(var_or_idx_code, rcp, per_hor, stat, q)

                # Record current map, statistic and quantile, RCP and period.
                arr_ds_maps.append(ds_map)
                arr_items.append([stat, q, rcp, per_hor])

                # Reference map.
                ds_map_ref = None
                if per_hor != cfg.per_ref:
                    for k in range(len(arr_ds_maps)):
                        if (arr_items[k][0] == stat) and\
                           (arr_items[k][1] == q) and\
                           (arr_items[k][2] == cfg.rcp_xx) and\
                           (arr_items[k][3] == cfg.per_ref):
                            ds_map_ref = arr_ds_maps[k]
                            break

                # Record XY grid boundaries.
                if ((grid_x is None) or (grid_y is None)) and (not cfg.opt_ra):
                    grid_x, grid_y = utils.get_coordinates(ds_map, True)

                # Extract values.
                vals = ds_map[var_or_idx].values
                vals_delta = None
                if ds_map_ref is not None:
                    vals_delta = ds_map[var_or_idx].values - ds_map_ref[var_or_idx].values

                # Record mean values.
                # Mean absolute values.
                z_min_j = np.nanmin(vals)
                z_max_j = np.nanmax(vals)
                z_min = z_min_j if np.isnan(z_min) else min(z_min, z_min_j)
                z_max = z_max_j if np.isnan(z_max) else max(z_max, z_max_j)
                # Mean delta values.
                if vals_delta is not None:
                    z_min_delta_j = np.nanmin(vals_delta)
                    z_max_delta_j = np.nanmax(vals_delta)
                    z_min_delta = z_min_delta_j if np.isnan(z_min_delta) else min(z_min_delta, z_min_delta_j)
                    z_max_delta = z_max_delta_j if np.isnan(z_max_delta) else max(z_max_delta, z_max_delta_j)

    # Generate maps ----------------------------------------------------------------------------------------------------

    stn = "stns" if not cfg.opt_ra else cfg.obs_src

    # Loop through maps.
    for i in range(len(arr_ds_maps)):

        # Current map.
        ds_map = arr_ds_maps[i]

        # Current RCP and horizon.
        stat = arr_items[i][0]
        q = arr_items[i][1]
        rcp = arr_items[i][2]
        per_hor = arr_items[i][3]

        # Reference map.
        ds_map_ref = None
        if per_hor != cfg.per_ref:
            for j in range(len(arr_ds_maps)):
                if (arr_items[j][0] == stat) and \
                   (arr_items[j][1] == q) and \
                   (arr_items[j][2] == cfg.rcp_xx) and \
                   (arr_items[j][3] == cfg.per_ref):
                    ds_map_ref = arr_ds_maps[j]
                    break

        # Perform twice (for values, then for deltas).
        for j in range(2):

            # Skip if not relevant.
            if (j == 1) and \
               ((ds_map_ref is None) or
                ((cat == cfg.cat_scen) and (not cfg.opt_map_delta[0])) or
                ((cat == cfg.cat_idx) and (not cfg.opt_map_delta[1]))):
                continue

            # Extract dataset (calculate delta if required).
            if j == 0:
                da_map = ds_map[var_or_idx]
            else:
                da_map = ds_map[var_or_idx] - ds_map_ref[var_or_idx]

            # Plots ----------------------------------------------------------------------------------------------------

            # Path.
            d_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/" + cat + "/maps", var_or_idx_code + "/" + per_str)
            if stat in [cfg.stat_mean, cfg.stat_min, cfg.stat_max]:
                stat_str = "_" + stat
            else:
                stat_str = "_q" + str(int(q*100)).rjust(2, "0")
            fn_fig = var_or_idx + "_" + rcp + "_" + str(per_hor[0]) + "_" + str(per_hor[1]) + stat_str + cfg.f_ext_png
            p_fig = d_fig + fn_fig
            if j == 1:
                p_fig = p_fig.replace(cfg.f_ext_png, "_delta" + cfg.f_ext_png)

            # Generate plot.
            if ((cat == cfg.cat_scen) and (cfg.opt_map[0])) or ((cat == cfg.cat_idx) and (cfg.opt_map[1])):
                z_min_map = z_min if j == 0 else z_min_delta
                z_max_map = z_max if j == 0 else z_max_delta
                plot.plot_heatmap(da_map, stn, var_or_idx_code, grid_x, grid_y, rcp, per_hor, stat, q, z_min_map,
                                  z_max_map, j == 1, p_fig)

            # CSV files ------------------------------------------------------------------------------------------------

            # Path.
            d_csv = cfg.get_d_scen(stn, cfg.cat_fig + "/" + cat + "/maps", var_or_idx_code + "/" + per_str + "_csv")
            fn_csv = fn_fig.replace(cfg.f_ext_png, cfg.f_ext_csv)
            p_csv = d_csv + fn_csv
            if j == 1:
                p_csv = p_csv.replace(cfg.f_ext_csv, "_delta" + cfg.f_ext_csv)

            # Save.
            if cfg.opt_save_csv and (not os.path.exists(p_csv) or cfg.opt_force_overwrite):

                # Extract data.
                lon_list = []
                lat_list = []
                val_list = []
                for k in range(len(da_map.longitude.values)):
                    for l in range(len(da_map.latitude.values)):
                        lon_list.append(da_map.longitude.values[k])
                        lat_list.append(da_map.latitude.values[l])
                        val_list.append(da_map.values[l, k])

                # Build dataframe.
                dict_pd = {cfg.dim_longitude: lon_list, cfg.dim_latitude: lat_list, var_or_idx: val_list}
                df = pd.DataFrame(dict_pd)

                # Save to file.
                utils.save_csv(df, p_csv)


def calc_heatmap_rcp(var_or_idx_code: str, rcp: str, per: [int], stat: str, q: float) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate heat map for a given RCP.

    Parameters
    ----------
    var_or_idx_code: str
        Climate variable (ex: cfg.var_cordex_tasmax) or climate index code (ex: cfg.idx_txdaysabove).
    rcp: str
        Emission scenario.
    per: [int]
        Period.
    stat: str
        Statistic = {"mean" or "quantile"}
    q: float
        Quantile.
    --------------------------------------------------------------------------------------------------------------------
    """

    cat = cfg.cat_scen if var_or_idx_code in cfg.variables_cordex else cfg.cat_idx
    var_or_idx = var_or_idx_code if cat == cfg.cat_scen else cfg.extract_idx(var_or_idx_code)

    utils.log("Processing: " + var_or_idx_code + ", " +
              ("q" + str(int(q * 100)).rjust(2, "0") if q >= 0 else stat) + ", " +
              cfg.get_rcp_desc(rcp) + ", " + str(per[0]) + "-" + str(per[1]) + "", True)

    # Number of years and stations.
    if rcp == cfg.rcp_ref:
        n_year = cfg.per_ref[1] - cfg.per_ref[0] + 1
    else:
        n_year = cfg.per_fut[1] - cfg.per_ref[1] + 1

    # List stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]

    # Observations -----------------------------------------------------------------------------------------------------

    if not cfg.opt_ra:

        # Get information on stations.
        p_stn = glob.glob(cfg.get_d_stn(cfg.var_cordex_tas) + "../*" + cfg.f_ext_csv)[0]
        df = pd.read_csv(p_stn, sep=cfg.f_sep)

        # Collect values for each station and determine overall boundaries.
        utils.log("Collecting emissions scenarios at each station.", True)
        x_bnds = []
        y_bnds = []
        data_stn = []
        for stn in stns:

            # Get coordinates.
            lon = df[df["station"] == stn][cfg.dim_lon].values[0]
            lat = df[df["station"] == stn][cfg.dim_lat].values[0]

            # Calculate statistics.
            if rcp == cfg.rcp_ref:
                ds_stat =\
                    calc_stat(cfg.cat_obs, cfg.freq_YS, cfg.freq_YS, stn, var_or_idx_code, rcp, None, False, stat, q)
            else:
                ds_stat =\
                    calc_stat(cfg.cat_scen, cfg.freq_YS, cfg.freq_YS, stn, var_or_idx_code, rcp, None, False, stat, q)
            if ds_stat is None:
                continue

            # Extract data from stations.
            data = [[], [], []]
            n = ds_stat.dims[cfg.dim_time]
            for year in range(0, n):

                # Collect data.
                x = float(lon)
                y = float(lat)
                z = float(ds_stat[var_or_idx][0][0][year])
                if math.isnan(z):
                    z = float(0)
                data[0].append(x)
                data[1].append(y)
                data[2].append(z)

                # Update overall boundaries (round according to the variable 'step').
                if not x_bnds:
                    x_bnds = [x, x]
                    y_bnds = [y, y]
                else:
                    x_bnds = [min(x_bnds[0], x), max(x_bnds[1], x)]
                    y_bnds = [min(y_bnds[0], y), max(y_bnds[1], y)]

            # Add data from station to complete dataset.
            data_stn.append(data)

        # Build the list of x and y locations for which interpolation is needed.
        utils.log("Collecting the coordinates of stations.", True)
        grid_time = range(0, n_year)

        def round_to_nearest_decimal(val, step):
            if val < 0:
                val_rnd = math.floor(val/step) * step
            else:
                val_rnd = math.ceil(val/step) * step
            return val_rnd

        for i in range(0, 2):
            x_bnds[i] = round_to_nearest_decimal(x_bnds[i], cfg.idx_resol)
            y_bnds[i] = round_to_nearest_decimal(y_bnds[i], cfg.idx_resol)
        grid_x = np.arange(x_bnds[0], x_bnds[1] + cfg.idx_resol, cfg.idx_resol)
        grid_y = np.arange(y_bnds[0], y_bnds[1] + cfg.idx_resol, cfg.idx_resol)

        # Perform interpolation.
        # There is a certain flexibility regarding the number of years in a dataset. Ideally, the station should not
        # have been considered in the analysis, unless there is no better option.
        utils.log("Performing interpolation.", True)
        new_grid = np.meshgrid(grid_x, grid_y)
        new_grid_data = np.empty((n_year, len(grid_y), len(grid_x)))
        for i_year in range(0, n_year):
            arr_x = []
            arr_y = []
            arr_z = []
            for i_stn in range(len(data_stn)):
                if i_year < len(data_stn[i_stn][0]):
                    arr_x.append(data_stn[i_stn][0][i_year])
                    arr_y.append(data_stn[i_stn][1][i_year])
                    arr_z.append(data_stn[i_stn][2][i_year])
            new_grid_data[i_year, :, :] =\
                griddata((arr_x, arr_y), np.array(arr_z), (new_grid[0], new_grid[1]), fill_value=np.nan,
                         method="linear")
        da_itp = xr.DataArray(new_grid_data,
                              coords={cfg.dim_time: grid_time, cfg.dim_lat: grid_y, cfg.dim_lon: grid_x},
                              dims=[cfg.dim_time, cfg.dim_lat, cfg.dim_lon])
        ds_itp = da_itp.to_dataset(name=var_or_idx)

    # There is no need to interpolate.
    else:

        # Reference period.
        if var_or_idx in cfg.variables_cordex:
            p_sim_ref = cfg.get_d_scen(cfg.obs_src, cfg.cat_obs, var_or_idx) +\
                        var_or_idx + "_" + cfg.obs_src + cfg.f_ext_nc
        else:
            p_sim_ref = cfg.get_d_idx(cfg.obs_src, var_or_idx_code) + var_or_idx + "_ref" + cfg.f_ext_nc
        if rcp == cfg.rcp_ref:

            # Open dataset.
            p_sim = p_sim_ref
            if not os.path.exists(p_sim):
                return xr.Dataset(None)
            ds_sim = utils.open_netcdf(p_sim)

            # Select period.
            ds_sim = utils.remove_feb29(ds_sim)
            ds_sim = utils.sel_period(ds_sim, per)

            # Extract units.
            units = 1
            if cfg.attrs_units in ds_sim[var_or_idx].attrs:
                units = ds_sim[var_or_idx].attrs[cfg.attrs_units]
            elif cfg.attrs_units in ds_sim.data_vars:
                units = ds_sim[cfg.attrs_units]

            # Calculate mean.
            ds_itp = ds_sim.mean(dim=cfg.dim_time)
            if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                ds_itp = ds_itp * 365
            if cat == cfg.cat_scen:
                ds_itp[var_or_idx].attrs[cfg.attrs_units] = units
            else:
                ds_itp.attrs[cfg.attrs_units] = units

        # Future period.
        else:

            # List scenarios or indices for the current RCP.
            if cat == cfg.cat_scen:
                d = cfg.get_d_scen(cfg.obs_src, cfg.cat_qqmap, var_or_idx)
            else:
                d = cfg.get_d_scen(cfg.obs_src, cfg.cat_idx, var_or_idx_code)
            p_sim_list = [i for i in glob.glob(d + "*" + cfg.f_ext_nc) if i != p_sim_ref]

            # Collect simulations.
            arr_sim = []
            ds_itp  = None
            n_sim   = 0
            units   = 1
            for p_sim in p_sim_list:
                if os.path.exists(p_sim) and ((rcp in p_sim) or (rcp == cfg.rcp_xx)):

                    # Open dataset.
                    ds_sim = utils.open_netcdf(p_sim)

                    # Extract units.
                    if cfg.attrs_units in ds_sim[var_or_idx].attrs:
                        units = ds_sim[var_or_idx].attrs[cfg.attrs_units]
                    elif cfg.attrs_units in ds_sim.data_vars:
                        units = ds_sim[cfg.attrs_units]

                    # Select period.
                    ds_sim = utils.remove_feb29(ds_sim)
                    ds_sim = utils.sel_period(ds_sim, per)

                    # Calculate mean and add to array.
                    ds_sim = ds_sim.mean(dim=cfg.dim_time)
                    if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                        ds_sim = ds_sim * 365
                    if cat == cfg.cat_scen:
                        ds_sim[var_or_idx].attrs[cfg.attrs_units] = units
                    else:
                        ds_sim.attrs[cfg.attrs_units] = units
                    arr_sim.append(ds_sim)

                    # The first dataset will be used to return result.
                    if n_sim == 0:
                        ds_itp = ds_sim
                    n_sim = n_sim + 1

            if ds_itp is None:
                return xr.Dataset(None)

            # Calculate statistics.
            dims_itp = ds_itp[var_or_idx].dims
            if cfg.dim_rlat in dims_itp:
                lat = ds_itp[var_or_idx][cfg.dim_rlat]
                lon = ds_itp[var_or_idx][cfg.dim_rlon]
            elif cfg.dim_lat in dims_itp:
                lat = ds_itp[var_or_idx][cfg.dim_lat]
                lon = ds_itp[var_or_idx][cfg.dim_lon]
            else:
                lat = ds_itp[var_or_idx][cfg.dim_latitude]
                lon = ds_itp[var_or_idx][cfg.dim_longitude]
            len_lat = len(lat)
            len_lon = len(lon)
            for i_lat in range(len_lat):
                for i_lon in range(len_lon):

                    # Extract the value for the current cell for all simulations.
                    vals = []
                    for ds_sim in arr_sim:
                        dims_sim = ds_itp[var_or_idx].dims
                        if cfg.dim_rlat in dims_sim:
                            val_sim = float(ds_sim[var_or_idx].isel(rlon=i_lon, rlat=i_lat))
                        elif cfg.dim_lat in dims_sim:
                            val_sim = float(ds_sim[var_or_idx].isel(lon=i_lon, lat=i_lat))
                        else:
                            val_sim = float(ds_sim[var_or_idx].isel(longitude=i_lon, latitude=i_lat))
                        vals.append(val_sim)

                    # Calculate statistic.
                    val = np.nan
                    if stat == cfg.stat_mean:
                        val = np.mean(vals)
                    elif stat == cfg.stat_min:
                        val = np.min(vals)
                    elif stat == cfg.stat_max:
                        val = np.max(vals)
                    elif stat == cfg.stat_quantile:
                        val = np.quantile(vals, q)

                    # Record value.
                    ds_itp[var_or_idx][i_lat, i_lon] = val

        # Remember units.
        units = ds_itp[var_or_idx].attrs[cfg.attrs_units] if cat == cfg.cat_scen else ds_itp.attrs[cfg.attrs_units]

        # Convert units.
        if (var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
           (ds_itp[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K):
            ds_itp = ds_itp - cfg.d_KC
            ds_itp[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_C
        elif (var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]) and \
             (ds_itp[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kg_m2s1):
            ds_itp = ds_itp * cfg.spd
            ds_itp[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_mm
        else:
            ds_itp[var_or_idx].attrs[cfg.attrs_units] = units

        # Adjust coordinate names (required for clipping).
        if cfg.dim_longitude not in list(ds_itp.dims):
            if cfg.dim_rlon in ds_itp.dims:
                ds_itp = ds_itp.rename_dims({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})
                ds_itp[cfg.dim_longitude] = ds_itp[cfg.dim_rlon]
                del ds_itp[cfg.dim_rlon]
                ds_itp[cfg.dim_latitude] = ds_itp[cfg.dim_rlat]
                del ds_itp[cfg.dim_rlat]
            else:
                ds_itp = ds_itp.rename_dims({cfg.dim_lon: cfg.dim_longitude, cfg.dim_lat: cfg.dim_latitude})
                ds_itp[cfg.dim_longitude] = ds_itp[cfg.dim_lon]
                del ds_itp[cfg.dim_lon]
                ds_itp[cfg.dim_latitude] = ds_itp[cfg.dim_lat]
                del ds_itp[cfg.dim_lat]

    return ds_itp


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
            for i_var_or_idx in range(len(var_or_idx_list)):
                var_or_idx = cfg.idx_names[i_var_or_idx]
                var_or_idx_code = cfg.idx_codes[i_var_or_idx]

                # List NetCDF files.
                p_list = list(glob.glob(cfg.get_d_scen(stn, cat, var_or_idx_code) + cfg.f_ext_nc))
                n_files = len(p_list)
                if n_files == 0:
                    continue
                p_list.sort()

                utils.log("Processing: '" + stn + "', '" + var_or_idx_code + "'", True)

                # Scalar processing mode.
                if cfg.n_proc == 1:
                    for i_file in range(n_files):
                        conv_nc_csv_single(p_list, var_or_idx_code, i_file)

                # Parallel processing mode.
                else:

                    # Loop until all files have been converted.
                    while True:

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before =\
                            len(list(glob.glob(cfg.get_d_scen(stn, cat, var_or_idx_code) + "*" + cfg.f_ext_csv)))

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
                        n_files_proc_after =\
                            len(list(glob.glob(cfg.get_d_scen(stn, cat, var_or_idx_code) + "*" + cfg.f_ext_csv)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_files_proc_before == n_files_proc_after):
                            break


def conv_nc_csv_single(p_list: [str], var_or_idx_code: str, i_file: int):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a single NetCDF to CSV file.

    Parameters
    ----------
    p_list : [str]
        List of paths.
    var_or_idx_code : str
        Climate variable or index code.
    i_file : int
        Rank of file in 'p_list'.
    --------------------------------------------------------------------------------------------------------------------
    """

    var_or_idx = var_or_idx_code if (var_or_idx_code in cfg.variables_cordex) else cfg.extract_idx(var_or_idx_code)

    # Paths.
    p = p_list[i_file]
    p_csv = p.replace("/" + var_or_idx_code + "/", "/" + var_or_idx_code + "_" + cfg.f_csv + "/").\
        replace(cfg.f_ext_nc, cfg.f_ext_csv)
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
       (ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kg_m2s1):
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
