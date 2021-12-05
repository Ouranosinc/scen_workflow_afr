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


def calc_stat(
    data_type: str,
    freq_in: str,
    freq_out: str,
    stn: str,
    varidx_code: str,
    rcp: str,
    hor: [int],
    per_region: bool,
    clip: bool,
    stat: str,
    q: float = -1
) -> Union[xr.Dataset, None]:

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
    varidx_code : str
        Climate variable or index code.
    rcp : str
        Emission scenario: {cfg.rcp_26, cfg.rcp_45, cfg_rcp_85, cfg_rcp_xx}
    hor : [int]
        Horizon: ex: [1981, 2010]
        If None is specified, the complete time range is considered.
    per_region : bool
        If True, statistics are calculated for a region as a whole.
    clip : bool
        If True, clip according to 'cfg.d_bounds'.
    stat : str
        Statistic: {cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_quantile"}
    q : float, optional
        Quantile: value between 0 and 1.
    --------------------------------------------------------------------------------------------------------------------
    """

    cat = cfg.cat_scen if varidx_code in cfg.variables_cordex else cfg.cat_idx
    varidx_name = varidx_code if cat == cfg.cat_scen else cfg.extract_idx(varidx_code)
    varidx_code_grp = cfg.get_idx_group(varidx_code)

    # List files.
    if data_type == cfg.cat_obs:
        if varidx_name in cfg.variables_cordex:
            p_sim_l = [cfg.get_d_scen(stn, cfg.cat_obs, varidx_name) + varidx_name + "_" + stn + cfg.f_ext_nc]
        else:
            p_sim_l = [cfg.get_d_idx(stn, varidx_code_grp) + varidx_code_grp + "_ref" + cfg.f_ext_nc]
    else:
        if varidx_name in cfg.variables_cordex:
            d = cfg.get_d_scen(stn, cfg.cat_qqmap, varidx_name)
        else:
            d = cfg.get_d_idx(stn, varidx_code_grp)
        p_sim_l = glob.glob(d + "*_" + ("*rcp*" if rcp == cfg.rcp_xx else rcp) + cfg.f_ext_nc)

    # Exit if there is not file corresponding to the criteria.
    if (len(p_sim_l) == 0) or \
       ((len(p_sim_l) > 0) and not(os.path.isdir(os.path.dirname(p_sim_l[0])))):
        return None

    # List days.
    ds = utils.open_netcdf(p_sim_l[0])
    if (cfg.dim_lon in ds.variables) or (cfg.dim_lon in ds.dims):
        lon = ds[cfg.dim_lon]
        lat = ds[cfg.dim_lat]
    elif (cfg.dim_rlon in ds.variables) or (cfg.dim_rlon in ds.dims):
        lon = ds[cfg.dim_rlon]
        lat = ds[cfg.dim_rlat]
    else:
        lon = ds[cfg.dim_longitude]
        lat = ds[cfg.dim_latitude]
    if cfg.attrs_units in ds[varidx_name].attrs:
        units = ds[varidx_name].attrs[cfg.attrs_units]
    else:
        units = 1
    n_sim = len(p_sim_l)

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
        ds = utils.open_netcdf(p_sim_l[i_sim])

        # Patch used to fix potentially unordered dimensions of index 'cfg.idx_drought_code'.
        if varidx_code in cfg.idx_codes:
            ds = ds.resample(time=cfg.freq_YS).mean(dim=cfg.dim_time, keep_attrs=True)

        # Select years and adjust units.
        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
        ds = ds.sel(time=slice(years_str[0], years_str[1]))
        if cfg.attrs_units in ds[varidx_name].attrs:
            if (varidx_name in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
               (ds[varidx_name].attrs[cfg.attrs_units] == cfg.unit_K):
                ds = ds - cfg.d_KC
                ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_C
            elif varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                ds = ds * cfg.spd
                ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_mm
            elif varidx_name in [cfg.var_cordex_uas, cfg.var_cordex_vas, cfg.var_cordex_sfcwindmax]:
                ds = ds * cfg.km_h_per_m_s
                ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_km_h

        if cfg.opt_ra:

            # Statistics are calculated at a point.
            if cfg.d_bounds == "":
                ds = utils.subset_ctrl_pt(ds)

            # Statistics are calculated on a surface.
            else:

                # Clip to geographic boundaries.
                if clip and (cfg.d_bounds != ""):
                    ds = utils.subset_shape(ds)

                # Calculate mean value.
                ds = utils.squeeze_lon_lat(ds)

        # Records values.
        # Simulation data is assumed to be complete.
        if ds[varidx_name].time.size == n_time:
            if (cfg.dim_lon in ds.dims) or (cfg.dim_rlon in ds.dims) or (cfg.dim_longitude in ds.dims):
                vals = ds.squeeze()[varidx_name].values.tolist()
            else:
                vals = ds[varidx_name].values.tolist()
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
                            if ds_i[varidx_name].size != 0:
                                day_of_year = ds_i[varidx_name].time.dt.dayofyear.values[0]
                                if len(np.array(ds_i[varidx_name].values).shape) > 1:
                                    val = ds_i[varidx_name].values[0][0][0]
                                else:
                                    val = ds_i[varidx_name].values[0]
                                vals[(i_year - year_1) * 365 + day_of_year - 1] = val
                        except:
                            pass
            if cfg.opt_ra and (varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]):
                vals = [i * cfg.spd for i in vals]
            arr_vals.append(vals)

    # Calculate the mean value of all years.
    if per_region:
        for i in range(len(arr_vals)):
            arr_vals[i] = [np.nanmean(arr_vals[i])]
        n_time = 1

    # Transpose.
    arr_vals_t = []

    # Collapse to yearly frequency.
    if (freq_in == cfg.freq_D) and (freq_out == cfg.freq_YS) and cfg.opt_ra:
        for i_sim in range(n_sim):
            vals_sim = []
            for i_year in range(0, year_n - year_1 + 1):
                vals_year = arr_vals[i_sim][(365 * i_year):(365 * (i_year + 1))]
                da_vals = xr.DataArray(np.array(vals_year))
                vals_year = list(da_vals[np.isnan(da_vals.values) == False].values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                        val_year = np.nansum(vals_year)
                    else:
                        val_year = np.nanmean(vals_year)
                vals_sim.append(val_year)
            if (varidx_name != cfg.var_cordex_pr) | (sum(vals_sim) > 0):
                arr_vals_t.append(vals_sim)
        arr_vals = arr_vals_t
        n_time = year_n - year_1 + 1
    n_sim = len(arr_vals)

    # Transpose array.
    arr_vals_t = []
    for i_time in range(n_time):
        vals = []
        for i_sim in range(n_sim):
            if (varidx_name != cfg.idx_rain_season_prcptot) or\
               ((varidx_name == cfg.idx_rain_season_prcptot) and (max(arr_vals[i_sim]) > 0)):
                if n_time == 1:
                    vals.append(arr_vals[i_sim][0])
                else:
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
                                  ((varidx_name != cfg.var_cordex_pr) | (sum(da_vals.values) > 0))].values)
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
    da_stat = xr.DataArray(np.array(arr_stat), name=varidx_name, coords=[(cfg.dim_time, np.arange(n_time))])
    ds_stat = da_stat.to_dataset()
    ds_stat = ds_stat.expand_dims(lon=1, lat=1)

    # Adjust coordinates and time.
    if not(cfg.dim_lon in ds_stat.dims):
        ds_stat[cfg.dim_lon] = lon
        ds_stat[cfg.dim_lat] = lat
    if n_time > 1:
        ds_stat[cfg.dim_time] = utils.reset_calendar(ds_stat, year_1, year_n, freq_out)

    # Adjust units.
    ds_stat[varidx_name].attrs[cfg.attrs_units] = units

    return ds_stat


def calc_stats(
    cat: str
):

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

        # Explode lists of codes and names.
        idx_names_exploded = cfg.explode_idx_l(cfg.idx_names)
        idx_codes_exploded = cfg.explode_idx_l(cfg.idx_codes)

        # Loop through variables (or indices).
        varidx_name_l = cfg.variables_cordex if cat == cfg.cat_scen else idx_names_exploded
        for i_varidx in range(len(varidx_name_l)):
            varidx_name = varidx_name_l[i_varidx]
            varidx_code = varidx_name if cat == cfg.cat_scen else idx_codes_exploded[i_varidx]
            varidx_code_grp = cfg.get_idx_group(varidx_code)

            # Skip iteration if the file already exists.
            p_csv = cfg.get_d_scen(stn, cfg.cat_stat, cat + cfg.sep + varidx_code_grp) + varidx_code + cfg.f_ext_csv
            if os.path.exists(p_csv) and (not cfg.opt_force_overwrite):
                continue

            # Containers.
            stn_l  = []
            rcp_l  = []
            hor_l  = []
            stat_l = []
            q_l    = []
            val_l  = []

            # Loop through emission scenarios.
            for rcp in rcps:

                # Select years.
                if rcp == cfg.rcp_ref:
                    hors = [cfg.per_ref]
                    cat_rcp = cfg.cat_obs
                    if cat == cfg.cat_scen:
                        d = os.path.dirname(cfg.get_p_obs(stn, varidx_code_grp))
                    else:
                        d = cfg.get_d_idx(stn, varidx_code_grp)
                else:
                    hors = cfg.per_hors
                    if cat == cfg.cat_scen:
                        cat_rcp = cfg.cat_scen
                        d = cfg.get_d_scen(stn, cfg.cat_qqmap, varidx_code_grp)
                    else:
                        cat_rcp = cfg.cat_idx
                        d = cfg.get_d_idx(stn, varidx_code_grp)

                if not os.path.isdir(d):
                    continue

                idx_desc = varidx_name
                if varidx_code_grp != varidx_name:
                    idx_desc = varidx_code_grp + "." + idx_desc
                utils.log("Processing: " + stn + ", " + idx_desc + ", " + rcp + "", True)

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
                            # The use of boundaries was disabled, because the function cliops.subset does not always
                            # work (different result obtained for different runs with same exact data).
                            ds_stat =\
                                calc_stat(cat_rcp, freq, cfg.freq_YS, stn, varidx_code, rcp, hor,
                                          (freq == cfg.freq_YS) and (cat == cfg.cat_scen), cfg.opt_stat_clip, stat, q)
                            if ds_stat is None:
                                continue

                            # Select period.
                            if cfg.opt_ra:
                                ds_stat_hor = utils.sel_period(ds_stat.squeeze(), hor)
                            else:
                                ds_stat_hor = ds_stat.copy(deep=True)

                            # Extract value.
                            if varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                                val = float(ds_stat_hor[varidx_name].sum()) / (hor[1] - hor[0] + 1)
                            else:
                                val = float(ds_stat_hor[varidx_name].mean())

                            # Add row.
                            stn_l.append(stn)
                            rcp_l.append(rcp)
                            hor_l.append(str(hor[0]) + "-" + str(hor[1]))
                            if cat_rcp == cfg.cat_obs:
                                stat = "none"
                            stat_l.append(stat)
                            q_l.append(str(q))
                            val_l.append(round(val, 6))

                            # Clearing cache.
                            # This is an ugly patch. Otherwise, the value of 'val' is incorrect.
                            try:
                                caching.clear_cache()
                            except AttributeError:
                                pass

            # Save results.
            if len(stn_l) > 0:

                # Build pandas dataframe.
                dict_pd =\
                    {"stn": stn_l, ("var" if cat == cfg.cat_scen else "idx"): [varidx_name] * len(stn_l),
                     "rcp": rcp_l, "hor": hor_l, "stat": stat_l, "q": q_l, "val": val_l}
                df = pd.DataFrame(dict_pd)

                # Save file.
                utils.save_csv(df, p_csv)


def calc_ts(
    cat: str
):

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

        # Explode lists of codes and names.
        idx_names_exploded = cfg.explode_idx_l(cfg.idx_names)
        idx_codes_exploded = cfg.explode_idx_l(cfg.idx_codes)

        # Loop through variables.
        varidx_name_l = cfg.variables_cordex if cat == cfg.cat_scen else idx_names_exploded
        for i_varidx in range(len(varidx_name_l)):
            varidx_name = varidx_name_l[i_varidx]
            varidx_code = varidx_name if cat == cfg.cat_scen else idx_codes_exploded[i_varidx]
            varidx_code_grp = cfg.get_idx_group(varidx_code)

            # Minimum and maximum values along the y-axis
            ylim = []

            idx_desc = varidx_name
            if varidx_code_grp != varidx_name:
                idx_desc = varidx_code_grp + "." + idx_desc
            utils.log("Processing: " + stn + ", " + idx_desc, True)

            # Files to be created.
            p_csv = cfg.get_d_scen(stn, cfg.cat_fig + cfg.sep + cat + cfg.sep + "time_series", varidx_code_grp +
                                   "_csv") + varidx_name + cfg.f_ext_csv
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
                    if varidx_name in cfg.variables_cordex:
                        p_sim_l = [cfg.get_d_stn(varidx_code_grp) + varidx_code_grp + "_" + stn + cfg.f_ext_nc]
                    else:
                        p_sim_l = [cfg.get_d_idx(stn, varidx_code_grp) + varidx_code_grp + "_ref" + cfg.f_ext_nc]
                else:
                    if varidx_name in cfg.variables_cordex:
                        d = cfg.get_d_scen(stn, cfg.cat_qqmap, varidx_code_grp)
                    else:
                        d = cfg.get_d_idx(stn, varidx_code_grp)
                    p_sim_l = glob.glob(d + "*_" + ("*" if rcp == cfg.rcp_xx else rcp) + cfg.f_ext_nc)

                # Exit if there is no file corresponding to the criteria.
                if (len(p_sim_l) == 0) or \
                   ((len(p_sim_l) > 0) and not(os.path.isdir(os.path.dirname(p_sim_l[0])))):
                    continue

                # Loop through simulation files.
                for i_sim in range(len(p_sim_l)):

                    # Skip iteration if file doesn't exist.
                    if not os.path.exists(p_sim_l[i_sim]):
                        continue

                    # Load dataset.
                    ds = utils.open_netcdf(p_sim_l[i_sim]).squeeze()
                    if (rcp == cfg.rcp_ref) and (varidx_name in cfg.variables_cordex):
                        ds = utils.remove_feb29(ds)
                        ds = utils.sel_period(ds, cfg.per_ref)

                    # Records years and units.
                    years  = ds.groupby(ds.time.dt.year).groups.keys()
                    units =\
                        ds[varidx_name].attrs[cfg.attrs_units] if cat == cfg.cat_scen else ds.attrs[cfg.attrs_units]
                    if units == "degree_C":
                        units = cfg.unit_C

                    # Select control point.
                    if cfg.opt_ra:
                        if cfg.d_bounds == "":
                            ds = utils.subset_ctrl_pt(ds)
                        else:
                            ds = utils.squeeze_lon_lat(ds, varidx_name)

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
                        if varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                            ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
                        else:
                            ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
                    n_time = len(ds[cfg.dim_time].values)
                    da = xr.DataArray(np.array(ds[varidx_name].values), name=varidx_name,
                                      coords=[(cfg.dim_time, np.arange(n_time))])

                    # Create dataset.
                    ds = da.to_dataset()
                    ds[cfg.dim_time] = utils.reset_calendar_l(years)
                    ds[varidx_name].attrs[cfg.attrs_units] = units

                    # Convert units.
                    if varidx_name in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                        if ds[varidx_name].attrs[cfg.attrs_units] == cfg.unit_K:
                            ds = ds - cfg.d_KC
                            ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_C
                    elif varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                        if ds[varidx_name].attrs[cfg.attrs_units] == cfg.unit_kg_m2s1:
                            ds = ds * cfg.spd
                            ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_mm
                    elif varidx_name in [cfg.var_cordex_uas, cfg.var_cordex_vas, cfg.var_cordex_sfcwindmax]:
                        ds = ds * cfg.km_h_per_m_s
                        ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_km_h

                    # Calculate minimum and maximum values along the y-axis.
                    if not ylim:
                        ylim = [min(ds[varidx_name].values), max(ds[varidx_name].values)]
                    else:
                        ylim = [min(ylim[0], min(ds[varidx_name].values)),
                                max(ylim[1], max(ds[varidx_name].values))]

                    # Append to list of datasets.
                    if rcp == cfg.rcp_ref:
                        ds_ref = ds
                    else:
                        ds_is_ok = True
                        if (varidx_name == cfg.idx_rain_season_prcptot) and (float(ds[varidx_name].max().values) == 0):
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
                        ds_rcp_26_grp = calc_stat_mean_min_max(ds_rcp_26, varidx_name)
                    elif rcp == cfg.rcp_45:
                        ds_rcp_45_grp = calc_stat_mean_min_max(ds_rcp_45, varidx_name)
                    elif rcp == cfg.rcp_85:
                        ds_rcp_85_grp = calc_stat_mean_min_max(ds_rcp_85, varidx_name)
                    ds_rcp_xx_grp = calc_stat_mean_min_max(ds_rcp_xx, varidx_name)

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
                            vals = ds_ref[varidx_name].values
                            for i in range(len(vals)):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                                    df[cfg.rcp_ref][df["year"] == years[i]] = vals[i]
                        elif rcp == cfg.rcp_26:
                            df[cfg.rcp_26 + "_moy"] = ds_rcp_26_grp[0][varidx_name].values
                            df[cfg.rcp_26 + "_min"] = ds_rcp_26_grp[1][varidx_name].values
                            df[cfg.rcp_26 + "_max"] = ds_rcp_26_grp[2][varidx_name].values
                        elif rcp == cfg.rcp_45:
                            df[cfg.rcp_45 + "_moy"] = ds_rcp_45_grp[0][varidx_name].values
                            df[cfg.rcp_45 + "_min"] = ds_rcp_45_grp[1][varidx_name].values
                            df[cfg.rcp_45 + "_max"] = ds_rcp_45_grp[2][varidx_name].values
                        elif rcp == cfg.rcp_85:
                            df[cfg.rcp_85 + "_moy"] = ds_rcp_85_grp[0][varidx_name].values
                            df[cfg.rcp_85 + "_min"] = ds_rcp_85_grp[1][varidx_name].values
                            df[cfg.rcp_85 + "_max"] = ds_rcp_85_grp[2][varidx_name].values
                        else:
                            df[cfg.rcp_xx + "_moy"] = ds_rcp_xx_grp[0][varidx_name].values
                            df[cfg.rcp_xx + "_min"] = ds_rcp_xx_grp[1][varidx_name].values
                            df[cfg.rcp_xx + "_max"] = ds_rcp_xx_grp[2][varidx_name].values

                    # Save file.
                    utils.save_csv(df, p_csv)

                # Generate plots.
                if ((cat == cfg.cat_scen) and (cfg.opt_plot[0])) or ((cat == cfg.cat_idx) and (cfg.opt_plot[1])):

                    # Time series with simulations grouped by RCP scenario.
                    p_fig_rcp = cfg.get_d_scen(stn, cfg.cat_fig + cfg.sep + cat + cfg.sep + "time_series",
                                               varidx_code_grp) + varidx_name + "_rcp" + cfg.f_ext_png
                    plot.plot_ts(ds_ref, ds_rcp_26_grp, ds_rcp_45_grp, ds_rcp_85_grp, stn.capitalize(),
                                 varidx_code, rcps, ylim, p_fig_rcp, 1)

                    # Time series showing individual simulations.
                    p_fig_sim = p_fig_rcp.replace("_rcp" + cfg.f_ext_png, "_sim" + cfg.f_ext_png)
                    plot.plot_ts(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(),
                                 varidx_code, rcps, ylim, p_fig_sim, 2)


def calc_stat_mean_min_max(
    ds_l: [xr.Dataset],
    varidx_name: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate mean, minimum and maximum values within a group of datasets.
    TODO: Include coordinates in the returned datasets.

    Parameters
    ----------
    ds_l : [xr.Dataset]
        Array of datasets from a given group.
    varidx_name : str
        Climate variable or index code.

    Returns
    -------
    ds_mean_min_max : [xr.Dataset]
        Array of datasets with mean, minimum and maximum values.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_mean_min_max = []

    # Get years, units and coordinates.
    units = ds_l[0][varidx_name].attrs[cfg.attrs_units]
    year_1 = int(str(ds_l[0].time.values[0])[0:4])
    year_n = int(str(ds_l[0].time.values[len(ds_l[0].time.values) - 1])[0:4])
    n_time = year_n - year_1 + 1

    # Calculate statistics.
    arr_vals_mean = []
    arr_vals_min = []
    arr_vals_max = []
    for i_time in range(n_time):
        vals = []
        for ds in ds_l:
            val = float(ds[varidx_name][i_time].values)
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
        da = xr.DataArray(np.array(arr_vals), name=varidx_name, coords=[(cfg.dim_time, np.arange(n_time))])
        ds = da.to_dataset()
        ds[cfg.dim_time] = utils.reset_calendar(ds, year_1, year_n, cfg.freq_YS)
        ds[varidx_name].attrs[cfg.attrs_units] = units

        ds_mean_min_max.append(ds)

    return ds_mean_min_max


def calc_by_freq(
    ds: xr.Dataset,
    var: str,
    per: [int, int],
    freq: str
) -> List[xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate monthly values (mean, min, max).

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    var: str
        Climate variable.
    per: [int, int]
        Period of interest, for instance, [1981, 2010].
    freq: str
        Frequency.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Initialize array of results.
    ds_l = []

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
        if cfg.opt_ra:
            da_m = da_m.mean(dim={cfg.dim_longitude, cfg.dim_latitude})
        if freq != cfg.freq_MS:

            # Extract values.
            if var in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
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

                # No longitude and latitude for a station.
                if not cfg.opt_ra:
                    ds_m = ds_m.squeeze()

                ds_m[var].attrs[cfg.attrs_units] = ds[var].attrs[cfg.attrs_units]
                ds_l.append(ds_m)

        else:

            # Calculate statistics for each month-year combination.
            # Looping through years is less efficient but required to avoid a problem with incomplete at-a-station
            # datasets.
            arr_all = []
            for m in range(1, 13):
                vals_m = []
                for y in range(per[0], per[1] + 1):
                    if var in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                        val_m_y = np.nansum(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    else:
                        val_m_y = np.nanmean(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    vals_m.append(val_m_y)
                arr_all.append(vals_m)

            # Create dataset.
            dict_pd = {var: arr_all}
            df = pd.DataFrame(dict_pd)
            ds_l = df.to_xarray()
            ds_l.attrs[cfg.attrs_units] = ds[var].attrs[cfg.attrs_units]

    return ds_l


def calc_heatmap(
    varidx_code: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate heat map.

    Parameters
    ----------
    varidx_code: str
        Climate index code.
    --------------------------------------------------------------------------------------------------------------------
    """

    cat = cfg.cat_scen if varidx_code in cfg.variables_cordex else cfg.cat_idx
    varidx_name = varidx_code if cat == cfg.cat_scen else cfg.extract_idx(varidx_code)
    varidx_code_grp = cfg.get_idx_group(varidx_code)
    rcps = [cfg.rcp_ref, cfg.rcp_xx] + cfg.rcps

    msg = "Calculating maps."
    utils.log(msg, True)

    # Prepare data -----------------------------------------------------------------------------------------------------

    # Calculate the overall minimum and maximum values (considering all maps for the current 'varidx').
    # There is one z-value per period.
    n_per_hors = len(cfg.per_hors)
    z_min = [np.nan] * n_per_hors
    z_max = [np.nan] * n_per_hors
    z_min_delta = [np.nan] * n_per_hors
    z_max_delta = [np.nan] * n_per_hors

    # Calculated datasets and item description.
    arr_ds_maps = []
    arr_items = []

    # Build arrays for statistics to calculate.
    arr_stat = [cfg.stat_mean]
    arr_q = [-1]
    if cfg.opt_map_quantiles is not None:
        arr_stat = arr_stat + ([cfg.stat_quantile] * len(cfg.opt_map_quantiles))
        arr_q = arr_q + cfg.opt_map_quantiles

    # Reference map (to calculate deltas).
    ds_map_ref = None

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
            else:
                per_hors = cfg.per_hors
            for k_per_hor in range(len(per_hors)):
                per_hor = per_hors[k_per_hor]

                # Current map.
                ds_map = calc_heatmap_rcp(varidx_code, rcp, per_hor, stat, q)

                # Record current map, statistic and quantile, RCP and period.
                arr_ds_maps.append(ds_map)
                arr_items.append([stat, q, rcp, per_hor])

                # Calculate reference map.
                if (ds_map_ref is None) and\
                   (stat == cfg.stat_mean) and (rcp == cfg.rcp_xx) and (per_hor == cfg.per_ref):
                    ds_map_ref = ds_map

                # Extract values.
                vals = ds_map[varidx_name].values
                vals_delta = None
                if per_hor != cfg.per_ref:
                    vals_delta = ds_map[varidx_name].values - ds_map_ref[varidx_name].values

                # Record mean values.
                # Mean absolute values.
                z_min_j = np.nanmin(vals)
                z_max_j = np.nanmax(vals)
                z_min[k_per_hor] = z_min_j if np.isnan(z_min[k_per_hor]) else min(z_min[k_per_hor], z_min_j)
                z_max[k_per_hor] = z_max_j if np.isnan(z_max[k_per_hor]) else max(z_max[k_per_hor], z_max_j)
                # Mean delta values.
                if vals_delta is not None:
                    z_min_delta_j = np.nanmin(vals_delta)
                    z_max_delta_j = np.nanmax(vals_delta)
                    z_min_delta[k_per_hor] = z_min_delta_j if np.isnan(z_min_delta[k_per_hor])\
                        else min(z_min_delta[k_per_hor], z_min_delta_j)
                    z_max_delta[k_per_hor] = z_max_delta_j if np.isnan(z_max_delta[k_per_hor])\
                        else max(z_max_delta[k_per_hor], z_max_delta_j)

    # Generate maps ----------------------------------------------------------------------------------------------------

    stn = "stns" if not cfg.opt_ra else cfg.obs_src

    # Loop through maps.
    for i in range(len(arr_ds_maps)):

        # Current map.
        ds_map = arr_ds_maps[i]

        # Current RCP and horizon.
        stat      = arr_items[i][0]
        q         = arr_items[i][1]
        rcp       = arr_items[i][2]
        per_hor   = arr_items[i][3]
        i_per_hor = cfg.per_hors.index(per_hor)

        # Perform twice (for values, then for deltas).
        for j in range(2):

            # Skip if delta map generation if the option is disabled or if it's the reference period.
            if (j == 1) and\
                ((per_hor == cfg.per_ref) or
                 ((cat == cfg.cat_scen) and (not cfg.opt_map_delta[0])) or
                 ((cat == cfg.cat_idx) and (not cfg.opt_map_delta[1]))):
                continue

            # Extract dataset (calculate delta if required).
            if j == 0:
                da_map = ds_map[varidx_name]
            else:
                da_map = ds_map[varidx_name] - ds_map_ref[varidx_name]

            # Perform twice (for min/max values specific to a period, then for overall min/max values).
            for k in range(2):

                # Specific period.
                if k == 0:
                    per_str = str(per_hor[0]) + "-" + str(per_hor[1])
                    z_min_map = z_min[i_per_hor] if j == 0 else z_min_delta[i_per_hor]
                    z_max_map = z_max[i_per_hor] if j == 0 else z_max_delta[i_per_hor]
                # All periods combined.
                else:
                    per_str = str(cfg.per_ref[0]) + "-" + str(cfg.per_hors[len(cfg.per_hors) - 1][1])
                    z_min_map = np.nanmin(z_min if j == 0 else z_min_delta)
                    z_max_map = np.nanmax(z_max if j == 0 else z_max_delta)

                # Plots ------------------------------------------------------------------------------------------------

                # Path.
                d_fig = cfg.get_d_scen(stn, cfg.cat_fig + cfg.sep + cat + cfg.sep + "maps", varidx_code_grp + cfg.sep +
                                       varidx_code + cfg.sep + per_str)
                if stat in [cfg.stat_mean, cfg.stat_min, cfg.stat_max]:
                    stat_str = "_" + stat
                else:
                    stat_str = "_q" + str(int(q*100)).rjust(2, "0")
                fn_fig = varidx_name + "_" + rcp + "_" + str(per_hor[0]) + "_" + str(per_hor[1]) + stat_str +\
                    cfg.f_ext_png
                p_fig = d_fig + fn_fig
                if j == 1:
                    p_fig = p_fig.replace(cfg.f_ext_png, "_delta" + cfg.f_ext_png)

                # Generate plot.
                if ((cat == cfg.cat_scen) and (cfg.opt_map[0])) or ((cat == cfg.cat_idx) and (cfg.opt_map[1])):
                    plot.plot_heatmap(da_map, stn, varidx_code, rcp, per_hor, stat, q, z_min_map, z_max_map, j == 1,
                                      p_fig)

                # CSV files --------------------------------------------------------------------------------------------

                # Path.
                d_csv = cfg.get_d_scen(stn, cfg.cat_fig + cfg.sep + cat + cfg.sep + "maps", varidx_code_grp + cfg.sep +
                                       varidx_code + "_csv" + cfg.sep + per_str)
                fn_csv = fn_fig.replace(cfg.f_ext_png, cfg.f_ext_csv)
                p_csv = d_csv + fn_csv
                if j == 1:
                    p_csv = p_csv.replace(cfg.f_ext_csv, "_delta" + cfg.f_ext_csv)

                # Save.
                if cfg.opt_save_csv and (not os.path.exists(p_csv) or cfg.opt_force_overwrite):

                    # Extract data.
                    arr_lon = []
                    arr_lat = []
                    arr_val = []
                    for m in range(len(da_map.longitude.values)):
                        for n in range(len(da_map.latitude.values)):
                            arr_lon.append(da_map.longitude.values[m])
                            arr_lat.append(da_map.latitude.values[n])
                            arr_val.append(da_map.values[n, m])

                    # Build dataframe.
                    dict_pd = {cfg.dim_longitude: arr_lon, cfg.dim_latitude: arr_lat, varidx_name: arr_val}
                    df = pd.DataFrame(dict_pd)

                    # Save to file.
                    utils.save_csv(df, p_csv)


def calc_heatmap_rcp(
    varidx_code: str,
    rcp: str,
    per: [int],
    stat: str,
    q: float
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate heat map for a given RCP.

    Parameters
    ----------
    varidx_code: str
        Climate variable or index code.
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

    cat = cfg.cat_scen if varidx_code in cfg.variables_cordex else cfg.cat_idx
    varidx_name = varidx_code if cat == cfg.cat_scen else cfg.extract_idx(varidx_code)
    varidx_code_grp = cfg.get_idx_group(varidx_code)

    utils.log("Processing: " + varidx_code + ", " +
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
            ds_stat = calc_stat(cfg.cat_obs if rcp == cfg.rcp_ref else cfg.cat_scen, cfg.freq_YS, cfg.freq_YS, stn,
                                varidx_code, rcp, None, False, cfg.opt_map_clip, stat, q)
            if ds_stat is None:
                continue

            # Extract data from stations.
            data = [[], [], []]
            n = ds_stat.dims[cfg.dim_time]
            for year in range(0, n):

                # Collect data.
                x = float(lon)
                y = float(lat)
                z = float(ds_stat[varidx_name][0][0][year])
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

        def round_to_nearest_decimal(val_inner, step):
            if val_inner < 0:
                val_rnd = math.floor(val_inner/step) * step
            else:
                val_rnd = math.ceil(val_inner/step) * step
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
        ds_res = da_itp.to_dataset(name=varidx_name)

    # There is no need to interpolate.
    else:

        # Reference period.
        if varidx_name in cfg.variables_cordex:
            p_sim_ref = cfg.get_d_scen(cfg.obs_src, cfg.cat_obs, varidx_code_grp) +\
                varidx_code_grp + "_" + cfg.obs_src + cfg.f_ext_nc
        else:
            p_sim_ref = cfg.get_d_idx(cfg.obs_src, varidx_code_grp) + varidx_code_grp + "_ref" + cfg.f_ext_nc
        if rcp == cfg.rcp_ref:

            # Open dataset.
            p_sim = p_sim_ref
            if not os.path.exists(p_sim):
                return xr.Dataset(None)
            ds_sim = utils.open_netcdf(p_sim)

            # TODO: The following patch was added to fix dimensions for cfg.idx_drought_code.
            if varidx_code in cfg.idx_codes:
                ds_sim = ds_sim.resample(time=cfg.freq_YS).mean(dim=cfg.dim_time, keep_attrs=True)

            # Select period.
            ds_sim = utils.remove_feb29(ds_sim)
            ds_sim = utils.sel_period(ds_sim, per)

            # Extract units.
            units = 1
            if cfg.attrs_units in ds_sim[varidx_name].attrs:
                units = ds_sim[varidx_name].attrs[cfg.attrs_units]
            elif cfg.attrs_units in ds_sim.data_vars:
                units = ds_sim[cfg.attrs_units]

            # Calculate mean.
            ds_res = ds_sim.mean(dim=cfg.dim_time)
            if varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                ds_res = ds_res * 365
            if cat == cfg.cat_scen:
                ds_res[varidx_name].attrs[cfg.attrs_units] = units
            else:
                ds_res.attrs[cfg.attrs_units] = units

        # Future period.
        else:

            # List scenarios or indices for the current RCP.
            if cat == cfg.cat_scen:
                d = cfg.get_d_scen(cfg.obs_src, cfg.cat_qqmap, varidx_code_grp)
            else:
                d = cfg.get_d_scen(cfg.obs_src, cfg.cat_idx, varidx_code_grp)
            p_sim_l = [i for i in glob.glob(d + "*" + cfg.f_ext_nc) if i != p_sim_ref]

            # Collect simulations.
            arr_sim = []
            ds_res  = None
            n_sim   = 0
            units   = 1
            for p_sim in p_sim_l:
                if os.path.exists(p_sim) and ((rcp in p_sim) or (rcp == cfg.rcp_xx)):

                    # Open dataset.
                    ds_sim = utils.open_netcdf(p_sim)

                    # TODO: The following patch was added to fix dimensions for cfg.idx_drought_code.
                    if varidx_code in cfg.idx_codes:
                        ds_sim = ds_sim.resample(time=cfg.freq_YS).mean(dim=cfg.dim_time, keep_attrs=True)

                    # Extract units.
                    if cfg.attrs_units in ds_sim[varidx_name].attrs:
                        units = ds_sim[varidx_name].attrs[cfg.attrs_units]
                    elif cfg.attrs_units in ds_sim.data_vars:
                        units = ds_sim[cfg.attrs_units]

                    # Select period.
                    ds_sim = utils.remove_feb29(ds_sim)
                    ds_sim = utils.sel_period(ds_sim, per)

                    # Calculate mean and add to array.
                    ds_sim = ds_sim.mean(dim=cfg.dim_time)
                    if varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                        ds_sim = ds_sim * 365
                    if cat == cfg.cat_scen:
                        ds_sim[varidx_name].attrs[cfg.attrs_units] = units
                    else:
                        ds_sim.attrs[cfg.attrs_units] = units
                    arr_sim.append(ds_sim)

                    # The first dataset will be used to return result.
                    if n_sim == 0:
                        ds_res = ds_sim.copy(deep=True)
                        ds_res[varidx_name][:, :] = np.nan
                    n_sim = n_sim + 1

            if ds_res is None:
                return xr.Dataset(None)

            # Get longitude and latitude.
            lon_l, lat_l = utils.get_coordinates(arr_sim[0][varidx_name], True)

            # Calculate statistics.
            len_lat = len(lat_l)
            len_lon = len(lon_l)
            for i_lat in range(len_lat):
                for j_lon in range(len_lon):

                    # Extract the value for the current cell for all simulations.
                    vals = []
                    for ds_sim_k in arr_sim:
                        dims_sim = ds_sim_k[varidx_name].dims
                        if cfg.dim_rlat in dims_sim:
                            val_sim = float(ds_sim_k[varidx_name].isel(rlon=j_lon, rlat=i_lat))
                        elif cfg.dim_lat in dims_sim:
                            val_sim = float(ds_sim_k[varidx_name].isel(lon=j_lon, lat=i_lat))
                        else:
                            val_sim = float(ds_sim_k[varidx_name].isel(longitude=j_lon, latitude=i_lat))
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
                    ds_res[varidx_name][i_lat, j_lon] = val

        # Remember units.
        units = ds_res[varidx_name].attrs[cfg.attrs_units] if cat == cfg.cat_scen else ds_res.attrs[cfg.attrs_units]

        # Convert units.
        if (varidx_name in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
           (ds_res[varidx_name].attrs[cfg.attrs_units] == cfg.unit_K):
            ds_res = ds_res - cfg.d_KC
            ds_res[varidx_name].attrs[cfg.attrs_units] = cfg.unit_C
        elif (varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]) and \
             (ds_res[varidx_name].attrs[cfg.attrs_units] == cfg.unit_kg_m2s1):
            ds_res = ds_res * cfg.spd
            ds_res[varidx_name].attrs[cfg.attrs_units] = cfg.unit_mm
        elif varidx_name in [cfg.var_cordex_uas, cfg.var_cordex_vas, cfg.var_cordex_sfcwindmax]:
            ds_res = ds_res * cfg.km_h_per_m_s
            ds_res[varidx_name].attrs[cfg.attrs_units] = cfg.unit_km_h
        else:
            ds_res[varidx_name].attrs[cfg.attrs_units] = units

        # Adjust coordinate names (required for clipping).
        if cfg.dim_longitude not in list(ds_res.dims):
            if cfg.dim_rlon in ds_res.dims:
                ds_res = ds_res.rename_dims({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})
                ds_res[cfg.dim_longitude] = ds_res[cfg.dim_rlon]
                del ds_res[cfg.dim_rlon]
                ds_res[cfg.dim_latitude] = ds_res[cfg.dim_rlat]
                del ds_res[cfg.dim_rlat]
            else:
                ds_res = ds_res.rename_dims({cfg.dim_lon: cfg.dim_longitude, cfg.dim_lat: cfg.dim_latitude})
                ds_res[cfg.dim_longitude] = ds_res[cfg.dim_lon]
                del ds_res[cfg.dim_lon]
                ds_res[cfg.dim_latitude] = ds_res[cfg.dim_lat]
                del ds_res[cfg.dim_lat]

    return ds_res


def conv_nc_csv(
    cat: str
):

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
        cat_l = [cfg.cat_obs, cfg.cat_raw, cfg.cat_regrid, cfg.cat_qqmap]
        if cat == cfg.cat_idx:
            cat_l = [cfg.cat_idx]
        for cat in cat_l:

            # Explode lists of codes and names.
            idx_names_exploded = cfg.explode_idx_l(cfg.idx_names)
            idx_codes_exploded = cfg.explode_idx_l(cfg.idx_codes)

            # Loop through variables or indices.
            varidx_name_l = cfg.variables_cordex if cat != cfg.cat_idx else idx_names_exploded
            for i_varidx_name in range(len(varidx_name_l)):
                varidx_name = idx_names_exploded[i_varidx_name]
                varidx_code = idx_codes_exploded[i_varidx_name]
                varidx_code_grp = cfg.get_idx_group(varidx_code)

                # List NetCDF files.
                p_l = list(glob.glob(cfg.get_d_scen(stn, cat, varidx_code_grp) + "*" + cfg.f_ext_nc))
                n_files = len(p_l)
                if n_files == 0:
                    continue
                p_l.sort()

                utils.log("Processing: " + stn + ", " + varidx_code, True)

                # Scalar processing mode.
                if cfg.n_proc == 1:
                    for i_file in range(n_files):
                        conv_nc_csv_single(p_l, varidx_code, i_file)

                # Parallel processing mode.
                else:

                    # Loop until all files have been converted.
                    while True:

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before =\
                            len(list(glob.glob(cfg.get_d_scen(stn, cat, varidx_code) + "*" + cfg.f_ext_csv)))

                        try:
                            utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=min(cfg.n_proc, len(p_l)))
                            func = functools.partial(conv_nc_csv_single, p_l, varidx_name)
                            pool.map(func, list(range(n_files)))
                            pool.close()
                            pool.join()
                            utils.log("Fork ended.", True)
                        except Exception as e:
                            utils.log(str(e))
                            pass

                        # Calculate the number of files processed (after conversion).
                        n_files_proc_after =\
                            len(list(glob.glob(cfg.get_d_scen(stn, cat, varidx_code_grp) + "*" + cfg.f_ext_csv)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_files_proc_before == n_files_proc_after):
                            break


def conv_nc_csv_single(
    p_l: [str],
    varidx_code: str,
    i_file: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a single NetCDF to CSV file.

    Parameters
    ----------
    p_l : [str]
        List of paths.
    varidx_code : str
        Climate variable or index code.
    i_file : int
        Rank of file in 'p_l'.
    --------------------------------------------------------------------------------------------------------------------
    """

    varidx_name = varidx_code if (varidx_code in cfg.variables_cordex) else cfg.extract_idx(varidx_code)
    varidx_code_grp = cfg.get_idx_group(varidx_code)

    # Paths.
    p = p_l[i_file]
    p_csv = p.replace(cfg.f_ext_nc, cfg.f_ext_csv).\
        replace(cfg.sep + varidx_code_grp + "_", cfg.sep + varidx_name + "_").\
        replace(cfg.sep + varidx_code_grp + cfg.sep, cfg.sep + varidx_code_grp + "_" + cfg.f_csv + cfg.sep)

    if os.path.exists(p_csv) and (not cfg.opt_force_overwrite):
        return()

    # Explode lists of codes and names.
    idx_names_exploded = cfg.explode_idx_l(cfg.idx_names)

    # Open dataset.
    ds = xr.open_dataset(p)

    # Extract time.
    time_l = list(ds.time.values)
    n_time = len(time_l)
    for i in range(n_time):
        if varidx_name not in idx_names_exploded:
            time_l[i] = str(time_l[i])[0:10]
        else:
            time_l[i] = str(time_l[i])[0:4]

    # Extract longitude and latitude.
    # Calculate average values (only if the analysis is based on observations at a station).
    lon_l = None
    lat_l = None
    if cfg.opt_ra:
        lon_l, lat_l = utils.get_coordinates(ds, True)

    # Extract values.
    # Calculate average values (only if the analysis is based on observations at a station).
    val_l = list(ds[varidx_name].values)
    if not cfg.opt_ra:
        if varidx_name not in cfg.idx_names:
            for i in range(n_time):
                val_l[i] = val_l[i].mean()
        else:
            if cfg.rcp_ref in p:
                for i in range(n_time):
                    val_l[i] = val_l[i][0][0]
            else:
                val_l = list(val_l[0][0])

    # Convert values to more practical units (if required).
    if (varidx_name in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]) and\
       (ds[varidx_name].attrs[cfg.attrs_units] == cfg.unit_kg_m2s1):
        for i in range(n_time):
            val_l[i] = val_l[i] * cfg.spd
    elif (varidx_name in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
         (ds[varidx_name].attrs[cfg.attrs_units] == cfg.unit_K):
        for i in range(n_time):
            val_l[i] = val_l[i] - cfg.d_KC

    # Build pandas dataframe.
    if not cfg.opt_ra:
        dict_pd = {cfg.dim_time: time_l, varidx_name: val_l}
    else:
        time_l_1d = []
        lon_l_1d  = []
        lat_l_1d  = []
        val_l_1d  = []
        for i in range(len(time_l)):
            for j in range(len(lon_l)):
                for k in range(len(lat_l)):
                    time_l_1d.append(time_l[i])
                    lon_l_1d.append(lon_l[j])
                    lat_l_1d.append(lat_l[k])
                    val_l_1d.append(list(val_l)[i][j][k])
        dict_pd = {cfg.dim_time: time_l_1d, cfg.dim_lon: lon_l_1d, cfg.dim_lat: lat_l_1d, varidx_name: val_l_1d}
    df = pd.DataFrame(dict_pd)

    # Save CSV file.
    utils.save_csv(df, p_csv)
