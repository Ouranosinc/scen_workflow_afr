# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Statistics functions.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------
from pandas import DataFrame

import constants as const
import functools
import file_utils as fu
import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
# Do not delete the line below: 'rioxarray' must be added even if it's not explicitly used in order to have a 'rio'
# variable in DataArrays.
import rioxarray as rio
import utils
import warnings
import xarray as xr
import xesmf as xe
from config import cfg
from pandas.core.common import SettingWithCopyWarning
from scipy.interpolate import griddata
from streamlit import caching
from typing import Union, List, Tuple, Optional

import sys
sys.path.append("dashboard")
from dashboard import def_context, def_delta, def_project, def_hor, def_lib, def_rcp, def_sim, def_stat,\
    def_varidx as vi, def_view, dash_plot


def calc_stat(
    data_type: str,
    freq_in: str,
    freq_out: str,
    stn: str,
    vi_code: str,
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
        Dataset type: {const.obs, const.cat_scen}
    freq_in : str
        Frequency (input): const.freq_D=daily; const.freq_YS=annual
    freq_out : str
        Frequency (output): const.freq_D=daily; const.freq_YS=annual
    stn : str
        Station.
    vi_code : str
        Climate variable or index code.
    rcp : str
        Emission scenario: {def_rcp.rcp_26, def_rcp.rcp_45, def_rcp, def_rcp.rcp_xx}
    hor : [int]
        Horizon: ex: [1981, 2010]
        If None is specified, the complete time range is considered.
    per_region : bool
        If True, statistics are calculated for a region as a whole.
    clip : bool
        If True, clip according to 'cfg.p_bounds'.
    stat : str
        Statistic: {def_stat.code_mean, def_stat.code_min, def_stat.code_max, def_stat.code_quantile}
    q : float, optional
        Quantile: value between 0 and 1.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Create variable instance.
    varidx = vi.VarIdx(vi_code)

    # Extract name and group.
    cat = const.cat_scen if vi_code in cfg.variables else const.cat_idx
    vi_name = vi_code if cat == const.cat_scen else varidx.get_name()
    vi_code_grp = vi.get_group(vi_code)

    # List files.
    if data_type == const.cat_obs:
        if vi_name in cfg.variables:
            p_sim_l = [cfg.get_d_scen(stn, const.cat_obs, vi_name) + vi_name + "_" + stn + fu.f_ext_nc]
        else:
            p_sim_l = [cfg.get_d_idx(stn, vi_code_grp) + vi_code_grp + "_ref" + fu.f_ext_nc]
    else:
        if vi_name in cfg.variables:
            d = cfg.get_d_scen(stn, const.cat_qqmap, vi_name)
        else:
            d = cfg.get_d_idx(stn, vi_code_grp)
        p_sim_l = glob.glob(d + "*_" + ("*rcp*" if rcp == def_rcp.rcp_xx else rcp) + fu.f_ext_nc)

    # Exit if there is no file corresponding to the criteria.
    if (len(p_sim_l) == 0) or \
       ((len(p_sim_l) > 0) and not(os.path.isdir(os.path.dirname(p_sim_l[0])))):
        return None

    # List days.
    ds = fu.open_netcdf(p_sim_l[0])
    if (const.dim_lon in ds.variables) or (const.dim_lon in ds.dims):
        lon = ds[const.dim_lon]
        lat = ds[const.dim_lat]
    elif (const.dim_rlon in ds.variables) or (const.dim_rlon in ds.dims):
        lon = ds[const.dim_rlon]
        lat = ds[const.dim_rlat]
    else:
        lon = ds[const.dim_longitude]
        lat = ds[const.dim_latitude]
    if const.attrs_units in ds[vi_name].attrs:
        units = ds[vi_name].attrs[const.attrs_units]
    else:
        units = 1
    n_sim = len(p_sim_l)

    # First and last years.
    year_1 = int(str(ds.time.values[0])[0:4])
    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
    if not(hor is None):
        year_1 = max(year_1, hor[0])
        year_n = min(year_n, hor[1])
    n_time = (year_n - year_1 + 1) * (365 if freq_in == const.freq_D else 1)

    # Collect values from simulations.
    arr_vals = []
    for i_sim in range(n_sim):

        # Load dataset.
        ds = fu.open_netcdf(p_sim_l[i_sim])

        # Patch used to fix potentially unordered dimensions of index 'cfg.idx_drought_code'.
        if vi_code in cfg.idx_codes:
            ds = ds.resample(time=const.freq_YS).mean(dim=const.dim_time, keep_attrs=True)

        # Select years and adjust units.
        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
        ds = ds.sel(time=slice(years_str[0], years_str[1]))
        if const.attrs_units in ds[vi_name].attrs:
            if (vi_name in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]) and\
               (ds[vi_name].attrs[const.attrs_units] == const.unit_K):
                ds = ds - const.d_KC
                ds[vi_name].attrs[const.attrs_units] = const.unit_C
            elif vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
                ds = ds * const.spd
                ds[vi_name].attrs[const.attrs_units] = const.unit_mm
            elif vi_name in [vi.v_uas, vi.v_vas, vi.v_sfcwindmax]:
                ds = ds * const.km_h_per_m_s
                ds[vi_name].attrs[const.attrs_units] = const.unit_km_h

        if cfg.opt_ra:

            # Statistics are calculated at a point.
            if cfg.p_bounds == "":
                ds = utils.subset_ctrl_pt(ds)

            # Statistics are calculated on a surface.
            else:

                # Clip to geographic boundaries.
                if clip and (cfg.p_bounds != ""):
                    ds = utils.subset_shape(ds)

                # Calculate mean value.
                ds = utils.squeeze_lon_lat(ds)

        # Records values.
        # Simulation data is assumed to be complete.
        if ds[vi_name].time.size == n_time:
            if (const.dim_lon in ds.dims) or (const.dim_rlon in ds.dims) or (const.dim_longitude in ds.dims):
                vals = ds.squeeze()[vi_name].values.tolist()
            else:
                vals = ds[vi_name].values.tolist()
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
                            if ds_i[vi_name].size != 0:
                                day_of_year = ds_i[vi_name].time.dt.dayofyear.values[0]
                                if len(np.array(ds_i[vi_name].values).shape) > 1:
                                    val = ds_i[vi_name].values[0][0][0]
                                else:
                                    val = ds_i[vi_name].values[0]
                                vals[(i_year - year_1) * 365 + day_of_year - 1] = val
                        except:
                            pass
            if cfg.opt_ra and (vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]):
                vals = [i * const.spd for i in vals]
            arr_vals.append(vals)

    # Calculate the mean value of all years.
    if per_region:
        for i in range(len(arr_vals)):
            arr_vals[i] = [np.nanmean(arr_vals[i])]
        n_time = 1

    # Transpose.
    arr_vals_t = []

    # Collapse to yearly frequency.
    if (freq_in == const.freq_D) and (freq_out == const.freq_YS) and cfg.opt_ra:
        for i_sim in range(n_sim):
            vals_sim = []
            for i_year in range(0, year_n - year_1 + 1):
                vals_year = arr_vals[i_sim][(365 * i_year):(365 * (i_year + 1))]
                da_vals = xr.DataArray(np.array(vals_year))
                vals_year = list(da_vals[np.isnan(da_vals.values) == False].values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
                        val_year = np.nansum(vals_year)
                    else:
                        val_year = np.nanmean(vals_year)
                vals_sim.append(val_year)
            if (vi_name != vi.v_pr) | (sum(vals_sim) > 0):
                arr_vals_t.append(vals_sim)
        arr_vals = arr_vals_t
        n_time = year_n - year_1 + 1
    n_sim = len(arr_vals)

    # Transpose array.
    arr_vals_t = []
    for i_time in range(n_time):
        vals = []
        for i_sim in range(n_sim):
            if (vi_name != vi.i_rain_season_prcptot) or\
               ((vi_name == vi.i_rain_season_prcptot) and (max(arr_vals[i_sim]) > 0)):
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
                                  ((vi_name != vi.v_pr) | (sum(da_vals.values) > 0))].values)
        if (stat == def_stat.code_min) or (q == 0):
            val_stat = np.min(vals_t)
        elif (stat == def_stat.code_max) or (q == 1):
            val_stat = np.max(vals_t)
        elif stat == def_stat.code_mean:
            val_stat = np.mean(vals_t)
        elif stat == def_stat.code_quantile:
            val_stat = np.quantile(vals_t, q)
        elif stat == def_stat.code_sum:
            val_stat = np.sum(vals_t)
        arr_stat.append(val_stat)

    # Build dataset.
    da_stat = xr.DataArray(np.array(arr_stat), name=vi_name, coords=[(const.dim_time, np.arange(n_time))])
    ds_stat = da_stat.to_dataset()
    ds_stat = ds_stat.expand_dims(lon=1, lat=1)

    # Adjust coordinates and time.
    if not(const.dim_lon in ds_stat.dims):
        ds_stat[const.dim_lon] = lon
        ds_stat[const.dim_lat] = lat
    if freq_out == const.freq_YS:
        ds_stat[const.dim_time] = utils.reset_calendar(ds_stat, year_1, year_n, freq_out)

    # Adjust units.
    ds_stat[vi_name].attrs[const.attrs_units] = units

    return ds_stat


def calc_stats(
    vi_code_l: List[str],
    i_vi_proc: int,
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate statistics.

    Parameters
    ----------
    vi_code_l: List[str],
        Variables or index codes.
    i_vi_proc: int
        Rank of variable or index to process.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get variable or index code.
    vi_code = vi_code_l[i_vi_proc]

    cat = const.cat_scen if vi_code in cfg.variables else const.cat_idx
    if cat == const.cat_scen:
        vi_code_l = [vi_code]
        vi_name_l = [vi_code]
    else:
        vi_code_l = vi.explode_idx_l([vi_code])
        vi_name_l = vi.explode_idx_l(vi.VarIdx(vi_code).get_name())

    # List emission scenarios.
    rcps = [def_rcp.rcp_ref] + cfg.rcps
    if len(rcps) > 2:
        rcps = rcps + [def_rcp.rcp_xx]

    # Data frequency.
    freq = const.freq_D if cat == const.cat_scen else const.freq_YS

    # Loop through stations.
    stns = (cfg.stns if not cfg.opt_ra else [cfg.obs_src])
    for stn in stns:

        # Loop through variables or indices.
        # There should be a single item in the list, unless 'vi_code' is a group of indices.
        for i_vi in range(len(vi_name_l)):
            vi_code = vi_code_l[i_vi]
            vi_name = vi_name_l[i_vi]
            vi_code_grp = vi.get_group(vi_code)

            # Skip iteration if the file already exists.
            p_csv = cfg.get_d_scen(stn, const.cat_stat, cat + cfg.sep + vi_code_grp) + vi_code + fu.f_ext_csv
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
                if rcp == def_rcp.rcp_ref:
                    hors = [cfg.per_ref]
                    cat_rcp = const.cat_obs
                    if cat == const.cat_scen:
                        d = os.path.dirname(cfg.get_p_obs(stn, vi_code_grp))
                    else:
                        d = cfg.get_d_idx(stn, vi_code_grp)
                else:
                    hors = cfg.per_hors
                    if cat == const.cat_scen:
                        cat_rcp = const.cat_scen
                        d = cfg.get_d_scen(stn, const.cat_qqmap, vi_code_grp)
                    else:
                        cat_rcp = const.cat_idx
                        d = cfg.get_d_idx(stn, vi_code_grp)

                if not os.path.isdir(d):
                    continue

                idx_desc = vi_name
                if vi_code_grp != vi_name:
                    idx_desc = vi_code_grp + "." + idx_desc
                fu.log("Processing: " + stn + ", " + idx_desc + ", " + rcp + "", True)

                # Loop through statistics.
                stats = [def_stat.code_mean]
                if cat_rcp != const.cat_obs:
                    stats = stats + [def_stat.code_min, def_stat.code_max, def_stat.code_quantile]
                for stat in stats:
                    stat_quantiles = cfg.opt_stat_quantiles
                    if stat != def_stat.code_quantile:
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
                                calc_stat(cat_rcp, freq, const.freq_YS, stn, vi_code, rcp, hor,
                                          (freq == const.freq_YS) and (cat == const.cat_scen),
                                          cfg.opt_stat_clip, stat, q)
                            if ds_stat is None:
                                continue

                            # Select period.
                            if cfg.opt_ra:
                                ds_stat_hor = utils.sel_period(ds_stat.squeeze(), hor)
                            else:
                                ds_stat_hor = ds_stat.copy(deep=True)

                            # Extract value.
                            if vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
                                val = float(ds_stat_hor[vi_name].sum()) / (hor[1] - hor[0] + 1)
                            else:
                                val = float(ds_stat_hor[vi_name].mean())

                            # Add row.
                            stn_l.append(stn)
                            rcp_l.append(rcp)
                            hor_l.append(str(hor[0]) + "-" + str(hor[1]))
                            if cat_rcp == const.cat_obs:
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
                    {"stn": stn_l, ("var" if cat == const.cat_scen else "idx"): [vi_name] * len(stn_l),
                     "rcp": rcp_l, "hor": hor_l, "stat": stat_l, "q": q_l, "val": val_l}
                df = pd.DataFrame(dict_pd)

                # Save file.
                fu.save_csv(df, p_csv)


def calc_ts(
    view_code: str,
    vi_code_l: List[str],
    i_vi_proc: int,
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Plot time series for individual simulations.

    Parameters
    ----------
    view_code : str
        View code.
    vi_code_l: List[str],
        Variables or index codes.
    i_vi_proc: int
        Rank of variable or index to process.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get variable or index code.
    vi_code = vi_code_l[i_vi_proc]

    cat = const.cat_scen if vi_code in cfg.variables else const.cat_idx

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Extract information about the variable of index.
        varidx = vi.VarIdx(vi_code)
        vi_name = str(varidx.get_name())
        vi_code_grp = vi.get_group(vi_code)

        # Status message.
        msg = vi_code
        if vi_code_grp != vi_code:
            msg = vi_code_grp + "." + msg
        fu.log("Processing: " + stn + ", " + msg, True)

        # Path of files to be created.
        cat_fig = const.cat_fig + cfg.sep + cat + cfg.sep + view_code
        fn = vi_name + "_" + dash_plot.mode_rcp
        # CSV files:
        p_rcp_csv = cfg.get_d_scen(stn, cat_fig, vi_code_grp + "_" + fu.f_csv) + fn + fu.f_ext_csv
        p_sim_csv = p_rcp_csv.replace("_" + dash_plot.mode_rcp, "_" + dash_plot.mode_sim)
        p_rcp_del_csv = p_rcp_csv.replace(fu.f_ext_csv, "_delta" + fu.f_ext_csv)
        p_sim_del_csv = p_sim_csv.replace(fu.f_ext_csv, "_delta" + fu.f_ext_csv)
        # PNG files.
        p_rcp_fig = cfg.get_d_scen(stn, cat_fig, vi_code_grp) + fn + fu.f_ext_png
        p_sim_fig = p_rcp_fig.replace("_" + dash_plot.mode_rcp + fu.f_ext_png,
                                      "_" + dash_plot.mode_sim + fu.f_ext_png)
        p_rcp_del_fig = p_rcp_fig.replace(fu.f_ext_png, "_delta" + fu.f_ext_png)
        p_sim_del_fig = p_sim_fig.replace(fu.f_ext_png, "_delta" + fu.f_ext_png)

        # Skip if no work required.
        save_csv = (not os.path.exists(p_rcp_csv) or not os.path.exists(p_sim_csv) or
                    not os.path.exists(p_rcp_del_csv) or not os.path.exists(p_sim_del_csv)) and \
                   (cfg.opt_save_csv[0 if cat == const.cat_scen else 1])
        save_fig = (not os.path.exists(p_rcp_fig) or not os.path.exists(p_sim_fig) or
                    not os.path.exists(p_rcp_del_fig) or not os.path.exists(p_sim_del_fig))
        if (not save_csv) and (not save_fig) and (not cfg.opt_force_overwrite):
            continue

        # Create context.
        cntx = def_context.Context(def_context.code_script)
        cntx.view = def_view.View(view_code)
        cntx.lib = def_lib.Lib(def_lib.mode_mat)
        cntx.varidx = vi.VarIdx(vi_code)
        cntx.rcps = def_rcp.RCPs(cfg.rcps)
        cntx.rcp = def_rcp.RCP("")
        cntx.sim = def_sim.Sim("")

        # Attempt loading CSV files into dataframes.
        if os.path.exists(p_rcp_csv) and os.path.exists(p_sim_csv) and \
           os.path.exists(p_rcp_del_csv) and os.path.exists(p_sim_del_csv):
            df_rcp = pd.read_csv(p_rcp_csv)
            df_sim = pd.read_csv(p_sim_csv)
            df_rcp_del = pd.read_csv(p_rcp_del_csv)
            df_sim_del = pd.read_csv(p_sim_del_csv)

        # Generate dataframes.
        else:
            df_rcp, df_sim, df_rcp_del, df_sim_del = calc_ts_prep(cntx, stn)

        # CSV file format ------------------------------------------------------------------------------------------

        if fu.f_csv in cfg.opt_ts_format:
            if (not os.path.exists(p_rcp_csv)) or cfg.opt_force_overwrite:
                fu.save_csv(df_rcp, p_rcp_csv)
            if (not os.path.exists(p_sim_csv)) or cfg.opt_force_overwrite:
                fu.save_csv(df_sim, p_sim_csv)
            if (not os.path.exists(p_rcp_del_csv)) or cfg.opt_force_overwrite:
                fu.save_csv(df_rcp_del, p_rcp_del_csv)
            if (not os.path.exists(p_sim_del_csv)) or cfg.opt_force_overwrite:
                fu.save_csv(df_sim_del, p_sim_del_csv)

        # PNG file format ------------------------------------------------------------------------------------------

        if fu.f_png in cfg.opt_ts_format:

            # Loop through modes (rcp|sim).
            for mode in [dash_plot.mode_rcp, dash_plot.mode_sim]:

                # Loop through delta.
                for delta in [False, True]:
                    cntx.delta = def_delta.Del(delta)

                    # Select dataset and output file name.
                    if mode == dash_plot.mode_rcp:
                        if not delta:
                            df_mode = df_rcp
                            p_fig = p_rcp_fig
                        else:
                            df_mode = df_rcp_del
                            p_fig = p_rcp_del_fig
                    else:
                        if not delta:
                            df_mode  = df_sim
                            p_fig = p_sim_fig
                        else:
                            df_mode = df_sim_del
                            p_fig = p_sim_del_fig

                    # Generate plot.
                    if (not os.path.exists(p_fig)) or cfg.opt_force_overwrite:
                        fig = dash_plot.gen_ts(cntx, df_mode, mode)
                        fu.save_plot(fig, p_fig)


def calc_ts_prep(
    cntx: def_context.Context,
    stn: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Prepare datasets for the generation of  time series.

    Parameters
    ----------
    cntx : def_context.Context
        Context.
    stn : str
        Station.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Two dataframes (rcp|sim).
    --------------------------------------------------------------------------------------------------------------------
    """

    df_rcp, df_sim, df_rcp_del, df_sim_del = None, None, None, None

    # Extract relevant information from the context.
    vi_code = str(cntx.varidx.get_code())
    vi_name = str(cntx.varidx.get_name())
    vi_code_grp = vi.get_group(vi_code)
    view_code = str(cntx.view.get_code())

    # Emission scenarios.
    rcps = [def_rcp.rcp_ref] + cfg.rcps

    for delta in [False, True]:

        # Loop through emission scenarios.
        ds_ref = None
        ds_rcp_26, ds_rcp_26_grp, rcp_26_name_l = [], [], []
        ds_rcp_45, ds_rcp_45_grp, rcp_45_name_l = [], [], []
        ds_rcp_85, ds_rcp_85_grp, rcp_85_name_l = [], [], []
        ds_rcp_xx, ds_rcp_xx_grp, rcp_xx_name_l = [], [], []
        for rcp in rcps:

            # List files.
            if vi_code in cfg.variables:
                p_ref = cfg.get_d_stn(vi_code_grp) + vi_code_grp + "_" + stn + fu.f_ext_nc
            else:
                p_ref = cfg.get_d_idx(stn, vi_code_grp) + vi_code_grp + "_ref" + fu.f_ext_nc
            if rcp == def_rcp.rcp_ref:
                p_sim_l = [p_ref]
            else:
                if vi_code in cfg.variables:
                    d = cfg.get_d_scen(stn, const.cat_qqmap, vi_code_grp)
                else:
                    d = cfg.get_d_idx(stn, vi_code_grp)
                p_sim_l = glob.glob(d + "*_" + ("*" if rcp == def_rcp.rcp_xx else rcp) + fu.f_ext_nc)

            # Exit if there is no file corresponding to the criteria.
            if (len(p_sim_l) == 0) or ((len(p_sim_l) > 0) and not (os.path.isdir(os.path.dirname(p_sim_l[0])))):
                continue

            # Loop through simulation files.
            for i_sim in range(len(p_sim_l)):

                # Extract simulation name.
                if "rcp" not in p_sim_l[i_sim]:
                    sim_code = def_rcp.RCP(def_rcp.rcp_ref).get_desc()
                else:
                    tokens = os.path.basename(p_sim_l[i_sim]).replace(fu.f_ext_nc, "").split("_")
                    sim_code = tokens[1] + "_" + tokens[2] + "_" + tokens[3] + "_" + tokens[4]

                # Calculate dataset.
                if (view_code == def_view.code_ts_bias) and (vi_name in cfg.variables):
                    p_val = p_sim_l[i_sim]
                    if rcp != def_rcp.rcp_ref:
                        p_val = p_val.replace(cfg.sep + const.cat_qqmap, cfg.sep + const.cat_regrid).\
                            replace(fu.f_ext_nc, "_4" + const.cat_qqmap + fu.f_ext_nc)
                    ds_val = calc_ts_ds(cntx, p_val)
                    if not delta:
                        ds = ds_val
                    else:
                        ds = xr.zeros_like(ds_val)
                        if rcp != def_rcp.rcp_ref:
                            ds_del = calc_ts_ds(cntx, p_sim_l[i_sim])
                            ds[vi_name] = ds_val[vi_name] - ds_del[vi_name]
                else:
                    ds_val = calc_ts_ds(cntx, p_sim_l[i_sim])
                    if not delta:
                        ds = ds_val
                    else:
                        ds = xr.zeros_like(ds_val)
                        if rcp != def_rcp.rcp_ref:
                            ds_del = calc_ts_ds(cntx, p_ref)
                            ds[vi_name] = ds_val[vi_name] - float(ds_del[vi_name].mean().values)
                ds[vi_name].attrs[const.attrs_units] = ds_val[vi_name].attrs[const.attrs_units]

                # Append to list of datasets.
                if rcp == def_rcp.rcp_ref:
                    ds_ref = ds
                else:
                    ds_is_ok = True
                    if (vi_name == vi.i_rain_season_prcptot) and (float(ds[vi_name].max().values) == 0):
                        ds_is_ok = False
                    if ds_is_ok:
                        if rcp == def_rcp.rcp_26:
                            ds_rcp_26.append(ds)
                            rcp_26_name_l.append(sim_code)
                        elif rcp == def_rcp.rcp_45:
                            ds_rcp_45.append(ds)
                            rcp_45_name_l.append(sim_code)
                        elif rcp == def_rcp.rcp_85:
                            ds_rcp_85.append(ds)
                            rcp_85_name_l.append(sim_code)
                        ds_rcp_xx.append(ds)
                        rcp_xx_name_l.append(sim_code)

            # Group by RCP.
            if rcp != def_rcp.rcp_ref:
                if (rcp == def_rcp.rcp_26) and (len(ds_rcp_26) > 0):
                    ds_rcp_26_grp = calc_stat_mean_min_max(ds_rcp_26, vi_name)
                elif (rcp == def_rcp.rcp_45) and (len(ds_rcp_45) > 0):
                    ds_rcp_45_grp = calc_stat_mean_min_max(ds_rcp_45, vi_name)
                elif (rcp == def_rcp.rcp_85) and (len(ds_rcp_85) > 0):
                    ds_rcp_85_grp = calc_stat_mean_min_max(ds_rcp_85, vi_name)
                if len(ds_rcp_xx) > 0:
                    ds_rcp_xx_grp = calc_stat_mean_min_max(ds_rcp_xx, vi_name)

        if (ds_ref is not None) or (ds_rcp_26 != []) or (ds_rcp_45 != []) or (ds_rcp_85 != []):

            # Extract years.
            years = []
            if (def_rcp.rcp_26 in rcps) and (len(ds_rcp_26_grp) > 0):
                years = utils.extract_date_field(ds_rcp_26_grp[0], "year")
            elif (def_rcp.rcp_45 in rcps) and (len(ds_rcp_45_grp) > 0):
                years = utils.extract_date_field(ds_rcp_45_grp[0], "year")
            elif (def_rcp.rcp_85 in rcps) and (len(ds_rcp_85_grp) > 0):
                years = utils.extract_date_field(ds_rcp_85_grp[0], "year")
            dict_pd = {"year": years}

            # Create pandas dataframe (absolute values).
            df_rcp_x, df_sim_x = pd.DataFrame(dict_pd), pd.DataFrame(dict_pd)
            df_rcp_x[def_rcp.rcp_ref], df_sim_x[def_rcp.rcp_ref] = None, None
            for rcp in rcps:
                if rcp == def_rcp.rcp_ref:
                    years = utils.extract_date_field(ds_ref, "year")
                    vals = ds_ref[vi_name].values
                    for i in range(len(vals)):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                            df_rcp_x[def_rcp.rcp_ref][df_rcp_x["year"] == years[i]] = vals[i]
                            df_sim_x[def_rcp.rcp_ref][df_sim_x["year"] == years[i]] = vals[i]
                elif rcp == def_rcp.rcp_26:
                    df_rcp_x[def_rcp.rcp_26 + "_moy"] = ds_rcp_26_grp[0][vi_name].values
                    df_rcp_x[def_rcp.rcp_26 + "_min"] = ds_rcp_26_grp[1][vi_name].values
                    df_rcp_x[def_rcp.rcp_26 + "_max"] = ds_rcp_26_grp[2][vi_name].values
                    for i in range(len(ds_rcp_26)):
                        df_sim_x[rcp_26_name_l[i]] = ds_rcp_26[i][vi_name].values
                elif rcp == def_rcp.rcp_45:
                    df_rcp_x[def_rcp.rcp_45 + "_moy"] = ds_rcp_45_grp[0][vi_name].values
                    df_rcp_x[def_rcp.rcp_45 + "_min"] = ds_rcp_45_grp[1][vi_name].values
                    df_rcp_x[def_rcp.rcp_45 + "_max"] = ds_rcp_45_grp[2][vi_name].values
                    for i in range(len(ds_rcp_45)):
                        df_sim_x[rcp_45_name_l[i]] = ds_rcp_45[i][vi_name].values
                elif rcp == def_rcp.rcp_85:
                    df_rcp_x[def_rcp.rcp_85 + "_moy"] = ds_rcp_85_grp[0][vi_name].values
                    df_rcp_x[def_rcp.rcp_85 + "_min"] = ds_rcp_85_grp[1][vi_name].values
                    df_rcp_x[def_rcp.rcp_85 + "_max"] = ds_rcp_85_grp[2][vi_name].values
                    for i in range(len(ds_rcp_85)):
                        df_sim_x[rcp_85_name_l[i]] = ds_rcp_85[i][vi_name].values
                else:
                    df_rcp_x[def_rcp.rcp_xx + "_moy"] = ds_rcp_xx_grp[0][vi_name].values
                    df_rcp_x[def_rcp.rcp_xx + "_min"] = ds_rcp_xx_grp[1][vi_name].values
                    df_rcp_x[def_rcp.rcp_xx + "_max"] = ds_rcp_xx_grp[2][vi_name].values

            # Record.
            if not delta:
                df_rcp = df_rcp_x.copy()
                df_sim = df_sim_x.copy()
            else:
                df_rcp_del = df_rcp_x.copy()
                df_sim_del = df_sim_x.copy()

    return df_rcp, df_sim, df_rcp_del, df_sim_del


def calc_ts_ds(
    cntx: def_context.Context,
    p: str
) -> Union[xr.Dataset, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    As part of the caculation of time series, get dataset.

    Parameters
    ----------
    cntx : def_context.Context
        Context.
    p : str
        Path of NetCDF file to use if using absolute values (not delta).

    Returns
    -------
    Union[xr.Dataset, None]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    if not os.path.exists(p):
        return None

    vi_name = cntx.varidx.get_name()

    # Load dataset.
    ds = fu.open_netcdf(p).squeeze()
    if (def_rcp.rcp_ref in p) and (vi_name in cfg.variables):
        ds = utils.remove_feb29(ds)
        ds = utils.sel_period(ds, cfg.per_ref)

    # Records units.
    if cntx.varidx.get_ens() == vi.ens_cordex:
        units = ds[vi_name].attrs[const.attrs_units]
    else:
        units = ds.attrs[const.attrs_units]
    if units == "degree_C":
        units = const.unit_C

    # Select control point.
    if cfg.opt_ra:
        if cfg.p_bounds == "":
            ds = utils.subset_ctrl_pt(ds)
        else:
            ds = utils.squeeze_lon_lat(ds, vi_name)

    # First and last years.
    year_1 = int(str(ds.time.values[0])[0:4])
    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
    year_1 = max(year_1, cfg.per_ref[0])
    if ("rcp" not in p) or (cntx.view.get_code() == def_view.code_ts_bias):
        year_n = min(year_n, cfg.per_ref[1])
    else:
        year_n = min(year_n, cfg.per_fut[1])

    # Select years.
    ds = ds.sel(time=slice(str(year_1), str(year_n)))

    # Calculate statistics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
            ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
        else:
            ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
    n_time = len(ds[const.dim_time].values)
    da = xr.DataArray(np.array(ds[vi_name].values), name=vi_name,
                      coords=[(const.dim_time, np.arange(n_time))])

    # Create dataset.
    ds = da.to_dataset()
    ds[const.dim_time] = utils.reset_calendar_l(list(range(year_1, year_n + 1)))
    ds[vi_name].attrs[const.attrs_units] = units

    # Convert units.
    if vi_name in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
        if ds[vi_name].attrs[const.attrs_units] == const.unit_K:
            ds = ds - const.d_KC
            ds[vi_name].attrs[const.attrs_units] = const.unit_C
    elif vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
        if ds[vi_name].attrs[const.attrs_units] == const.unit_kg_m2s1:
            ds = ds * const.spd
            ds[vi_name].attrs[const.attrs_units] = const.unit_mm
    elif vi_name in [vi.v_uas, vi.v_vas, vi.v_sfcwindmax]:
        ds = ds * const.km_h_per_m_s
        ds[vi_name].attrs[const.attrs_units] = const.unit_km_h

    return ds


def calc_stat_mean_min_max(
    ds_l: [xr.Dataset],
    vi_name: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate mean, minimum and maximum values within a group of datasets.
    TODO: Include coordinates in the returned datasets.

    Parameters
    ----------
    ds_l : [xr.Dataset]
        Array of datasets from a given group.
    vi_name : str
        Climate variable or index code.

    Returns
    -------
    ds_mean_min_max : [xr.Dataset]
        Array of datasets with mean, minimum and maximum values.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_mean_min_max = []

    # Get years, units and coordinates.
    units = ds_l[0][vi_name].attrs[const.attrs_units]
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
            val = float(ds[vi_name][i_time].values)
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
        da = xr.DataArray(np.array(arr_vals), name=vi_name, coords=[(const.dim_time, np.arange(n_time))])
        ds = da.to_dataset()
        ds[const.dim_time] = utils.reset_calendar(ds, year_1, year_n, const.freq_YS)
        ds[vi_name].attrs[const.attrs_units] = units

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
    if const.dim_rlon in ds.dims:
        da_m = ds[var].rename({const.dim_rlon: const.dim_longitude, const.dim_rlat: const.dim_latitude})
    elif const.dim_lon in ds.dims:
        da_m = ds[var].rename({const.dim_lon: const.dim_longitude, const.dim_lat: const.dim_latitude})
    else:
        da_m = ds[var]

    # Grouping frequency.
    freq_str = "time.month" if freq == const.freq_MS else "time.dayofyear"
    time_str = "M" if freq == const.freq_MS else "1D"

    # Summarize data per month.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        if cfg.opt_ra:
            da_m = da_m.mean(dim={const.dim_longitude, const.dim_latitude})
        if freq != const.freq_MS:

            # Extract values.
            if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
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

                ds_m[var].attrs[const.attrs_units] = ds[var].attrs[const.attrs_units]
                ds_l.append(ds_m)

        else:

            # Calculate statistics for each month-year combination.
            # Looping through years is less efficient but required to avoid a problem with incomplete at-a-station
            # datasets.
            arr_all = []
            for m in range(1, 13):
                vals_m = []
                for y in range(per[0], per[1] + 1):
                    if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
                        val_m_y = np.nansum(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    else:
                        val_m_y = np.nanmean(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    vals_m.append(val_m_y)
                arr_all.append(vals_m)

            # Create dataset.
            dict_pd = {var: arr_all}
            df = pd.DataFrame(dict_pd)
            ds_l = df.to_xarray()
            ds_l.attrs[const.attrs_units] = ds[var].attrs[const.attrs_units]

    return ds_l


def calc_map(
    vi_code_l: List[str],
    i_vi_proc: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate heat map.

    Parameters
    ----------
    vi_code_l: List[str],
        Variables or indices.
    i_vi_proc: int
        Rank of variable or index to process.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get variable or index.
    vi_code = vi_code_l[i_vi_proc]

    # Extract variable name, group and RCPs.
    cat = const.cat_scen if vi_code in cfg.variables else const.cat_idx
    varidx = vi.VarIdx(vi_code)
    vi_name = vi_code if cat == const.cat_scen else varidx.get_name()
    vi_code_grp = vi.get_group(vi_code)
    rcps = [def_rcp.rcp_ref, def_rcp.rcp_xx] + cfg.rcps

    stn = "stns" if not cfg.opt_ra else cfg.obs_src

    msg = "Calculating maps."
    fu.log(msg, True)

    # Get statistic string.
    def get_stat_str(
        _stat: str,
        _q: float
    ):

        if _stat in [def_stat.code_mean, def_stat.code_min, def_stat.code_max]:
            _stat_str = "_" + _stat
        else:
            _stat_str = "_q" + str(int(_q * 100)).rjust(2, "0")

        return _stat_str

    # Get the path of the CSV and PNG files.
    def get_p_csv_fig(
        _per: List[int],
        _per_str: str,
        _stat: str,
        _delta: Optional[bool] = False
    ) -> Tuple[str, str]:

        # PNG file.
        _d_fig =\
            cfg.get_d_scen(stn, const.cat_fig + cfg.sep + cat + cfg.sep + const.cat_fig_map,
                           (vi_code_grp + cfg.sep if vi_code_grp != vi_code else "") + vi_code + cfg.sep + _per_str)
        _stat_str = get_stat_str(stat, q)
        _fn_fig = vi_name + "_" + rcp + "_" + str(_per[0]) + "_" + str(_per[1]) + _stat_str + fu.f_ext_png
        _p_fig = _d_fig + _fn_fig
        if _delta:
            _p_fig = _p_fig.replace(fu.f_ext_png, "_delta" + fu.f_ext_png)

        # CSV file.
        _d_csv = cfg.get_d_scen(stn, const.cat_fig + cfg.sep + cat + cfg.sep + "map",
                                (vi_code_grp + cfg.sep if vi_code_grp != vi_code else "") +
                                vi_code + "_csv" + cfg.sep + _per_str)
        _fn_csv = _fn_fig.replace(fu.f_ext_png, fu.f_ext_csv)
        _p_csv = _d_csv + _fn_csv
        if _delta:
            _p_csv = _p_csv.replace(fu.f_ext_csv, "_delta" + fu.f_ext_csv)

        return _p_csv, _p_fig

    # Prepare data -----------------------------------------------------------------------------------------------------

    # Calculate the overall minimum and maximum values (considering all maps for the current 'varidx').
    # There is one z-value per period.
    n_per = len(cfg.per_hors)
    z_min_net = [np.nan] * n_per
    z_max_net = [np.nan] * n_per
    z_min_del = [np.nan] * n_per
    z_max_del = [np.nan] * n_per

    # Calculated datasets and item description.
    arr_ds_maps = []
    arr_items = []

    # Build arrays for statistics to calculate.
    arr_stat = [def_stat.code_mean]
    arr_q = [-1]
    if cfg.opt_map_quantiles is not None:
        arr_stat = arr_stat + ([def_stat.code_quantile] * len(cfg.opt_map_quantiles))
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
            pers = [cfg.per_ref] if (j == 0) else cfg.per_hors
            for k_per in range(len(pers)):
                per = pers[k_per]
                per_str = str(per[0]) + "-" + str(per[1])

                # Path of CSV and PNG files.
                p_csv, p_fig = get_p_csv_fig(per, per_str, stat)

                if os.path.exists(p_csv) and not cfg.opt_force_overwrite:

                    # Load data.
                    df = pd.read_csv(p_csv)

                    # Convert to DataArray.
                    df = pd.DataFrame(df, columns=["longitude", "latitude", vi_code])
                    df = df.sort_values(by=["latitude", "longitude"])
                    lat = list(set(df["latitude"]))
                    lat.sort()
                    lon = list(set(df["longitude"]))
                    lon.sort()
                    arr = np.reshape(list(df[vi_code]), (len(lat), len(lon)))
                    da = xr.DataArray(data=arr, dims=["latitude", "longitude"],
                                      coords=[("latitude", lat), ("longitude", lon)])
                    da.name = vi_name
                    ds_map = da.to_dataset()

                else:

                    # Calculate map.
                    ds_map = calc_map_rcp(vi_code, rcp, per, stat, q)

                    # Clip to geographic boundaries.
                    if cfg.opt_map_clip and (cfg.p_bounds != ""):
                        ds_map = utils.subset_shape(ds_map)

                # Record current map, statistic and quantile, RCP and period.
                arr_ds_maps.append(ds_map)
                arr_items.append([stat, q, rcp, per])

                # Calculate reference map.
                if (ds_map_ref is None) and\
                   (stat == def_stat.code_mean) and (rcp == def_rcp.rcp_xx) and (per == cfg.per_ref):
                    ds_map_ref = ds_map

                # Extract values.
                vals_net = ds_map[vi_name].values
                vals_del = None
                if per != cfg.per_ref:
                    vals_del = ds_map[vi_name].values - ds_map_ref[vi_name].values

                # Record mean absolute values.
                z_min_j = np.nanmin(vals_net)
                z_max_j = np.nanmax(vals_net)
                z_min_net[k_per] = z_min_j if np.isnan(z_min_net[k_per]) else min(z_min_net[k_per], z_min_j)
                z_max_net[k_per] = z_max_j if np.isnan(z_max_net[k_per]) else max(z_max_net[k_per], z_max_j)

                # Record mean delta values.
                if vals_del is not None:
                    z_min_del_j = np.nanmin(vals_del)
                    z_max_del_j = np.nanmax(vals_del)
                    z_min_del[k_per] = z_min_del_j if np.isnan(z_min_del[k_per])\
                        else min(z_min_del[k_per], z_min_del_j)
                    z_max_del[k_per] = z_max_del_j if np.isnan(z_max_del[k_per])\
                        else max(z_max_del[k_per], z_max_del_j)

    # Generate maps ----------------------------------------------------------------------------------------------------

    # Loop through maps.
    for i in range(len(arr_ds_maps)):

        # Current map.
        ds_map = arr_ds_maps[i]

        # Current RCP and horizon.
        stat  = arr_items[i][0]
        q     = arr_items[i][1]
        rcp   = arr_items[i][2]
        per   = arr_items[i][3]
        i_per = cfg.per_hors.index(per)

        # Perform twice (for values, then for deltas).
        for j in range(2):

            # Skip if delta map generation if the option is disabled or if it's the reference period.
            if (j == 1) and\
                ((per == cfg.per_ref) or
                 ((cat == const.cat_scen) and (not cfg.opt_map_delta[0])) or
                 ((cat == const.cat_idx) and (not cfg.opt_map_delta[1]))):
                continue

            # Extract dataset (calculate delta if required).
            if j == 0:
                da_map = ds_map[vi_name]
            else:
                da_map = ds_map[vi_name] - ds_map_ref[vi_name]

            # Perform twice (for min/max values specific to a period, then for overall min/max values).
            for k in range(2):

                # Specific period.
                if k == 0:
                    per_str = str(per[0]) + "-" + str(per[1])
                    z_min = z_min_net[i_per] if j == 0 else z_min_del[i_per]
                    z_max = z_max_net[i_per] if j == 0 else z_max_del[i_per]
                # All periods combined.
                else:
                    per_str = str(cfg.per_ref[0]) + "-" + str(cfg.per_hors[len(cfg.per_hors) - 1][1])
                    z_min = np.nanmin(z_min_net if j == 0 else z_min_del)
                    z_max = np.nanmax(z_max_net if j == 0 else z_max_del)

                # PNG and CSV formats ----------------------------------------------------------------------------------

                # Path of output files.
                p_csv, p_fig = get_p_csv_fig(per, per_str, stat, j == 1)

                # Create context.
                cntx = def_context.Context(def_context.code_script)
                cntx.project     = def_project.Project("x", cntx=cntx)
                cntx.p_locations = cfg.p_locations
                cntx.p_bounds    = cfg.p_bounds
                cntx.view        = def_view.View(def_view.code_map)
                cntx.lib         = def_lib.Lib(def_lib.mode_mat)
                cntx.varidx      = vi.VarIdx(vi_name)
                cntx.project.set_quantiles("x", cntx, cfg.opt_map_quantiles)
                cntx.RCP         = def_rcp.RCP(rcp)
                cntx.hor         = def_hor.Hor(per)
                cntx.stat        = def_stat.Stat(get_stat_str(stat, q).replace("_", ""))
                cntx.delta       = def_delta.Del(j == 1)

                # Update colors.
                if len(cfg.opt_map_col_temp_var) > 0:
                    cntx.opt_map_col_temp_var = cfg.opt_map_col_temp_var
                if len(cfg.opt_map_col_temp_idx_1) > 0:
                    cntx.opt_map_col_temp_idx_1 = cfg.opt_map_col_temp_idx_1
                if len(cfg.opt_map_col_temp_idx_2) > 0:
                    cntx.opt_map_col_temp_idx_2 = cfg.opt_map_col_temp_idx_2
                if len(cfg.opt_map_col_prec_var) > 0:
                    cntx.opt_map_col_prec_var = cfg.opt_map_col_prec_var
                if len(cfg.opt_map_col_prec_idx_1) > 0:
                    cntx.opt_map_col_prec_idx_1 = cfg.opt_map_col_prec_idx_1
                if len(cfg.opt_map_col_prec_idx_2) > 0:
                    cntx.opt_map_col_prec_idx_2 = cfg.opt_map_col_prec_idx_2
                if len(cfg.opt_map_col_prec_idx_3) > 0:
                    cntx.opt_map_col_prec_idx_3 = cfg.opt_map_col_prec_idx_3
                if len(cfg.opt_map_col_wind_var) > 0:
                    cntx.opt_map_col_wind_var = cfg.opt_map_col_wind_var
                if len(cfg.opt_map_col_wind_idx_1) > 0:
                    cntx.opt_map_col_wind_idx_1 = cfg.opt_map_col_wind_idx_1
                if len(cfg.opt_map_col_default) > 0:
                    cntx.opt_map_col_default = cfg.opt_map_col_default

                # Determine if PNG and CSV files need to be saved.
                save_fig = cfg.opt_force_overwrite or \
                           ((not os.path.exists(p_fig)) and (fu.f_png in cfg.opt_map_format))
                save_csv = cfg.opt_force_overwrite or \
                           ((not os.path.exists(p_csv)) and (fu.f_csv in cfg.opt_map_format))

                # Create dataframe.
                arr_lon, arr_lat, arr_val = [], [], []
                for m in range(len(da_map.longitude.values)):
                    for n in range(len(da_map.latitude.values)):
                        arr_lon.append(da_map.longitude.values[m])
                        arr_lat.append(da_map.latitude.values[n])
                        arr_val.append(da_map.values[n, m])
                dict_pd = {const.dim_longitude: arr_lon, const.dim_latitude: arr_lat, vi_name: arr_val}
                df = pd.DataFrame(dict_pd)

                # Generate plot and save PNG file.
                if save_fig:
                    fig = dash_plot.gen_map(cntx, df, [z_min, z_max])
                    fu.save_plot(fig, p_fig)

                # Save CSV file.
                if save_csv:
                    fu.save_csv(df, p_csv)

                # TIF format -------------------------------------------------------------------------------------------

                # Path of TIF file.
                p_tif = p_fig.replace(vi_code + cfg.sep, vi_code + "_" + fu.f_tif + cfg.sep).\
                    replace(fu.f_ext_png, fu.f_ext_tif)

                if (fu.f_tif in cfg.opt_map_format) and ((not os.path.exists(p_tif)) or cfg.opt_force_overwrite):

                    # TODO: da_tif.rio.reproject is now crashing. It was working in July 2021.

                    # Increase resolution.
                    da_tif = da_map.copy()
                    if cfg.opt_map_res > 0:
                        lat_vals = np.arange(min(da_tif.latitude), max(da_tif.latitude), cfg.opt_map_res)
                        lon_vals = np.arange(min(da_tif.longitude), max(da_tif.longitude), cfg.opt_map_res)
                        da_tif = da_tif.rename({const.dim_latitude: const.dim_lat, const.dim_longitude: const.dim_lon})
                        da_grid = xr.Dataset(
                            {const.dim_lat: ([const.dim_lat], lat_vals), const.dim_lon: ([const.dim_lon], lon_vals)})
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            da_tif = xe.Regridder(da_tif, da_grid, "bilinear")(da_tif)

                    # Project data.
                    da_tif.rio.set_crs("EPSG:4326")
                    if cfg.opt_map_spat_ref != "EPSG:4326":
                        da_tif.rio.set_spatial_dims(const.dim_lon, const.dim_lat, inplace=True)
                        da_tif = da_tif.rio.reproject(cfg.opt_map_spat_ref)
                        da_tif.values[da_tif.values == -9999] = np.nan
                        da_tif = da_tif.rename({"y": const.dim_lat, "x": const.dim_lon})

                    # Save.
                    d = os.path.dirname(p_tif)
                    if not (os.path.isdir(d)):
                        os.makedirs(d)
                    da_tif.rio.to_raster(p_tif)


def calc_map_rcp(
    vi_code: str,
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
    vi_code: str
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

    # Extract variable name and group.
    cat = const.cat_scen if vi_code in cfg.variables else const.cat_idx
    varidx = vi.VarIdx(vi_code)
    vi_name = vi_code if cat == const.cat_scen else varidx.get_name()
    vi_code_grp = vi.get_group(vi_code)

    fu.log("Processing: " + vi_code + ", " +
           ("q" + str(int(q * 100)).rjust(2, "0") if q >= 0 else stat) + ", " +
           str(def_rcp.RCP(rcp).get_desc()) + ", " + str(per[0]) + "-" + str(per[1]) + "", True)

    # Number of years and stations.
    if rcp == def_rcp.rcp_ref:
        n_year = cfg.per_ref[1] - cfg.per_ref[0] + 1
    else:
        n_year = cfg.per_fut[1] - cfg.per_ref[1] + 1

    # List stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]

    # Observations -----------------------------------------------------------------------------------------------------

    if not cfg.opt_ra:

        # Get information on stations.
        p_stn = glob.glob(cfg.get_d_stn(vi.v_tas) + "../*" + fu.f_ext_csv)[0]
        df = pd.read_csv(p_stn, sep=cfg.f_sep)

        # Collect values for each station and determine overall boundaries.
        fu.log("Collecting emissions scenarios at each station.", True)
        x_bnds = []
        y_bnds = []
        data_stn = []
        for stn in stns:

            # Get coordinates.
            lon = df[df["station"] == stn][const.dim_lon].values[0]
            lat = df[df["station"] == stn][const.dim_lat].values[0]

            # Calculate statistics.
            ds_stat = calc_stat(const.cat_obs if rcp == def_rcp.rcp_ref else const.cat_scen, const.freq_YS,
                                const.freq_YS, stn, vi_code, rcp, None, False, cfg.opt_map_clip, stat, q)
            if ds_stat is None:
                continue

            # Extract data from stations.
            data = [[], [], []]
            n = ds_stat.dims[const.dim_time]
            for year in range(0, n):

                # Collect data.
                x = float(lon)
                y = float(lat)
                z = float(ds_stat[vi_name][0][0][year])
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
        fu.log("Collecting the coordinates of stations.", True)
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
        fu.log("Performing interpolation.", True)
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
                              coords={const.dim_time: grid_time, const.dim_lat: grid_y, const.dim_lon: grid_x},
                              dims=[const.dim_time, const.dim_lat, const.dim_lon])
        ds_res = da_itp.to_dataset(name=vi_name)

    # There is no need to interpolate.
    else:

        # Reference period.
        if vi_name in cfg.variables:
            p_sim_ref = cfg.get_d_scen(cfg.obs_src, const.cat_obs, vi_code_grp) +\
                vi_code_grp + "_" + cfg.obs_src + fu.f_ext_nc
        else:
            p_sim_ref = cfg.get_d_idx(cfg.obs_src, vi_code_grp) + vi_code_grp + "_ref" + fu.f_ext_nc
        if rcp == def_rcp.rcp_ref:

            # Open dataset.
            p_sim = p_sim_ref
            if not os.path.exists(p_sim):
                return xr.Dataset(None)
            ds_sim = fu.open_netcdf(p_sim)

            # TODO: The following patch was added to fix dimensions for vi.i_drought_code.
            if vi_code in cfg.idx_codes:
                ds_sim = ds_sim.resample(time=const.freq_YS).mean(dim=const.dim_time, keep_attrs=True)

            # Select period.
            ds_sim = utils.remove_feb29(ds_sim)
            ds_sim = utils.sel_period(ds_sim, per)

            # Extract units.
            units = 1
            if const.attrs_units in ds_sim[vi_name].attrs:
                units = ds_sim[vi_name].attrs[const.attrs_units]
            elif const.attrs_units in ds_sim.data_vars:
                units = ds_sim[const.attrs_units]

            # Calculate mean.
            ds_res = ds_sim.mean(dim=const.dim_time)
            if vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
                ds_res = ds_res * 365
            if cat == const.cat_scen:
                ds_res[vi_name].attrs[const.attrs_units] = units
            else:
                ds_res.attrs[const.attrs_units] = units

        # Future period.
        else:

            # List scenarios or indices for the current RCP.
            if cat == const.cat_scen:
                d = cfg.get_d_scen(cfg.obs_src, const.cat_qqmap, vi_code_grp)
            else:
                d = cfg.get_d_scen(cfg.obs_src, const.cat_idx, vi_code_grp)
            p_sim_l = [i for i in glob.glob(d + "*" + fu.f_ext_nc) if i != p_sim_ref]

            # Collect simulations.
            arr_sim = []
            ds_res  = None
            n_sim   = 0
            units   = 1
            for p_sim in p_sim_l:
                if os.path.exists(p_sim) and ((rcp in p_sim) or (rcp == def_rcp.rcp_xx)):

                    # Open dataset.
                    ds_sim = fu.open_netcdf(p_sim)

                    # TODO: The following patch was added to fix dimensions for vi.i_drought_code.
                    if vi_code in cfg.idx_codes:
                        ds_sim = ds_sim.resample(time=const.freq_YS).mean(dim=const.dim_time, keep_attrs=True)

                    # Extract units.
                    if const.attrs_units in ds_sim[vi_name].attrs:
                        units = ds_sim[vi_name].attrs[const.attrs_units]
                    elif const.attrs_units in ds_sim.data_vars:
                        units = ds_sim[const.attrs_units]

                    # Select period.
                    ds_sim = utils.remove_feb29(ds_sim)
                    ds_sim = utils.sel_period(ds_sim, per)

                    # Calculate mean and add to array.
                    ds_sim = ds_sim.mean(dim=const.dim_time)
                    if vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
                        ds_sim = ds_sim * 365
                    if cat == const.cat_scen:
                        ds_sim[vi_name].attrs[const.attrs_units] = units
                    else:
                        ds_sim.attrs[const.attrs_units] = units
                    arr_sim.append(ds_sim)

                    # The first dataset will be used to return result.
                    if n_sim == 0:
                        ds_res = ds_sim.copy(deep=True)
                        ds_res[vi_name][:, :] = np.nan
                    n_sim = n_sim + 1

            if ds_res is None:
                return xr.Dataset(None)

            # Get longitude and latitude.
            lon_l, lat_l = utils.get_coordinates(arr_sim[0][vi_name], True)

            # Calculate statistics.
            len_lat = len(lat_l)
            len_lon = len(lon_l)
            for i_lat in range(len_lat):
                for j_lon in range(len_lon):

                    # Extract the value for the current cell for all simulations.
                    vals = []
                    for ds_sim_k in arr_sim:
                        dims_sim = ds_sim_k[vi_name].dims
                        if const.dim_rlat in dims_sim:
                            val_sim = float(ds_sim_k[vi_name].isel(rlon=j_lon, rlat=i_lat))
                        elif const.dim_lat in dims_sim:
                            val_sim = float(ds_sim_k[vi_name].isel(lon=j_lon, lat=i_lat))
                        else:
                            val_sim = float(ds_sim_k[vi_name].isel(longitude=j_lon, latitude=i_lat))
                        vals.append(val_sim)

                    # Calculate statistic.
                    val = np.nan
                    if stat == def_stat.code_mean:
                        val = np.mean(vals)
                    elif stat == def_stat.code_min:
                        val = np.min(vals)
                    elif stat == def_stat.code_max:
                        val = np.max(vals)
                    elif stat == def_stat.code_quantile:
                        val = np.quantile(vals, q)

                    # Record value.
                    ds_res[vi_name][i_lat, j_lon] = val

        # Remember units.
        units = ds_res[vi_name].attrs[const.attrs_units] if cat == const.cat_scen else ds_res.attrs[const.attrs_units]

        # Convert units.
        if (vi_name in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]) and\
           (ds_res[vi_name].attrs[const.attrs_units] == const.unit_K):
            ds_res = ds_res - const.d_KC
            ds_res[vi_name].attrs[const.attrs_units] = const.unit_C
        elif (vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]) and \
             (ds_res[vi_name].attrs[const.attrs_units] == const.unit_kg_m2s1):
            ds_res = ds_res * const.spd
            ds_res[vi_name].attrs[const.attrs_units] = const.unit_mm
        elif vi_name in [vi.v_uas, vi.v_vas, vi.v_sfcwindmax]:
            ds_res = ds_res * const.km_h_per_m_s
            ds_res[vi_name].attrs[const.attrs_units] = const.unit_km_h
        else:
            ds_res[vi_name].attrs[const.attrs_units] = units

        # Adjust coordinate names (required for clipping).
        if const.dim_longitude not in list(ds_res.dims):
            if const.dim_rlon in ds_res.dims:
                ds_res = ds_res.rename_dims({const.dim_rlon: const.dim_longitude, const.dim_rlat: const.dim_latitude})
                ds_res[const.dim_longitude] = ds_res[const.dim_rlon]
                del ds_res[const.dim_rlon]
                ds_res[const.dim_latitude] = ds_res[const.dim_rlat]
                del ds_res[const.dim_rlat]
            else:
                ds_res = ds_res.rename_dims({const.dim_lon: const.dim_longitude, const.dim_lat: const.dim_latitude})
                ds_res[const.dim_longitude] = ds_res[const.dim_lon]
                del ds_res[const.dim_lon]
                ds_res[const.dim_latitude] = ds_res[const.dim_lat]
                del ds_res[const.dim_lat]

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
        Category: const.cat_scen is for climate scenarios or const.cat_idx for climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Loop through categories.
        cat_l = [const.cat_obs, const.cat_raw, const.cat_regrid, const.cat_qqmap]
        if cat == const.cat_idx:
            cat_l = [const.cat_idx]
        for cat in cat_l:

            # Explode lists of codes and names.
            idx_names_exploded = vi.explode_idx_l(cfg.idx_names)
            idx_codes_exploded = vi.explode_idx_l(cfg.idx_codes)

            # Loop through variables or indices.
            vi_name_l = cfg.variables if cat != const.cat_idx else idx_names_exploded
            for i_vi_name in range(len(vi_name_l)):
                vi_name = idx_names_exploded[i_vi_name]
                vi_code = idx_codes_exploded[i_vi_name]
                vi_code_grp = vi.get_group(vi_code)

                # List NetCDF files.
                p_l = list(glob.glob(cfg.get_d_scen(stn, cat, vi_code_grp) + "*" + fu.f_ext_nc))
                n_files = len(p_l)
                if n_files == 0:
                    continue
                p_l.sort()

                fu.log("Processing: " + stn + ", " + vi_code, True)

                # Scalar processing mode.
                if cfg.n_proc == 1:
                    for i_file in range(n_files):
                        conv_nc_csv_single(p_l, vi_code, i_file)

                # Parallel processing mode.
                else:

                    # Loop until all files have been converted.
                    while True:

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before =\
                            len(list(glob.glob(cfg.get_d_scen(stn, cat, vi_code) + "*" + fu.f_ext_csv)))

                        try:
                            fu.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=min(cfg.n_proc, len(p_l)))
                            func = functools.partial(conv_nc_csv_single, p_l, vi_name)
                            pool.map(func, list(range(n_files)))
                            fu.log("Work done!", True)
                            pool.close()
                            pool.join()
                            fu.log("Fork ended.", True)
                        except Exception as e:
                            fu.log(str(e))
                            pass

                        # Calculate the number of files processed (after conversion).
                        n_files_proc_after =\
                            len(list(glob.glob(cfg.get_d_scen(stn, cat, vi_code_grp) + "*" + fu.f_ext_csv)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_files_proc_before == n_files_proc_after):
                            break


def conv_nc_csv_single(
    p_l: [str],
    vi_code: str,
    i_file: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a single NetCDF to CSV file.

    Parameters
    ----------
    p_l : [str]
        List of paths.
    vi_code : str
        Climate variable or index code.
    i_file : int
        Rank of file in 'p_l'.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract variable name and group.
    varidx = vi.VarIdx(vi_code)
    vi_name = vi_code if (vi_code in cfg.variables) else varidx.get_name()
    vi_code_grp = vi.get_group(vi_code)

    # Paths.
    p = p_l[i_file]
    p_csv = p.replace(fu.f_ext_nc, fu.f_ext_csv).\
        replace(cfg.sep + vi_code_grp + "_", cfg.sep + vi_name + "_").\
        replace(cfg.sep + vi_code_grp + cfg.sep, cfg.sep + vi_code_grp + "_" + fu.f_csv + cfg.sep)

    if os.path.exists(p_csv) and (not cfg.opt_force_overwrite):
        if cfg.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Explode lists of codes and names.
    idx_names_exploded = vi.explode_idx_l(cfg.idx_names)

    # Open dataset.
    ds = xr.open_dataset(p)

    # Extract time.
    time_l = list(ds.time.values)
    n_time = len(time_l)
    for i in range(n_time):
        if vi_name not in idx_names_exploded:
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
    val_l = list(ds[vi_name].values)
    if not cfg.opt_ra:
        if vi_name not in cfg.idx_names:
            for i in range(n_time):
                val_l[i] = val_l[i].mean()
        else:
            if def_rcp.rcp_ref in p:
                for i in range(n_time):
                    val_l[i] = val_l[i][0][0]
            else:
                val_l = list(val_l[0][0])

    # Convert values to more practical units (if required).
    if (vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]) and\
       (ds[vi_name].attrs[const.attrs_units] == const.unit_kg_m2s1):
        for i in range(n_time):
            val_l[i] = val_l[i] * const.spd
    elif (vi_name in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]) and\
         (ds[vi_name].attrs[const.attrs_units] == const.unit_K):
        for i in range(n_time):
            val_l[i] = val_l[i] - const.d_KC

    # Build pandas dataframe.
    if not cfg.opt_ra:
        dict_pd = {const.dim_time: time_l, vi_name: val_l}
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
        dict_pd = {const.dim_time: time_l_1d, const.dim_lon: lon_l_1d, const.dim_lat: lat_l_1d, vi_name: val_l_1d}
    df = pd.DataFrame(dict_pd)

    # Save CSV file.
    fu.save_csv(df, p_csv)

    if cfg.n_proc > 1:
        fu.log("Work done!", True)


def calc_cycle(
    ds: xr.Dataset,
    stn: str,
    vi_name: str,
    per: [int, int],
    freq: str,
    title: str,
    i_trial: int = 1
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate monthly plots (for the reference period).

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    stn: str
        Station.
    vi_name: str
        Climate variable or index name.
    per: [int, int]
        Period of interest, for instance, [1981, 2010].
    freq: str
        Frequency = {const.freq_D, const.freq_MS}
    title: str
        Plot title.
    i_trial: int
        Iteration number. The purpose is to attempt doing the analysis again. It happens once in a while that the
        dictionary is missing values, which results in the impossibility to build a dataframe and save it.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Paths.
    cat_fig = const.cat_fig_cycle_ms if freq == const.freq_MS else const.cat_fig_cycle_d
    p_fig = cfg.get_d_scen(stn, const.cat_fig + cfg.sep + const.cat_scen + cfg.sep + cat_fig, vi_name)
    if vi_name in cfg.idx_names:
        p_fig = p_fig.replace(cfg.sep + const.cat_scen, cfg.sep + const.cat_idx)
    p_fig += title + fu.f_ext_png
    p_csv = p_fig.replace(cfg.sep + vi_name + cfg.sep, cfg.sep + vi_name + "_" + fu.f_csv + cfg.sep).\
        replace(fu.f_ext_png, fu.f_ext_csv)

    # Determine if the analysis is required.
    cat = const.cat_scen if (vi_name in cfg.variables) else const.cat_idx
    analysis_enabled = ((cat == const.cat_scen) and cfg.opt_cycle[0]) or ((cat == const.cat_idx) and cfg.opt_cycle[1])
    save_fig = (cfg.opt_force_overwrite or ((not os.path.exists(p_fig)) and (fu.f_png in cfg.opt_cycle_format)))
    save_csv = (cfg.opt_force_overwrite or ((not os.path.exists(p_csv)) and (fu.f_csv in cfg.opt_cycle_format)))
    if not (analysis_enabled and (save_fig or save_csv)):
        return

    # Load existing CSV file.
    if (not cfg.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Extract data.
        if i_trial == 1:
            ds = utils.sel_period(ds, per)
            if freq == const.freq_D:
                ds = utils.remove_feb29(ds)

        # Convert units.
        units = ds[vi_name].attrs[const.attrs_units]
        if (vi_name in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]) and \
           (ds[vi_name].attrs[const.attrs_units] == const.unit_K):
            ds = ds - const.d_KC
        elif (vi_name in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]) and \
             (ds[vi_name].attrs[const.attrs_units] == const.unit_kg_m2s1):
            ds = ds * const.spd
        ds[vi_name].attrs[const.attrs_units] = units

        # Calculate statistics.
        ds_l = calc_by_freq(ds, vi_name, per, freq)

        n = 12 if freq == const.freq_MS else 365

        # Remove February 29th.
        if (freq == const.freq_D) and (len(ds_l[0][vi_name]) > 365):
            for i in range(3):
                ds_l[i] = ds_l[i].rename_dims({"dayofyear": const.dim_time})
                ds_l[i] = ds_l[i][vi_name][ds_l[i][const.dim_time] != 59].to_dataset()
                ds_l[i][const.dim_time] = utils.reset_calendar(ds_l[i], cfg.per_ref[0], cfg.per_ref[0], const.freq_D)
                ds_l[i][vi_name].attrs[const.attrs_units] = ds[vi_name].attrs[const.attrs_units]

        # Create dataframe.
        if freq == const.freq_D:
            dict_pd = {
                ("month" if freq == const.freq_MS else "day"): range(1, n + 1),
                "mean": list(ds_l[0][vi_name].values),
                "min": list(ds_l[1][vi_name].values),
                "max": list(ds_l[2][vi_name].values), "var": [vi_name] * n
            }
            df = pd.DataFrame(dict_pd)
        else:
            year_l = list(range(per[0], per[1] + 1))
            dict_pd = {
                "year": year_l,
                "1": ds_l[vi_name].values[0], "2": ds_l[vi_name].values[1], "3": ds_l[vi_name].values[2],
                "4": ds_l[vi_name].values[3], "5": ds_l[vi_name].values[4], "6": ds_l[vi_name].values[5],
                "7": ds_l[vi_name].values[6], "8": ds_l[vi_name].values[7], "9": ds_l[vi_name].values[8],
                "10": ds_l[vi_name].values[9], "11": ds_l[vi_name].values[10], "12": ds_l[vi_name].values[11]
            }
            df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:

        # Create context.
        cntx = def_context.Context(def_context.code_script)
        cntx.p_locations = cfg.p_locations
        cntx.p_bounds = cfg.p_bounds
        cntx.lib = def_lib.Lib(def_lib.mode_mat)
        cntx.varidx = vi.VarIdx(vi_name)

        # Generate plot.
        if freq == const.freq_D:
            fig = dash_plot.gen_cycle_d(cntx, df)
        else:
            fig = dash_plot.gen_cycle_ms(cntx, df)

        # Save plot.
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    error = False
    if save_csv:
        try:
            fu.save_csv(df, p_csv)
        except:
            error = True

    # Attempt the same analysis again if an error occurred. Remove this option if it's no longer required.
    if error:

        # Log error.
        msg_err = "Unable to save " + ("daily" if (freq == const.freq_D) else "monthly") +\
                  " plot data (failed " + str(i_trial) + " time(s)):"
        fu.log(msg_err, True)
        fu.log(title, True)

        # Attempt the same analysis again.
        if i_trial < 3:
            calc_cycle(ds, stn, vi_name, per, freq, title, i_trial + 1)
