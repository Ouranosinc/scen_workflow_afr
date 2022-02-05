# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Statistics functions.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import functools
import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
# Do not delete the line below: 'rioxarray' must be added even if it's not explicitly used in order to have a 'rio'
# variable in DataArrays.
import rioxarray as rio
import sys
import warnings
import xarray as xr
import xesmf as xe
from pandas.core.common import SettingWithCopyWarning
from scipy.interpolate import griddata
from streamlit import caching
from typing import Union, List, Tuple, Optional

# Workflow libraries.
import file_utils as fu
import plot
import utils
from def_constant import const as c
from def_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import dash_plot, dash_statistics as dash_stats, dash_utils, def_varidx as VI
from dashboard.def_delta import Delta
from dashboard.def_hor import Hor, Hors
from dashboard.def_lib import Lib
from dashboard.def_rcp import RCP, RCPs
from dashboard.def_sim import Sim
from dashboard.def_stat import Stat, Stats
from dashboard.def_varidx import VarIdx
from dashboard.def_view import View


def calc_stat(
    stn: str,
    varidx: VarIdx,
    rcp: RCP,
    hor: Union[None, Hor],
    stat: Stat,
    per_region: bool,
    freq_in: str,
    freq_out: str,
    clip: bool
) -> Union[xr.Dataset, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate quantiles.

    Parameters
    ----------
    stn: str
        Station.
    varidx: VarIdx
        Climate variable or index.
    rcp: RCP
        Emission scenario.
    hor: Union[None, Hor]
        Horizon: ex: [1981, 2010]
        If None is specified, the complete time range is considered.
    stat: Stat
        Statistic.
    per_region: bool
        If True, statistics are calculated for a region as a whole.
    freq_in: str
        Frequency (input): const.freq_D=daily; const.freq_YS=annual
    freq_out: str
        Frequency (output): const.freq_D=daily; const.freq_YS=annual
    clip: bool
        If True, clip according to 'cntx.p_bounds'.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract name and group.
    vi_code = varidx.code
    vi_name = varidx.name
    vi_code_grp = VI.group(vi_code) if varidx.is_group else vi_code
    vi_name_grp = VI.group(vi_name) if varidx.is_group else vi_name

    # List files.
    if rcp.code == c.ref:
        if varidx.is_var:
            p_sim_l = [cntx.d_scen(stn, c.cat_obs, vi_code_grp) + vi_name_grp + "_" + stn + c.f_ext_nc]
        else:
            p_sim_l = [cntx.d_idx(stn, vi_code_grp) + vi_name_grp + "_ref" + c.f_ext_nc]
    else:
        if varidx.is_var:
            d = cntx.d_scen(stn, c.cat_qqmap, vi_code_grp)
        else:
            d = cntx.d_idx(stn, vi_code_grp)
        p_sim_l = glob.glob(d + "*_" + ("*rcp*" if rcp.code == c.rcpxx else rcp.code) + c.f_ext_nc)

    # Exit if there is no file corresponding to the criteria.
    if (len(p_sim_l) == 0) or \
       ((len(p_sim_l) > 0) and not(os.path.isdir(os.path.dirname(p_sim_l[0])))):
        return None

    # List days.
    ds = fu.open_netcdf(p_sim_l[0])
    if (c.dim_lon in ds.variables) or (c.dim_lon in ds.dims):
        lon = ds[c.dim_lon]
        lat = ds[c.dim_lat]
    elif (c.dim_rlon in ds.variables) or (c.dim_rlon in ds.dims):
        lon = ds[c.dim_rlon]
        lat = ds[c.dim_rlat]
    else:
        lon = ds[c.dim_longitude]
        lat = ds[c.dim_latitude]
    if c.attrs_units in ds[vi_name].attrs:
        units = ds[vi_name].attrs[c.attrs_units]
    else:
        units = 1
    n_sim = len(p_sim_l)

    # First and last years.
    year_1 = int(str(ds.time.values[0])[0:4])
    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
    if hor is not None:
        year_1 = max(year_1, hor.year_1)
        year_n = min(year_n, hor.year_2)
    n_time = (year_n - year_1 + 1) * (365 if freq_in == c.freq_D else 1)

    # Collect values from simulations.
    arr_vals = []
    for i_sim in range(n_sim):

        # Load dataset.
        ds = fu.open_netcdf(p_sim_l[i_sim])

        # Patch used to fix potentially unordered dimensions of index 'c.i_drought_code'.
        if vi_code in cntx.idxs.code_l:
            ds = ds.resample(time=c.freq_YS).mean(dim=c.dim_time, keep_attrs=True)

        # Select years and adjust units.
        years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
        ds = ds.sel(time=slice(years_str[0], years_str[1]))
        if c.attrs_units in ds[vi_name].attrs:
            if (vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and\
               (ds[vi_name].attrs[c.attrs_units] == c.unit_K):
                ds = ds - c.d_KC
                ds[vi_name].attrs[c.attrs_units] = c.unit_C
            elif vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
                ds = ds * c.spd
                ds[vi_name].attrs[c.attrs_units] = c.unit_mm
            elif vi_name in [c.v_uas, c.v_vas, c.v_sfcwindmax]:
                ds = ds * c.km_h_per_m_s
                ds[vi_name].attrs[c.attrs_units] = c.unit_km_h

        if cntx.opt_ra:

            # Statistics are calculated at a point.
            if cntx.p_bounds == "":
                ds = utils.subset_ctrl_pt(ds)

            # Statistics are calculated on a surface.
            else:

                # Clip to geographic boundaries.
                if clip and (cntx.p_bounds != ""):
                    ds = utils.subset_shape(ds)

                # Calculate mean value.
                ds = utils.squeeze_lon_lat(ds)

        # Skip simulation if there is no data.
        if len(ds[c.dim_time]) == 0:
            continue

        # Records values.
        # Simulation data is assumed to be complete.
        if ds[vi_name].time.size == n_time:
            if (c.dim_lon in ds.dims) or (c.dim_rlon in ds.dims) or (c.dim_longitude in ds.dims):
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
                        except Exception as e:
                            fu.log(str(e))
                            pass
            if cntx.opt_ra and (vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]):
                vals = [i * c.spd for i in vals]
            arr_vals.append(vals)

    # Calculate the mean value of all years.
    if per_region:
        for i in range(len(arr_vals)):
            arr_vals[i] = [np.nanmean(arr_vals[i])]
        n_time = 1

    # Transpose.
    arr_vals_t = []

    # Collapse to yearly frequency.
    if (freq_in == c.freq_D) and (freq_out == c.freq_YS) and cntx.opt_ra:
        for i_sim in range(n_sim):
            vals_sim = []
            for i_year in range(0, year_n - year_1 + 1):
                vals_year = arr_vals[i_sim][(365 * i_year):(365 * (i_year + 1))]
                da_vals = xr.DataArray(np.array(vals_year))
                vals_year = list(da_vals[np.isnan(da_vals.values) == False].values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
                        val_year = np.nansum(vals_year)
                    else:
                        val_year = np.nanmean(vals_year)
                vals_sim.append(val_year)
            if (vi_name != c.v_pr) | (sum(vals_sim) > 0):
                arr_vals_t.append(vals_sim)
        arr_vals = arr_vals_t
        n_time = year_n - year_1 + 1
    n_sim = len(arr_vals)

    # Transpose array.
    arr_vals_t = []
    for i_time in range(n_time):
        vals = []
        for i_sim in range(n_sim):
            if (vi_name != c.i_rain_season_prcptot) or\
               ((vi_name == c.i_rain_season_prcptot) and (max(arr_vals[i_sim]) > 0)):
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
                                  ((vi_name != c.v_pr) | (sum(da_vals.values) > 0))].values)
        if (stat.code == c.stat_min) or ((stat.code == c.stat_quantile) and (stat.quantile == 0)):
            val_stat = np.min(vals_t)
        elif (stat.code == c.stat_max) or ((stat.code == c.stat_quantile) and (stat.quantile == 1)):
            val_stat = np.max(vals_t)
        elif stat.code == c.stat_mean:
            val_stat = np.mean(vals_t)
        elif stat.code == c.stat_quantile:
            val_stat = np.quantile(vals_t, stat.quantile)
        elif stat.code == c.stat_sum:
            val_stat = np.sum(vals_t)
        arr_stat.append(val_stat)

    # Build dataset.
    da_stat = xr.DataArray(np.array(arr_stat), name=vi_name, coords=[(c.dim_time, np.arange(n_time))])
    ds_stat = da_stat.to_dataset()
    ds_stat = ds_stat.expand_dims(lon=1, lat=1)

    # Adjust coordinates and time.
    if not(c.dim_lon in ds_stat.dims):
        ds_stat[c.dim_lon] = lon
        ds_stat[c.dim_lat] = lat
    if freq_out == c.freq_YS:
        ds_stat[c.dim_time] = utils.reset_calendar(ds_stat, year_1, year_n, freq_out)

    # Adjust units.
    ds_stat[vi_name].attrs[c.attrs_units] = units

    return ds_stat


def calc_stat_tbl(
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

    varidx = VarIdx(vi_code_l[i_vi_proc])

    cat = c.cat_scen if varidx.is_var else c.cat_idx
    if (cat == c.cat_scen) or (not varidx.is_group):
        vi_code_l = [varidx.code]
        vi_name_l = [varidx.name]
    else:
        vi_code_l = VI.explode_idx_l([varidx.code])
        vi_name_l = VI.explode_idx_l([varidx.name])

    # List emission scenarios.
    rcps = RCPs([c.ref] + cntx.rcps.code_l)
    if rcps.count > 2:
        rcps.add(c.rcpxx)

    # Data frequency.
    freq = c.freq_D if cat == c.cat_scen else c.freq_YS

    # Loop through stations.
    stns = (cntx.stns if not cntx.opt_ra else [cntx.obs_src])
    for stn in stns:

        # Loop through variables or indices.
        # There should be a single item in the list, unless 'vi_code' is a group of indices.
        for i_vi in range(len(vi_name_l)):
            vi_code = vi_code_l[i_vi]
            vi_name = vi_name_l[i_vi]
            vi_code_grp = str(VI.group(vi_code)) if varidx.is_group else vi_code

            # Skip iteration if the file already exists.
            p_csv = cntx.d_scen(stn, c.cat_stat, cat + cntx.sep + vi_code_grp) + vi_name + c.f_ext_csv
            if os.path.exists(p_csv) and (not cntx.opt_force_overwrite):
                continue

            # Containers.
            stn_l  = []
            rcp_l  = []
            hor_l  = []
            stat_l = []
            q_l    = []
            val_l  = []

            # Loop through emission scenarios.
            for rcp in rcps.items:

                # Select years.
                if rcp.code == c.ref:
                    hors = Hors([cntx.per_ref])
                    if cat == c.cat_scen:
                        d = os.path.dirname(cntx.p_obs(stn, vi_code_grp))
                    else:
                        d = cntx.d_idx(stn, vi_code_grp)
                else:
                    hors = Hors(cntx.per_hors)
                    if cat == c.cat_scen:
                        d = cntx.d_scen(stn, c.cat_qqmap, vi_code_grp)
                    else:
                        d = cntx.d_idx(stn, vi_code_grp)

                if not os.path.isdir(d):
                    continue

                idx_desc = vi_code
                if vi_code_grp != vi_code:
                    idx_desc = vi_code_grp + "." + idx_desc
                fu.log("Processing: " + stn + ", " + idx_desc + ", " + rcp.code + "", True)

                # Loop through statistics.
                stats = Stats([c.stat_mean])
                if rcp.code != c.ref:
                    stats.add([c.stat_min, c.stat_max, c.stat_quantile], inplace=True)
                for stat in stats.items:
                    quantile_l = cntx.opt_stat_quantiles
                    if stat.code != c.stat_quantile:
                        quantile_l = [-1]
                    for quantile in quantile_l:
                        if ((quantile <= 0) or (quantile >= 1)) and (quantile != -1):
                            continue
                        stat.quantile = quantile

                        # Loop through horizons.
                        for hor in hors.items:

                            # Calculate statistics.
                            ds_stat = calc_stat(stn, varidx, rcp, hor, stat,
                                                (freq == c.freq_YS) and (cat == c.cat_scen),
                                                freq, c.freq_YS, cntx.opt_stat_clip)
                            if ds_stat is None:
                                continue

                            # Select period.
                            if cntx.opt_ra:
                                ds_stat_hor = utils.sel_period(ds_stat.squeeze(), [hor.year_1, hor.year_2])
                            else:
                                ds_stat_hor = ds_stat.copy(deep=True)

                            # Extract value.
                            if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
                                val = float(ds_stat_hor[vi_name].sum()) / (hor.year_2 - hor.year_1 + 1)
                            else:
                                val = float(ds_stat_hor[vi_name].mean())

                            # Add row.
                            stn_l.append(stn)
                            rcp_l.append(rcp.code)
                            hor_l.append(hor.code)
                            stat_l.append("none" if rcp.code == c.ref else stat.code)
                            q_l.append(str(quantile))
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
                    {"stn": stn_l, ("var" if cat == c.cat_scen else "idx"): [vi_name] * len(stn_l),
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

    varidx = VarIdx(vi_code_l[i_vi_proc])
    vi_code_grp = str(VI.group(varidx.code)) if varidx.is_group else varidx.code

    cat = c.cat_scen if varidx.is_var else c.cat_idx

    # Loop through stations.
    stns = cntx.stns if not cntx.opt_ra else [cntx.obs_src]
    for stn in stns:

        # Status message.
        msg = varidx.code
        if varidx.is_group:
            msg = vi_code_grp + "." + msg
        fu.log("Processing: " + stn + ", " + msg, True)

        # Path of files to be created.
        cat_fig = c.cat_fig + cntx.sep + cat + cntx.sep + view_code
        fn = varidx.name + "_" + dash_plot.mode_rcp
        # CSV files:
        p_rcp_csv = cntx.d_scen(stn, cat_fig, vi_code_grp + "_" + c.f_csv) + fn + c.f_ext_csv
        p_sim_csv = p_rcp_csv.replace("_" + dash_plot.mode_rcp, "_" + dash_plot.mode_sim)
        p_rcp_del_csv = p_rcp_csv.replace(c.f_ext_csv, "_delta" + c.f_ext_csv)
        p_sim_del_csv = p_sim_csv.replace(c.f_ext_csv, "_delta" + c.f_ext_csv)
        # PNG files.
        p_rcp_fig = cntx.d_scen(stn, cat_fig, vi_code_grp) + fn + c.f_ext_png
        p_sim_fig = p_rcp_fig.replace("_" + dash_plot.mode_rcp + c.f_ext_png, "_" + dash_plot.mode_sim + c.f_ext_png)
        p_rcp_del_fig = p_rcp_fig.replace(c.f_ext_png, "_delta" + c.f_ext_png)
        p_sim_del_fig = p_sim_fig.replace(c.f_ext_png, "_delta" + c.f_ext_png)

        # Skip if no work required.
        save_csv = ((not os.path.exists(p_rcp_csv) or
                     not os.path.exists(p_sim_csv) or
                     not os.path.exists(p_rcp_del_csv) or
                     not os.path.exists(p_sim_del_csv)) and
                    (c.f_csv in cntx.opt_ts_format))
        save_fig = ((not os.path.exists(p_rcp_fig) or
                     not os.path.exists(p_sim_fig) or
                     not os.path.exists(p_rcp_del_fig) or
                     not os.path.exists(p_sim_del_fig)) and
                    (c.f_png in cntx.opt_ts_format))
        if (not save_csv) and (not save_fig) and (not cntx.opt_force_overwrite):
            continue

        # Update context.
        cntx.code   = c.platform_script
        cntx.view   = View(view_code)
        cntx.lib    = Lib(c.lib_mat)
        cntx.varidx = VarIdx(varidx.code)
        cntx.rcp    = RCP(c.rcpxx)
        cntx.sim    = Sim("")

        # Attempt loading CSV files into dataframes.
        if os.path.exists(p_rcp_csv) and os.path.exists(p_sim_csv) and \
           os.path.exists(p_rcp_del_csv) and os.path.exists(p_sim_del_csv):
            df_rcp = pd.read_csv(p_rcp_csv)
            df_sim = pd.read_csv(p_sim_csv)
            df_rcp_del = pd.read_csv(p_rcp_del_csv)
            df_sim_del = pd.read_csv(p_sim_del_csv)

        # Generate dataframes.
        else:
            df_rcp, df_sim, df_rcp_del, df_sim_del = calc_ts_prep(stn)

        # CSV file format ------------------------------------------------------------------------------------------

        if c.f_csv in cntx.opt_ts_format:
            if (not os.path.exists(p_rcp_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_rcp, p_rcp_csv)
            if (not os.path.exists(p_sim_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_sim, p_sim_csv)
            if (not os.path.exists(p_rcp_del_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_rcp_del, p_rcp_del_csv)
            if (not os.path.exists(p_sim_del_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_sim_del, p_sim_del_csv)

        # PNG file format ------------------------------------------------------------------------------------------

        if c.f_png in cntx.opt_ts_format:

            # Loop through modes (rcp|sim).
            for mode in [dash_plot.mode_rcp, dash_plot.mode_sim]:

                # Loop through delta.
                for delta in ["False", "True"]:
                    cntx.delta = Delta(delta)

                    # Select dataset and output file name.
                    if mode == dash_plot.mode_rcp:
                        if delta == "False":
                            df_mode = df_rcp
                            p_fig = p_rcp_fig
                        else:
                            df_mode = df_rcp_del
                            p_fig = p_rcp_del_fig
                    else:
                        if delta == "False":
                            df_mode  = df_sim
                            p_fig = p_sim_fig
                        else:
                            df_mode = df_sim_del
                            p_fig = p_sim_del_fig

                    # Generate plot.
                    if (not os.path.exists(p_fig)) or cntx.opt_force_overwrite:
                        fig = dash_plot.gen_ts(df_mode, mode)
                        fu.save_plot(fig, p_fig)


def calc_ts_prep(
    stn: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Prepare datasets for the generation of  time series.

    Parameters
    ----------
    stn: str
        Station.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Two dataframes (rcp|sim).
    --------------------------------------------------------------------------------------------------------------------
    """

    df_rcp, df_sim, df_rcp_del, df_sim_del = None, None, None, None

    # Extract relevant information from the context.
    vi_code = cntx.varidx.code
    vi_name = cntx.varidx.name
    vi_code_grp = VI.group(vi_code) if cntx.varidx.is_group else vi_code
    vi_name_grp = VI.group(vi_name) if cntx.varidx.is_group else vi_name
    view_code = cntx.view.code

    # Emission scenarios.
    rcps = RCPs([c.ref] + cntx.rcps.code_l)

    for delta in [False, True]:

        # Loop through emission scenarios.
        ds_ref = None
        ds_rcp26, ds_rcp26_grp, rcp26_name_l = [], [], []
        ds_rcp45, ds_rcp45_grp, rcp45_name_l = [], [], []
        ds_rcp85, ds_rcp85_grp, rcp85_name_l = [], [], []
        ds_rcpxx, ds_rcpxx_grp, rcpxx_name_l = [], [], []
        for rcp in rcps.items:

            # List files.
            if cntx.varidx.is_var:
                p_ref = cntx.d_stn(vi_code_grp) + vi_name_grp + "_" + stn + c.f_ext_nc
            else:
                p_ref = cntx.d_idx(stn, vi_code_grp) + vi_name_grp + "_ref" + c.f_ext_nc
            if rcp.code == c.ref:
                p_sim_l = [p_ref]
            else:
                if cntx.varidx.is_var:
                    d = cntx.d_scen(stn, c.cat_qqmap, vi_code_grp)
                else:
                    d = cntx.d_idx(stn, vi_code_grp)
                p_sim_l = glob.glob(d + "*_" + ("*" if rcp.code == c.rcpxx else rcp.code) + c.f_ext_nc)

            # Exit if there is no file corresponding to the criteria.
            if (len(p_sim_l) == 0) or ((len(p_sim_l) > 0) and not (os.path.isdir(os.path.dirname(p_sim_l[0])))):
                continue

            # Loop through simulation files.
            for i_sim in range(len(p_sim_l)):

                # Extract simulation name.
                if "rcp" not in p_sim_l[i_sim]:
                    sim_code = RCP(c.ref).desc
                else:
                    tokens = os.path.basename(p_sim_l[i_sim]).replace(c.f_ext_nc, "").split("_")
                    n_tokens = len(tokens)
                    sim_code = (tokens[n_tokens - 4] + "_" + tokens[n_tokens - 3] + "_" +
                                tokens[n_tokens - 2] + "_" + tokens[n_tokens - 1])

                # Calculate dataset.
                if (view_code == c.view_ts_bias) and cntx.varidx.is_var:
                    p_val = p_sim_l[i_sim]
                    if rcp.code != c.ref:
                        p_val = p_val.replace(cntx.sep + c.cat_qqmap, cntx.sep + c.cat_regrid).\
                            replace(c.f_ext_nc, "_4" + c.cat_qqmap + c.f_ext_nc)
                    ds_val = calc_ts_ds(p_val)
                    if not delta:
                        ds = ds_val
                    else:
                        ds = xr.zeros_like(ds_val)
                        if rcp.code != c.ref:
                            ds_del = calc_ts_ds(p_sim_l[i_sim])
                            ds[vi_name] = ds_val[vi_name] - ds_del[vi_name]
                else:
                    ds_val = calc_ts_ds(p_sim_l[i_sim])
                    if not delta:
                        ds = ds_val
                    else:
                        ds = xr.zeros_like(ds_val)
                        if rcp.code != c.ref:
                            ds_del = calc_ts_ds(p_ref)
                            ds[vi_name] = ds_val[vi_name] - float(ds_del[vi_name].mean().values)
                ds[vi_name].attrs[c.attrs_units] = ds_val[vi_name].attrs[c.attrs_units]

                # Append to list of datasets.
                if rcp.code == c.ref:
                    ds_ref = ds
                else:
                    ds_is_ok = True
                    if (vi_name == c.i_rain_season_prcptot) and (float(ds[vi_name].max().values) == 0):
                        ds_is_ok = False
                    if ds_is_ok:
                        if rcp.code == c.rcp26:
                            ds_rcp26.append(ds)
                            rcp26_name_l.append(sim_code)
                        elif rcp.code == c.rcp45:
                            ds_rcp45.append(ds)
                            rcp45_name_l.append(sim_code)
                        elif rcp.code == c.rcp85:
                            ds_rcp85.append(ds)
                            rcp85_name_l.append(sim_code)
                        ds_rcpxx.append(ds)
                        rcpxx_name_l.append(sim_code)

            # Group by RCP.
            if rcp.code != c.ref:
                if (rcp.code == c.rcp26) and (len(ds_rcp26) > 0):
                    ds_rcp26_grp = calc_stat_mean_min_max(ds_rcp26, vi_name)
                elif (rcp.code == c.rcp45) and (len(ds_rcp45) > 0):
                    ds_rcp45_grp = calc_stat_mean_min_max(ds_rcp45, vi_name)
                elif (rcp.code == c.rcp85) and (len(ds_rcp85) > 0):
                    ds_rcp85_grp = calc_stat_mean_min_max(ds_rcp85, vi_name)
                if len(ds_rcpxx) > 0:
                    ds_rcpxx_grp = calc_stat_mean_min_max(ds_rcpxx, vi_name)

        if (ds_ref is not None) or (ds_rcp26 != []) or (ds_rcp45 != []) or (ds_rcp85 != []):

            # Extract years.
            years = []
            if (c.rcp26 in rcps.code_l) and (len(ds_rcp26_grp) > 0):
                years = utils.extract_date_field(ds_rcp26_grp[0], "year")
            elif (c.rcp45 in rcps.code_l) and (len(ds_rcp45_grp) > 0):
                years = utils.extract_date_field(ds_rcp45_grp[0], "year")
            elif (c.rcp85 in rcps.code_l) and (len(ds_rcp85_grp) > 0):
                years = utils.extract_date_field(ds_rcp85_grp[0], "year")
            dict_pd = {"year": years}

            # Create pandas dataframe (absolute values).
            df_rcp_x, df_sim_x = pd.DataFrame(dict_pd), pd.DataFrame(dict_pd)
            df_rcp_x[c.ref], df_sim_x[c.ref] = None, None
            for rcp in rcps.items:
                if rcp.code == c.ref:
                    years = utils.extract_date_field(ds_ref, "year")
                    vals = ds_ref[vi_name].values
                    for i in range(len(vals)):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                            df_rcp_x[c.ref][df_rcp_x["year"] == years[i]] = vals[i]
                            df_sim_x[c.ref][df_sim_x["year"] == years[i]] = vals[i]
                elif rcp.code == c.rcp26:
                    df_rcp_x[c.rcp26 + "_moy"] = ds_rcp26_grp[0][vi_name].values
                    df_rcp_x[c.rcp26 + "_min"] = ds_rcp26_grp[1][vi_name].values
                    df_rcp_x[c.rcp26 + "_max"] = ds_rcp26_grp[2][vi_name].values
                    for i in range(len(ds_rcp26)):
                        df_sim_x[rcp26_name_l[i]] = ds_rcp26[i][vi_name].values
                elif rcp.code == c.rcp45:
                    df_rcp_x[c.rcp45 + "_moy"] = ds_rcp45_grp[0][vi_name].values
                    df_rcp_x[c.rcp45 + "_min"] = ds_rcp45_grp[1][vi_name].values
                    df_rcp_x[c.rcp45 + "_max"] = ds_rcp45_grp[2][vi_name].values
                    for i in range(len(ds_rcp45)):
                        df_sim_x[rcp45_name_l[i]] = ds_rcp45[i][vi_name].values
                elif rcp.code == c.rcp85:
                    df_rcp_x[c.rcp85 + "_moy"] = ds_rcp85_grp[0][vi_name].values
                    df_rcp_x[c.rcp85 + "_min"] = ds_rcp85_grp[1][vi_name].values
                    df_rcp_x[c.rcp85 + "_max"] = ds_rcp85_grp[2][vi_name].values
                    for i in range(len(ds_rcp85)):
                        df_sim_x[rcp85_name_l[i]] = ds_rcp85[i][vi_name].values
                else:
                    df_rcp_x[c.rcpxx + "_moy"] = ds_rcpxx_grp[0][vi_name].values
                    df_rcp_x[c.rcpxx + "_min"] = ds_rcpxx_grp[1][vi_name].values
                    df_rcp_x[c.rcpxx + "_max"] = ds_rcpxx_grp[2][vi_name].values

            # Record.
            if not delta:
                df_rcp = df_rcp_x.copy()
                df_sim = df_sim_x.copy()
            else:
                df_rcp_del = df_rcp_x.copy()
                df_sim_del = df_sim_x.copy()

    return df_rcp, df_sim, df_rcp_del, df_sim_del


def calc_ts_ds(
    p: str
) -> Union[xr.Dataset, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    As part of the caculation of time series, get dataset.

    Parameters
    ----------
    p: str
        Path of NetCDF file to use if using absolute values (not delta).

    Returns
    -------
    Union[xr.Dataset, None]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    if not os.path.exists(p):
        return None

    vi_name = cntx.varidx.name

    # Load dataset.
    ds = fu.open_netcdf(p).squeeze()
    if (c.ref in p) and cntx.varidx.is_var:
        ds = utils.remove_feb29(ds)
        ds = utils.sel_period(ds, cntx.per_ref)

    # Records units.
    if cntx.varidx.ens == c.ens_cordex:
        units = ds[vi_name].attrs[c.attrs_units]
    else:
        units = ds.attrs[c.attrs_units]
    if units == "degree_C":
        units = c.unit_C

    # Select control point.
    if cntx.opt_ra:
        if cntx.p_bounds == "":
            ds = utils.subset_ctrl_pt(ds)
        else:
            ds = utils.squeeze_lon_lat(ds, vi_name)

    # First and last years.
    year_1 = int(str(ds.time.values[0])[0:4])
    year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
    year_1 = max(year_1, cntx.per_ref[0])
    if ("rcp" not in p) or (cntx.view.code == c.view_ts_bias):
        year_n = min(year_n, cntx.per_ref[1])
    else:
        year_n = min(year_n, cntx.per_fut[1])

    # Select years.
    ds = ds.sel(time=slice(str(year_1), str(year_n)))

    # Calculate statistics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
        else:
            ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
    n_time = len(ds[c.dim_time].values)
    da = xr.DataArray(np.array(ds[vi_name].values), name=vi_name, coords=[(c.dim_time, np.arange(n_time))])

    # Create dataset.
    ds = da.to_dataset()
    ds[c.dim_time] = utils.reset_calendar_l(list(range(year_1, year_n + 1)))
    ds[vi_name].attrs[c.attrs_units] = units

    # Convert units.
    if vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
        if ds[vi_name].attrs[c.attrs_units] == c.unit_K:
            ds = ds - c.d_KC
            ds[vi_name].attrs[c.attrs_units] = c.unit_C
    elif vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
        if ds[vi_name].attrs[c.attrs_units] == c.unit_kg_m2s1:
            ds = ds * c.spd
            ds[vi_name].attrs[c.attrs_units] = c.unit_mm
    elif vi_name in [c.v_uas, c.v_vas, c.v_sfcwindmax]:
        ds = ds * c.km_h_per_m_s
        ds[vi_name].attrs[c.attrs_units] = c.unit_km_h

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
    units = ds_l[0][vi_name].attrs[c.attrs_units]
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
        da = xr.DataArray(np.array(arr_vals), name=vi_name, coords=[(c.dim_time, np.arange(n_time))])
        ds = da.to_dataset()
        ds[c.dim_time] = utils.reset_calendar(ds, year_1, year_n, c.freq_YS)
        ds[vi_name].attrs[c.attrs_units] = units

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
    if c.dim_rlon in ds.dims:
        da_m = ds[var].rename({c.dim_rlon: c.dim_longitude, c.dim_rlat: c.dim_latitude})
    elif c.dim_lon in ds.dims:
        da_m = ds[var].rename({c.dim_lon: c.dim_longitude, c.dim_lat: c.dim_latitude})
    else:
        da_m = ds[var]

    # Grouping frequency.
    freq_str = "time.month" if freq == c.freq_MS else "time.dayofyear"
    time_str = "M" if freq == c.freq_MS else "1D"

    # Summarize data per month.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        if cntx.opt_ra:
            da_m = da_m.mean(dim={c.dim_longitude, c.dim_latitude})
        if freq != c.freq_MS:

            # Extract values.
            if var in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
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
                if not cntx.opt_ra:
                    ds_m = ds_m.squeeze()

                ds_m[var].attrs[c.attrs_units] = ds[var].attrs[c.attrs_units]
                ds_l.append(ds_m)

        else:

            # Calculate statistics for each month-year combination.
            # Looping through years is less efficient but required to avoid a problem with incomplete at-a-station
            # datasets.
            arr_all = []
            for m in range(1, 13):
                vals_m = []
                for y in range(per[0], per[1] + 1):
                    if var in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
                        val_m_y = np.nansum(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    else:
                        val_m_y = np.nanmean(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    vals_m.append(val_m_y)
                arr_all.append(vals_m)

            # Create dataset.
            dict_pd = {var: arr_all}
            df = pd.DataFrame(dict_pd)
            ds_l = df.to_xarray()
            ds_l.attrs[c.attrs_units] = ds[var].attrs[c.attrs_units]

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

    varidx = VarIdx(vi_code_l[i_vi_proc])

    # Extract variable name, group and RCPs.
    cat = c.cat_scen if varidx.is_var else c.cat_idx
    varidx = VarIdx(varidx.code)
    vi_name = varidx.name
    vi_code_grp = VI.group(varidx.code) if varidx.is_group else varidx.code
    rcps = RCPs([c.ref, c.rcpxx] + cntx.rcps.code_l)

    stn = "stns" if not cntx.opt_ra else cntx.obs_src

    msg = "Calculating maps."
    fu.log(msg, True)

    # Get statistic string.
    def stat_str(
        _stat: Stat
    ):

        if _stat.code in [c.stat_mean, c.stat_min, c.stat_max]:
            _stat_str = _stat.code
        else:
            _stat_str = "q" + str(int(_stat.quantile * 100)).rjust(2, "0")

        return _stat_str

    # Get the path of the CSV and PNG files.
    def p_csv_fig(
        _rcp: RCP,
        _per: List[int],
        _per_str: str,
        _stat: Stat,
        _delta: Optional[bool] = False
    ) -> Tuple[str, str]:

        # PNG file.
        _d_fig = cntx.d_scen(stn, c.cat_fig + cntx.sep + cat + cntx.sep + c.view_map,
                             (vi_code_grp + cntx.sep if vi_code_grp != varidx.code else "") +
                             varidx.code + cntx.sep + per_str)
        _stat_str = stat_str(_stat)
        _fn_fig = vi_name + "_" + _rcp.code + "_" + str(_per[0]) + "_" + str(_per[1]) + "_" + _stat_str + c.f_ext_png
        _p_fig = _d_fig + _fn_fig
        if _delta:
            _p_fig = _p_fig.replace(c.f_ext_png, "_delta" + c.f_ext_png)

        # CSV file.
        _d_csv = cntx.d_scen(stn, c.cat_fig + cntx.sep + cat + cntx.sep + c.view_map,
                             (vi_code_grp + cntx.sep if vi_code_grp != varidx.code else "") +
                             varidx.code + "_csv" + cntx.sep + _per_str)
        _fn_csv = _fn_fig.replace(c.f_ext_png, c.f_ext_csv)
        _p_csv = _d_csv + _fn_csv
        if _delta:
            _p_csv = _p_csv.replace(c.f_ext_csv, "_delta" + c.f_ext_csv)

        return _p_csv, _p_fig

    # Prepare data -----------------------------------------------------------------------------------------------------

    # Calculate the overall minimum and maximum values (considering all maps for the current 'varidx').
    # There is one z-value per period.
    n_per = len(cntx.per_hors)
    z_min_net = [np.nan] * n_per
    z_max_net = [np.nan] * n_per
    z_min_del = [np.nan] * n_per
    z_max_del = [np.nan] * n_per

    # Calculated datasets and item description.
    arr_ds_maps = []
    arr_items = []

    # Identify statistics to calculate.
    stats = Stats([c.stat_mean])
    if cntx.opt_map_quantiles is not None:
        for quantile in cntx.opt_map_quantiles:
            stats.add(Stat(c.stat_quantile, quantile))

    # Reference map (to calculate deltas).
    ds_map_ref = None

    # Loop through statistics.
    for stat in stats.items:

        # Loop through categories of emission scenarios.
        for j in range(rcps.count):
            rcp = rcps.items[j]

            # Loop through horizons.
            pers = [cntx.per_ref] if (j == 0) else cntx.per_hors
            for k_per in range(len(pers)):
                per = pers[k_per]
                per_str = str(per[0]) + "-" + str(per[1])

                # Path of CSV and PNG files.
                p_csv, p_fig = p_csv_fig(rcp, per, per_str, stat)

                if os.path.exists(p_csv) and not cntx.opt_force_overwrite:

                    # Load data.
                    df = pd.read_csv(p_csv)

                    # Convert to DataArray.
                    df = pd.DataFrame(df, columns=["longitude", "latitude", varidx.code])
                    df = df.sort_values(by=["latitude", "longitude"])
                    lat = list(set(df["latitude"]))
                    lat.sort()
                    lon = list(set(df["longitude"]))
                    lon.sort()
                    arr = np.reshape(list(df[varidx.code]), (len(lat), len(lon)))
                    da = xr.DataArray(data=arr, dims=["latitude", "longitude"],
                                      coords=[("latitude", lat), ("longitude", lon)])
                    da.name = vi_name
                    ds_map = da.to_dataset()

                else:

                    # Calculate map.
                    ds_map = calc_map_rcp(varidx, rcp, per, stat)

                    # Clip to geographic boundaries.
                    if cntx.opt_map_clip and (cntx.p_bounds != ""):
                        ds_map = utils.subset_shape(ds_map)

                # Record current map, statistic and quantile, RCP and period.
                arr_ds_maps.append(ds_map)
                arr_items.append([stat.code, stat.quantile, rcp.code, per])

                # Calculate reference map.
                if (ds_map_ref is None) and (stat.code == c.stat_mean) and (rcp.code == c.rcpxx) and\
                   (per == cntx.per_ref):
                    ds_map_ref = ds_map

                # Extract values.
                vals_net = ds_map[vi_name].values
                vals_del = None
                if per != cntx.per_ref:
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
        stat  = Stat(arr_items[i][0], arr_items[i][1])
        rcp   = RCP(arr_items[i][2])
        per   = arr_items[i][3]
        i_per = cntx.per_hors.index(per)

        # Perform twice (for values, then for deltas).
        for j in range(2):

            # Skip if delta map generation if the option is disabled or if it's the reference period.
            if (j == 1) and\
                ((per == cntx.per_ref) or
                 ((cat == c.cat_scen) and (not cntx.opt_map_delta[0])) or
                 ((cat == c.cat_idx) and (not cntx.opt_map_delta[1]))):
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
                    per_str = str(cntx.per_ref[0]) + "-" + str(cntx.per_hors[len(cntx.per_hors) - 1][1])
                    z_min = np.nanmin(z_min_net if j == 0 else z_min_del)
                    z_max = np.nanmax(z_max_net if j == 0 else z_max_del)

                # PNG and CSV formats ----------------------------------------------------------------------------------

                # Path of output files.
                p_csv, p_fig = p_csv_fig(rcp, per, per_str, stat, j == 1)

                # Create context.
                cntx.code              = c.platform_script
                cntx.view              = View(c.view_map)
                cntx.lib               = Lib(c.lib_mat)
                cntx.varidx            = VarIdx(vi_name)
                cntx.project.quantiles = cntx.opt_map_quantiles
                cntx.rcp               = rcp
                cntx.hor               = Hor(per)
                cntx.stat              = stat
                cntx.delta             = Delta(str(j == 1))

                # Update colors.
                if len(cntx.opt_map_col_temp_var) > 0:
                    cntx.opt_map_col_temp_var = cntx.opt_map_col_temp_var
                if len(cntx.opt_map_col_temp_idx_1) > 0:
                    cntx.opt_map_col_temp_idx_1 = cntx.opt_map_col_temp_idx_1
                if len(cntx.opt_map_col_temp_idx_2) > 0:
                    cntx.opt_map_col_temp_idx_2 = cntx.opt_map_col_temp_idx_2
                if len(cntx.opt_map_col_prec_var) > 0:
                    cntx.opt_map_col_prec_var = cntx.opt_map_col_prec_var
                if len(cntx.opt_map_col_prec_idx_1) > 0:
                    cntx.opt_map_col_prec_idx_1 = cntx.opt_map_col_prec_idx_1
                if len(cntx.opt_map_col_prec_idx_2) > 0:
                    cntx.opt_map_col_prec_idx_2 = cntx.opt_map_col_prec_idx_2
                if len(cntx.opt_map_col_prec_idx_3) > 0:
                    cntx.opt_map_col_prec_idx_3 = cntx.opt_map_col_prec_idx_3
                if len(cntx.opt_map_col_wind_var) > 0:
                    cntx.opt_map_col_wind_var = cntx.opt_map_col_wind_var
                if len(cntx.opt_map_col_wind_idx_1) > 0:
                    cntx.opt_map_col_wind_idx_1 = cntx.opt_map_col_wind_idx_1
                if len(cntx.opt_map_col_default) > 0:
                    cntx.opt_map_col_default = cntx.opt_map_col_default

                # Determine if PNG and CSV files need to be saved.
                save_fig = cntx.opt_force_overwrite or\
                    ((not os.path.exists(p_fig)) and (c.f_png in cntx.opt_map_format))
                save_csv = cntx.opt_force_overwrite or\
                    ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_map_format))

                # Create dataframe.
                arr_lon, arr_lat, arr_val = [], [], []
                for m in range(len(da_map.longitude.values)):
                    for n in range(len(da_map.latitude.values)):
                        arr_lon.append(da_map.longitude.values[m])
                        arr_lat.append(da_map.latitude.values[n])
                        arr_val.append(da_map.values[n, m])
                dict_pd = {c.dim_longitude: arr_lon, c.dim_latitude: arr_lat, vi_name: arr_val}
                df = pd.DataFrame(dict_pd)

                # Generate plot and save PNG file.
                if save_fig:
                    fig = dash_plot.gen_map(df, [z_min, z_max])
                    fu.save_plot(fig, p_fig)

                # Save CSV file.
                if save_csv:
                    fu.save_csv(df, p_csv)

                # TIF format -------------------------------------------------------------------------------------------

                # Path of TIF file.
                p_tif = p_fig.replace(varidx.code + cntx.sep, varidx.code + "_" + c.f_tif + cntx.sep).\
                    replace(c.f_ext_png, c.f_ext_tif)

                if (c.f_tif in cntx.opt_map_format) and ((not os.path.exists(p_tif)) or cntx.opt_force_overwrite):

                    # TODO: da_tif.rio.reproject is now crashing. It was working in July 2021.

                    # Increase resolution.
                    da_tif = da_map.copy()
                    if cntx.opt_map_res > 0:
                        lat_vals = np.arange(min(da_tif.latitude), max(da_tif.latitude), cntx.opt_map_res)
                        lon_vals = np.arange(min(da_tif.longitude), max(da_tif.longitude), cntx.opt_map_res)
                        da_tif = da_tif.rename({c.dim_latitude: c.dim_lat, c.dim_longitude: c.dim_lon})
                        da_grid = xr.Dataset(
                            {c.dim_lat: ([c.dim_lat], lat_vals), c.dim_lon: ([c.dim_lon], lon_vals)})
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            da_tif = xe.Regridder(da_tif, da_grid, "bilinear")(da_tif)

                    # Project data.
                    da_tif.rio.set_crs("EPSG:4326")
                    if cntx.opt_map_spat_ref != "EPSG:4326":
                        da_tif.rio.set_spatial_dims(c.dim_lon, c.dim_lat, inplace=True)
                        da_tif = da_tif.rio.reproject(cntx.opt_map_spat_ref)
                        da_tif.values[da_tif.values == -9999] = np.nan
                        da_tif = da_tif.rename({"y": c.dim_lat, "x": c.dim_lon})

                    # Save.
                    d = os.path.dirname(p_tif)
                    if not (os.path.isdir(d)):
                        os.makedirs(d)
                    da_tif.rio.to_raster(p_tif)


def calc_map_rcp(
    varidx: VarIdx,
    rcp: RCP,
    per: [int],
    stat: Stat
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate heat map for a given RCP.

    Parameters
    ----------
    varidx: VarIdx
        Climate variable or index.
    rcp: RCP
        Emission scenario.
    per: [int]
        Period.
    stat: Stat
        Statistic.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract variable name and group.
    vi_code = varidx.code
    vi_name = varidx.name
    vi_code_grp = VI.group(vi_code) if varidx.is_group else vi_code
    vi_name_grp = VI.group(vi_name) if varidx.is_group else vi_name

    fu.log("Processing: " + vi_code + ", " +
           ("q" + str(int(stat.quantile * 100)).rjust(2, "0") if stat.quantile >= 0 else stat.code) + ", " +
           rcp.desc + ", " + str(per[0]) + "-" + str(per[1]) + "", True)

    # Number of years and stations.
    if rcp.code == c.ref:
        n_year = cntx.per_ref[1] - cntx.per_ref[0] + 1
    else:
        n_year = cntx.per_fut[1] - cntx.per_ref[1] + 1

    # List stations.
    stns = cntx.stns if not cntx.opt_ra else [cntx.obs_src]

    # Observations -----------------------------------------------------------------------------------------------------

    if not cntx.opt_ra:

        # Get information on stations.
        p_stn = glob.glob(cntx.d_stn(c.v_tas) + "../*" + c.f_ext_csv)[0]
        df = pd.read_csv(p_stn, sep=cntx.f_sep)

        # Collect values for each station and determine overall boundaries.
        fu.log("Collecting emissions scenarios at each station.", True)
        x_bnds = []
        y_bnds = []
        data_stn = []
        for stn in stns:

            # Get coordinates.
            lon = df[df["station"] == stn][c.dim_lon].values[0]
            lat = df[df["station"] == stn][c.dim_lat].values[0]

            # Calculate statistics.
            ds_stat = calc_stat(stn, varidx, rcp, None, stat, False, c.freq_YS, c.freq_YS, cntx.opt_map_clip)
            if ds_stat is None:
                continue

            # Extract data from stations.
            data = [[], [], []]
            n = ds_stat.dims[c.dim_time]
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
            x_bnds[i] = round_to_nearest_decimal(x_bnds[i], cntx.idx_resol)
            y_bnds[i] = round_to_nearest_decimal(y_bnds[i], cntx.idx_resol)
        grid_x = np.arange(x_bnds[0], x_bnds[1] + cntx.idx_resol, cntx.idx_resol)
        grid_y = np.arange(y_bnds[0], y_bnds[1] + cntx.idx_resol, cntx.idx_resol)

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
                              coords={c.dim_time: grid_time, c.dim_lat: grid_y, c.dim_lon: grid_x},
                              dims=[c.dim_time, c.dim_lat, c.dim_lon])
        ds_res = da_itp.to_dataset(name=vi_name)

    # There is no need to interpolate.
    else:

        # Reference period.
        if varidx.is_var:
            p_sim_ref =\
                cntx.d_scen(cntx.obs_src, c.cat_obs, vi_code_grp) + vi_name_grp + "_" + cntx.obs_src + c.f_ext_nc
        else:
            p_sim_ref = cntx.d_idx(cntx.obs_src, vi_code_grp) + vi_name_grp + "_ref" + c.f_ext_nc
        if rcp.code == c.ref:

            # Open dataset.
            p_sim = p_sim_ref
            if not os.path.exists(p_sim):
                return xr.Dataset(None)
            ds_sim = fu.open_netcdf(p_sim)

            # TODO: The following patch was added to fix dimensions for c.i_drought_code.
            if vi_code in cntx.idxs.code_l:
                ds_sim = ds_sim.resample(time=c.freq_YS).mean(dim=c.dim_time, keep_attrs=True)

            # Select period.
            ds_sim = utils.remove_feb29(ds_sim)
            ds_sim = utils.sel_period(ds_sim, per)

            # Extract units.
            units = 1
            if c.attrs_units in ds_sim[vi_name].attrs:
                units = ds_sim[vi_name].attrs[c.attrs_units]
            elif c.attrs_units in ds_sim.data_vars:
                units = ds_sim[c.attrs_units]

            # Calculate mean.
            ds_res = ds_sim.mean(dim=c.dim_time)
            if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
                ds_res = ds_res * 365
            if varidx.is_var:
                ds_res[vi_name].attrs[c.attrs_units] = units
            else:
                ds_res.attrs[c.attrs_units] = units

        # Future period.
        else:

            # List scenarios or indices for the current RCP.
            if varidx.is_var:
                d = cntx.d_scen(cntx.obs_src, c.cat_qqmap, vi_code_grp)
            else:
                d = cntx.d_scen(cntx.obs_src, c.cat_idx, vi_code_grp)
            p_sim_l = [i for i in glob.glob(d + "*" + c.f_ext_nc) if i != p_sim_ref]

            # Collect simulations.
            arr_sim = []
            ds_res  = None
            n_sim   = 0
            units   = 1
            for p_sim in p_sim_l:
                if os.path.exists(p_sim) and ((rcp.code in p_sim) or (rcp.code == c.rcpxx)):

                    # Open dataset.
                    ds_sim = fu.open_netcdf(p_sim)

                    # TODO: The following patch was added to fix dimensions for c.i_drought_code.
                    if vi_code in cntx.idxs.code_l:
                        ds_sim = ds_sim.resample(time=c.freq_YS).mean(dim=c.dim_time, keep_attrs=True)

                    # Extract units.
                    if c.attrs_units in ds_sim[vi_name].attrs:
                        units = ds_sim[vi_name].attrs[c.attrs_units]
                    elif c.attrs_units in ds_sim.data_vars:
                        units = ds_sim[c.attrs_units]

                    # Select period.
                    ds_sim = utils.remove_feb29(ds_sim)
                    ds_sim = utils.sel_period(ds_sim, per)

                    # Calculate mean and add to array.
                    ds_sim = ds_sim.mean(dim=c.dim_time)
                    if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
                        ds_sim = ds_sim * 365
                    if varidx.is_var:
                        ds_sim[vi_name].attrs[c.attrs_units] = units
                    else:
                        ds_sim.attrs[c.attrs_units] = units
                    arr_sim.append(ds_sim)

                    # The first dataset will be used to return result.
                    if n_sim == 0:
                        ds_res = ds_sim.copy(deep=True)
                        ds_res[vi_name][:, :] = np.nan
                    n_sim = n_sim + 1

            if ds_res is None:
                return xr.Dataset(None)

            # Get longitude and latitude.
            lon_l, lat_l = utils.coords(arr_sim[0][vi_name], True)

            # Calculate statistics.
            len_lat = len(lat_l)
            len_lon = len(lon_l)
            for i_lat in range(len_lat):
                for j_lon in range(len_lon):

                    # Extract the value for the current cell for all simulations.
                    vals = []
                    for ds_sim_k in arr_sim:
                        dims_sim = ds_sim_k[vi_name].dims
                        if c.dim_rlat in dims_sim:
                            val_sim = float(ds_sim_k[vi_name].isel(rlon=j_lon, rlat=i_lat))
                        elif c.dim_lat in dims_sim:
                            val_sim = float(ds_sim_k[vi_name].isel(lon=j_lon, lat=i_lat))
                        else:
                            val_sim = float(ds_sim_k[vi_name].isel(longitude=j_lon, latitude=i_lat))
                        vals.append(val_sim)

                    # Calculate statistic.
                    val = np.nan
                    if stat.code == c.stat_mean:
                        val = np.mean(vals)
                    elif stat.code == c.stat_min:
                        val = np.min(vals)
                    elif stat.code == c.stat_max:
                        val = np.max(vals)
                    elif stat.code == c.stat_quantile:
                        val = np.quantile(vals, stat.quantile)

                    # Record value.
                    ds_res[vi_name][i_lat, j_lon] = val

        # Remember units.
        units = ds_res[vi_name].attrs[c.attrs_units] if varidx.is_var else ds_res.attrs[c.attrs_units]

        # Convert units.
        if (vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and\
           (ds_res[vi_name].attrs[c.attrs_units] == c.unit_K):
            ds_res = ds_res - c.d_KC
            ds_res[vi_name].attrs[c.attrs_units] = c.unit_C
        elif (vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]) and \
             (ds_res[vi_name].attrs[c.attrs_units] == c.unit_kg_m2s1):
            ds_res = ds_res * c.spd
            ds_res[vi_name].attrs[c.attrs_units] = c.unit_mm
        elif vi_name in [c.v_uas, c.v_vas, c.v_sfcwindmax]:
            ds_res = ds_res * c.km_h_per_m_s
            ds_res[vi_name].attrs[c.attrs_units] = c.unit_km_h
        else:
            ds_res[vi_name].attrs[c.attrs_units] = units

        # Adjust coordinate names (required for clipping).
        if c.dim_longitude not in list(ds_res.dims):
            if c.dim_rlon in ds_res.dims:
                ds_res = ds_res.rename_dims({c.dim_rlon: c.dim_longitude, c.dim_rlat: c.dim_latitude})
                ds_res[c.dim_longitude] = ds_res[c.dim_rlon]
                del ds_res[c.dim_rlon]
                ds_res[c.dim_latitude] = ds_res[c.dim_rlat]
                del ds_res[c.dim_rlat]
            else:
                ds_res = ds_res.rename_dims({c.dim_lon: c.dim_longitude, c.dim_lat: c.dim_latitude})
                ds_res[c.dim_longitude] = ds_res[c.dim_lon]
                del ds_res[c.dim_lon]
                ds_res[c.dim_latitude] = ds_res[c.dim_lat]
                del ds_res[c.dim_lat]

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
    stns = cntx.stns if not cntx.opt_ra else [cntx.obs_src]
    for stn in stns:

        # Loop through categories.
        cat_l = [c.cat_obs, c.cat_raw, c.cat_regrid, c.cat_qqmap]
        if cat == c.cat_idx:
            cat_l = [c.cat_idx]
        for cat in cat_l:

            # Loop through variables or indices.
            vi_code_l = cntx.varidxs.code_l if cat != c.cat_idx else VI.explode_idx_l(cntx.varidxx.code_l)
            for vi_code in vi_code_l:

                # Extract variable/index information.
                varidx = VarIdx(vi_code)
                vi_code_grp = VI.group(vi_code) if varidx.is_group else vi_code

                # List NetCDF files.
                p_l = list(glob.glob(cntx.d_scen(stn, cat, vi_code_grp) + "*" + c.f_ext_nc))
                n_files = len(p_l)
                if n_files == 0:
                    continue
                p_l.sort()

                fu.log("Processing: " + stn + ", " + vi_code, True)

                # Scalar processing mode.
                if cntx.n_proc == 1:
                    for i_file in range(n_files):
                        conv_nc_csv_single(p_l, varidx, i_file)

                # Parallel processing mode.
                else:

                    # Loop until all files have been converted.
                    while True:

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before =\
                            len(list(glob.glob(cntx.d_scen(stn, cat, vi_code) + "*" + c.f_ext_csv)))

                        try:
                            fu.log("Splitting work between " + str(cntx.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=min(cntx.n_proc, len(p_l)))
                            func = functools.partial(conv_nc_csv_single, p_l, varidx)
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
                            len(list(glob.glob(cntx.d_scen(stn, cat, vi_code_grp) + "*" + c.f_ext_csv)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cntx.n_proc == 1) or (n_files_proc_before == n_files_proc_after):
                            break


def conv_nc_csv_single(
    p_l: [str],
    varidx: VarIdx,
    i_file: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a single NetCDF to CSV file.

    Parameters
    ----------
    p_l : [str]
        List of paths.
    varidx : VarIdx
        Climate variable or index.
    i_file : int
        Rank of file in 'p_l'.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract variable name and group.
    vi_code = varidx.code
    vi_name = varidx.name
    vi_code_grp = VI.group(vi_code) if varidx.is_group else vi_code

    # Paths.
    p = p_l[i_file]
    p_csv = p.replace(c.f_ext_nc, c.f_ext_csv).\
        replace(cntx.sep + vi_code_grp + "_", cntx.sep + vi_name + "_").\
        replace(cntx.sep + vi_code_grp + cntx.sep, cntx.sep + vi_code_grp + "_" + c.f_csv + cntx.sep)

    if os.path.exists(p_csv) and (not cntx.opt_force_overwrite):
        if cntx.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Explode lists of codes and names.
    idx_names_exploded = VI.explode_idx_l(cntx.idxs.name_l)

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
    if cntx.opt_ra:
        lon_l, lat_l = utils.coords(ds, True)

    # Extract values.
    # Calculate average values (only if the analysis is based on observations at a station).
    val_l = list(ds[vi_name].values)
    if not cntx.opt_ra:
        if vi_name not in cntx.idxs.name_l:
            for i in range(n_time):
                val_l[i] = val_l[i].mean()
        else:
            if c.ref in p:
                for i in range(n_time):
                    val_l[i] = val_l[i][0][0]
            else:
                val_l = list(val_l[0][0])

    # Convert values to more practical units (if required).
    if (vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]) and\
       (ds[vi_name].attrs[c.attrs_units] == c.unit_kg_m2s1):
        for i in range(n_time):
            val_l[i] = val_l[i] * c.spd
    elif (vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and\
         (ds[vi_name].attrs[c.attrs_units] == c.unit_K):
        for i in range(n_time):
            val_l[i] = val_l[i] - c.d_KC

    # Build pandas dataframe.
    if not cntx.opt_ra:
        dict_pd = {c.dim_time: time_l, vi_name: val_l}
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
        dict_pd = {c.dim_time: time_l_1d, c.dim_lon: lon_l_1d, c.dim_lat: lat_l_1d, vi_name: val_l_1d}
    df = pd.DataFrame(dict_pd)

    # Save CSV file.
    fu.save_csv(df, p_csv)

    if cntx.n_proc > 1:
        fu.log("Work done!", True)


def calc_cycle(
    ds: xr.Dataset,
    stn: str,
    varidx: VarIdx,
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
    varidx: VarIdx
        Climate variable or index.
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
    cat_fig = c.view_cycle_ms if freq == c.freq_MS else c.view_cycle_d
    p_fig = cntx.d_scen(stn, c.cat_fig + cntx.sep + c.cat_scen + cntx.sep + cat_fig, varidx.name)
    if varidx.name in cntx.idxs.name_l:
        p_fig = p_fig.replace(cntx.sep + c.cat_scen, cntx.sep + c.cat_idx)
    p_fig += title + c.f_ext_png
    p_csv = p_fig.replace(cntx.sep + varidx.name + cntx.sep, cntx.sep + varidx.name + "_" + c.f_csv + cntx.sep).\
        replace(c.f_ext_png, c.f_ext_csv)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.f_png in cntx.opt_cycle_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_cycle_format)))
    if not (cntx.opt_cycle and (save_fig or save_csv)):
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Extract data.
        # Exit if there is not at leat one year of data for the current period.
        if i_trial == 1:
            ds = utils.sel_period(ds, per)
            if len(ds[c.dim_time]) == 0:
                return
            if freq == c.freq_D:
                ds = utils.remove_feb29(ds)

        # Convert units.
        units = ds[varidx.name].attrs[c.attrs_units]
        if (varidx.name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and \
           (ds[varidx.name].attrs[c.attrs_units] == c.unit_K):
            ds = ds - c.d_KC
        elif (varidx.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]) and \
             (ds[varidx.name].attrs[c.attrs_units] == c.unit_kg_m2s1):
            ds = ds * c.spd
        ds[varidx.name].attrs[c.attrs_units] = units

        # Calculate statistics.
        ds_l = calc_by_freq(ds, varidx.name, per, freq)

        n = 12 if freq == c.freq_MS else 365

        # Remove February 29th.
        if (freq == c.freq_D) and (len(ds_l[0][varidx.name]) > 365):
            for i in range(3):
                ds_l[i] = ds_l[i].rename_dims({"dayofyear": c.dim_time})
                ds_l[i] = ds_l[i][varidx.name][ds_l[i][c.dim_time] != 59].to_dataset()
                ds_l[i][c.dim_time] = utils.reset_calendar(ds_l[i], cntx.per_ref[0], cntx.per_ref[0], c.freq_D)
                ds_l[i][varidx.name].attrs[c.attrs_units] = ds[varidx.name].attrs[c.attrs_units]

        # Create dataframe.
        if freq == c.freq_D:
            dict_pd = {
                ("month" if freq == c.freq_MS else "day"): range(1, n + 1),
                "mean": list(ds_l[0][varidx.name].values),
                "min": list(ds_l[1][varidx.name].values),
                "max": list(ds_l[2][varidx.name].values), "var": [varidx.name] * n
            }
            df = pd.DataFrame(dict_pd)
        else:
            year_l = list(range(per[0], per[1] + 1))
            dict_pd = {
                "year": year_l,
                "1": ds_l[varidx.name].values[0], "2": ds_l[varidx.name].values[1],
                "3": ds_l[varidx.name].values[2], "4": ds_l[varidx.name].values[3],
                "5": ds_l[varidx.name].values[4], "6": ds_l[varidx.name].values[5],
                "7": ds_l[varidx.name].values[6], "8": ds_l[varidx.name].values[7],
                "9": ds_l[varidx.name].values[8], "10": ds_l[varidx.name].values[9],
                "11": ds_l[varidx.name].values[10], "12": ds_l[varidx.name].values[11]
            }
            df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:

        # Update context.
        cntx.code   = c.platform_script
        cntx.view   = View(c.view_cluster)
        cntx.lib    = Lib(c.lib_mat)
        cntx.delta  = Delta("False")
        cntx.varidx = VarIdx(varidx.name)

        # Generate plot.
        if freq == c.freq_D:
            fig = dash_plot.gen_cycle_d(df)
        else:
            fig = dash_plot.gen_cycle_ms(df)

        # Save plot.
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    error = False
    if save_csv:
        try:
            fu.save_csv(df, p_csv)
        except Exception as e:
            fu.log(str(e))
            error = True

    # Attempt the same analysis again if an error occurred. Remove this option if it's no longer required.
    if error:

        # Log error.
        msg_err = "Unable to save " + ("daily" if (freq == c.freq_D) else "monthly") +\
                  " plot data (failed " + str(i_trial) + " time(s)):"
        fu.log(msg_err, True)
        fu.log(title, True)

        # Attempt the same analysis again.
        if i_trial < 3:
            calc_cycle(ds, stn, varidx, per, freq, title, i_trial + 1)


def calc_clusters():

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate cluster plot.

    One plot each generated from each possible cluster size.
    All simulations are considered (no matter the RCP).
    The only tested variable combination is the one entered in the configuration file.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Update context.
    cntx.view    = View(c.view_cluster)
    cntx.lib     = Lib(c.lib_mat)
    cntx.project.load_quantiles()
    cntx.varidxs = cntx.cluster_vars
    cntx.rcp     = RCP(c.rcpxx)

    # Assemble a string representing the combination of (sorted) variable codes.
    vars_str = ""
    var_code_l = cntx.varidxs.code_l
    var_code_l.sort()
    for i in range(len(var_code_l)):
        if i > 0:
            vars_str += "_"
        vars_str += var_code_l[i]

    # Loop through stations.
    stns = (cntx.stns if not cntx.opt_ra else [cntx.obs_src])
    for stn in stns:

        fu.log("Processing: " + stn, True)

        # Determine the maximum number of clusters.
        p_csv_ts_l = []
        for var in cntx.varidxs.items:
            p_i = cntx.d_scen(stn, c.cat_fig + cntx.sep + c.cat_scen + cntx.sep + c.view_ts,
                              var.code + "_" + c.f_csv) + var.code + "_sim" + c.f_ext_csv
            p_csv_ts_l.append(p_i)
        n_cluster_max = len(dash_utils.get_shared_sims(p_csv_ts_l))

        # Loop through all combinations of
        for n_cluster in range(1, n_cluster_max):

            # Paths.
            p_csv = cntx.d_scen(stn, c.cat_fig + cntx.sep + c.cat_scen, c.view_cluster) +\
                vars_str + "_" + c.f_csv + cntx.sep + vars_str + "_" + str(n_cluster) + c.f_ext_csv
            p_fig = p_csv.replace("_" + c.f_csv, "").replace(c.f_ext_csv, c.f_ext_png)

            # Determine if the analysis is required.
            analysis_enabled = cntx.opt_cluster
            save_fig = (cntx.opt_force_overwrite or
                        ((not os.path.exists(p_fig)) and (c.f_png in cntx.opt_cluster_format)))
            save_csv = (cntx.opt_force_overwrite or
                        ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_cluster_format)))
            if not (analysis_enabled and (save_fig or save_csv)):
                continue

            # Load existing CSV file.
            if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
                df = pd.read_csv(p_csv)

            # Prepare data.
            else:
                df = dash_stats.calc_clusters(n_cluster, p_csv_ts_l)

            # Save CSV file.
            if save_csv:
                try:
                    fu.save_csv(df, p_csv)
                except Exception as e:
                    fu.log(str(e))

            # Generate and save plot.
            if save_fig:

                # Generate plot.
                fig = dash_plot.gen_cluster_plot(n_cluster, p_csv_ts_l)

                # Save plot.
                fu.save_plot(fig, p_fig)


def calc_postprocess(
    p_ref: str,
    p_sim: str,
    p_sim_adj: str,
    varidx: VarIdx,
    p_fig: str,
    title: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate postprocess plots.

    Parameters
    ----------
    p_ref : str
        Path of NetCDF file containing reference data.
    p_sim : str
        Path of NetCDF file containing simulation data.
    p_sim_adj : str
        Path of NetCDF file containing adjusted simulation data.
    varidx : VarIdx
        Variable or index.
    p_fig : str
        Path of output figure.
    title : str
        Title of figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = varidx.name

    # Paths.
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.f_csv + cntx.sep). \
        replace(c.f_ext_png, c.f_ext_csv)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.f_png in cntx.opt_diagnostic_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_diagnostic_format)))
    if not (save_fig or save_csv):
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Load datasets.
        da_ref = fu.open_netcdf(p_ref)[vi_name]
        if c.dim_longitude in da_ref.dims:
            da_ref = da_ref.rename({c.dim_longitude: c.dim_rlon, c.dim_latitude: c.dim_rlat})
        da_sim = fu.open_netcdf(p_sim)[vi_name]
        da_sim_adj = fu.open_netcdf(p_sim_adj)[vi_name]

        # Select control point.
        if cntx.opt_ra:
            subset_ctrl_pt = False
            if c.dim_rlon in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.rlon) > 1) or (len(da_ref.rlat) > 1)
            elif c.dim_lon in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.lon) > 1) or (len(da_ref.lat) > 1)
            elif c.dim_longitude in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.longitude) > 1) or (len(da_ref.latitude) > 1)
            if subset_ctrl_pt:
                if cntx.p_bounds == "":
                    da_ref = utils.subset_ctrl_pt(da_ref)
                    da_sim = utils.subset_ctrl_pt(da_sim)
                    da_sim_adj = utils.subset_ctrl_pt(da_sim_adj)
                else:
                    da_ref = utils.squeeze_lon_lat(da_ref)
                    da_sim = utils.squeeze_lon_lat(da_sim)
                    da_sim_adj = utils.squeeze_lon_lat(da_sim_adj)

        # Conversion coefficient.
        coef = 1
        delta_ref, delta_sim, delta_sim_adj = 0, 0, 0
        if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            coef = c.spd * 365
        elif vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
            if da_ref.units == c.unit_K:
                delta_ref = -c.d_KC
            if da_sim.units == c.unit_K:
                delta_sim = -c.d_KC
            if da_sim_adj.units == c.unit_K:
                delta_sim_adj = -c.d_KC

        # Calculate annual mean values.
        da_sim_adj_mean = None
        if da_sim_adj is not None:
            da_sim_adj_mean = (da_sim_adj * coef + delta_sim_adj).groupby(da_sim_adj.time.dt.year).mean()
        da_sim_mean = (da_sim * coef + delta_sim).groupby(da_sim.time.dt.year).mean()
        da_ref_mean = (da_ref * coef + delta_ref).groupby(da_ref.time.dt.year).mean()

        # Create dataframe.
        n_ref = len(list(da_ref_mean.values))
        n_sim_adj = len(list(da_sim_adj_mean.values))
        dict_pd = {"year": list(range(1, n_sim_adj + 1)),
                   c.cat_obs: list(da_ref_mean.values) + [np.nan] * (n_sim_adj - n_ref),
                   c.cat_sim_adj: list(da_sim_adj_mean.values),
                   c.cat_sim: list(da_sim_mean.values)}
        df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:
        fig = plot.plot_postprocess(df, varidx, title)
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    if save_csv:
        fu.save_csv(df, p_csv)


def calc_workflow(
    varidx: VarIdx,
    nq: int,
    up_qmf: float,
    time_win: int,
    p_ref: str,
    p_sim: str,
    p_fig: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot allowing to compare reference and simulated data.

    Parameters
    ----------
    varidx : VarIdx
        Variable or index.
    nq : int
        Number of quantiles.
    up_qmf : float
        Upper limit of quantile mapping function.
    time_win : int
        Window.
    p_ref : str
        Path of the NetCDF file containing reference data.
    p_sim : str
        Path of the NetCDF file containing simulation data.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = varidx.name

    # Paths.
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.f_csv + cntx.sep). \
        replace(c.f_ext_png, c.f_ext_csv)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.f_png in cntx.opt_diagnostic_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_diagnostic_format)))
    if not (save_fig or save_csv):
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)
        units = []

    # Prepare data.
    else:

        # Load datasets.
        da_ref = fu.open_netcdf(p_ref)[vi_name]
        da_sim = fu.open_netcdf(p_sim)[vi_name]

        # Select control point.
        if cntx.opt_ra:
            subset_ctrl_pt = False
            if c.dim_rlon in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.rlon) > 1) or (len(da_ref.rlat) > 1)
            elif c.dim_lon in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.lon) > 1) or (len(da_ref.lat) > 1)
            elif c.dim_longitude in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.longitude) > 1) or (len(da_ref.latitude) > 1)
            if subset_ctrl_pt:
                if cntx.p_bounds == "":
                    da_ref = utils.subset_ctrl_pt(da_ref)
                    da_sim = utils.subset_ctrl_pt(da_sim)
                else:
                    da_ref = utils.squeeze_lon_lat(da_ref)
                    da_sim = utils.squeeze_lon_lat(da_sim)

        # Convert date format if the need is.
        if da_ref.time.dtype == c.dtype_obj:
            da_ref[c.dim_time] = utils.reset_calendar(da_ref)

        # Units.
        units = [da_ref.units, da_sim.units]

        # Create dataframe.
        n_ref = len(list(da_ref.values))
        n_sim = len(list(da_sim.values))
        dict_pd = {"year": list(range(1, n_sim + 1)),
                   c.cat_obs: list(da_ref.values) + [np.nan] * (n_sim - n_ref),
                   c.cat_sim: list(da_sim.values)}
        df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:
        title = os.path.basename(p_fig).replace(c.f_ext_png, "") +\
            "_nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
        fig = plot.plot_workflow(df, varidx, units, title)
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    if save_csv:
        fu.save_csv(df, p_csv)
