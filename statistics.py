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
import logging
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
from typing import Union, List, Tuple, Optional

# xclim libraries.
from xclim import ensembles

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


def calc_stats(
    view: View,
    stn: str,
    varidx: VarIdx,
    rcp: RCP,
    sim: Sim,
    delta: bool,
    stats: Stats,
    squeeze_coords: bool,
    clip: bool,
    cat_scen: Optional[str] = ""
) -> Union[dict, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate statistics using xclim libraries.

    The function returns a dictrionary of Dataset instances, each of which is associated with a statistic. The
    statistics returned by the function are the following: mean, std (standard deviation), max, min, cXX (a centile,
    where XX is an integer between 01 and 99).

    Parameters
    ----------
    view: View,
        View = {c.view_ts, c.view_ts_bias, c.view_tbl, c.view_map}
    stn: str
        Station.
    varidx: VarIdx
        Climate variable or index.
    rcp: RCP
        Emission scenario.
    sim: Sim
        Simulation. All simulations are considered if sim.code == c.simxx.
    delta: bool
        If True, calculate climate anomalies.
    stats: Stats
        Statistics.
    squeeze_coords: bool
        If True, squeeze coordinates.l
    clip: bool
        If True, clip according to 'cntx.p_bounds'.
    cat_scen: Optional[str]
        Scenario category = {c.cat_obs, c.cat_raw, c.cat_regrid, c.cat_qqmap}

    Returns
    -------
    Union[dict, None]
        Dictionary of Dataset instances (of statistics).
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
            p_sim_l = [cntx.d_scen(c.cat_obs, vi_code_grp) + vi_name_grp + "_" + stn + c.f_ext_nc]
        else:
            p_sim_l = [cntx.d_idx(vi_code_grp) + vi_name_grp + "_ref" + c.f_ext_nc]
    else:
        if varidx.is_var:
            d = cntx.d_scen(c.cat_qqmap, vi_code_grp)
        else:
            d = cntx.d_idx(vi_code_grp)
        p_sim_l = []
        for p in glob.glob(d + "*" + c.f_ext_nc):
            if ((rcp.code == c.rcpxx) or (rcp.code in p)) and ((sim.code == c.simxx) or (sim.code in p)):
                p_sim_l.append(p)

    # Adjust paths if doing the analysis for bias adjustment time series.
    if ((view.code == c.view_ts_bias) and (rcp.code != c.ref) and varidx.is_var) or (cat_scen == c.cat_regrid):
        for i_sim in range(len(p_sim_l)):
            p_sim_l[i_sim] = p_sim_l[i_sim].replace(cntx.sep + c.cat_qqmap, cntx.sep + c.cat_regrid).\
                replace(c.f_ext_nc, "_4" + c.cat_qqmap + c.f_ext_nc)

    # Exit if there is no file corresponding to the criteria.
    if (len(p_sim_l) == 0) or ((len(p_sim_l) > 0) and not(os.path.isdir(os.path.dirname(p_sim_l[0])))):
        return None

    # Create ensemble.
    xclim_logger_level = utils.get_logger_level("root")
    utils.set_logger_level("root", logging.CRITICAL)
    ds_ens = ensembles.create_ensemble(p_sim_l).load()
    ds_ens.close()
    utils.set_logger_level("root", xclim_logger_level)

    # Skip simulation if there is no data.
    if len(ds_ens[c.dim_time]) == 0:
        return None

    # Convert units.
    if c.attrs_units in ds_ens[vi_name].attrs:
        if (vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and (ds_ens[vi_name].attrs[c.attrs_units] == c.unit_K):
            ds_ens = ds_ens - c.d_KC
            ds_ens[vi_name].attrs[c.attrs_units] = c.unit_C
        elif varidx.is_summable:
            ds_ens = ds_ens * c.spd
            ds_ens[vi_name].attrs[c.attrs_units] = c.unit_mm
        elif vi_name in [c.v_uas, c.v_vas, c.v_sfcwindmax]:
            ds_ens = ds_ens * c.km_h_per_m_s
            ds_ens[vi_name].attrs[c.attrs_units] = c.unit_km_h
        units = ds_ens[vi_name].attrs[c.attrs_units]
    else:
        units = 1

    # Aggregate to the annual frequency.
    if varidx.is_summable:
        ds_ens = ds_ens.resample(time=c.freq_YS).sum()
    else:
        ds_ens = ds_ens.resample(time=c.freq_YS).mean()

    # Calculate statistics by year if there are multiple years.
    calc_by_year = (len(p_sim_l) > 1)

    # Loop through years.
    # The analysis is broken down into years to accomodate large datasets.
    ds_stats = []
    year_1 = int(ds_ens.time[0].dt.year)
    year_n = int(ds_ens.time[len(ds_ens.time) - 1].dt.year)
    year_l = list(range(year_1, year_n + 1))
    for y in year_l:

        # Select the current year if there are multiple simulations in the ensemble.
        if calc_by_year:
            ds_ens_y = utils.sel_period(ds_ens, [y, y])
        else:
            ds_ens_y = ds_ens

        # Calculate statistics (mean, std, max, min).
        ds_stats_basic = ensembles.ensemble_mean_std_max_min(ds_ens_y)

        # Calculate percentiles.
        ds_stats_cent = ensembles.ensemble_percentiles(ds_ens_y, values=stats.centile_l, split=False)

        # Loop through statistics.
        ds_stats_y_l = []
        for stat in stats.items:

            # Select the current statistic.
            if stat.code != c.stat_centile:
                ds_stats_y = ds_stats_basic
            else:
                ds_stats_y = ds_stats_cent.sel(percentiles=stat.centile)
                if c.dim_percentiles in ds_stats_y.dims:
                    ds_stats_y = ds_stats_y.squeeze(dim=c.dim_percentiles)

            # Rename the variable of interest and drop the other variables.
            if stat.code in [c.stat_mean, c.stat_std, c.stat_max, c.stat_min]:
                vi_name_stat = vi_name + "_" + stat.code
                ds_stats_y = ds_stats_y.rename({vi_name_stat: vi_name})
                ds_stats_y = ds_stats_y[[vi_name]]

            # There are no anomalies to calculate for the reference dataset.
            if (rcp.code == c.ref) and delta:
                ds_stats_y[vi_name] = xr.zeros_like(ds_stats_y[vi_name])

            # Subset by location.
            if cntx.opt_ra:

                # Statistics are calculated at a point.
                if cntx.p_bounds == "":
                    ds_stats_y = utils.subset_ctrl_pt(ds_stats_y)

                else:

                    # Statistics are calculated for an area.
                    if clip:
                        ds_stats_y = utils.subset_shape(ds_stats_y)

                    # Calculate the mean value for each year (i.e., all coordinates combined).
                    if squeeze_coords:
                        ds_stats_y = ds_stats_y.mean(dim=utils.coord_names(ds_stats_y))

            # Record this result.
            ds_stats_y_l.append(ds_stats_y)

        # Merge with previous years.
        if y == year_l[0]:
            ds_stats = ds_stats_y_l
            if not calc_by_year:
                break
        else:
            for i in range(len(ds_stats)):
                ds_stats[i] = xr.Dataset.merge(ds_stats[i], ds_stats_y_l[i])

    # Calculate anomalies.
    if delta and (rcp.code != c.ref):

        # Bias time series: adjusted simulation - non-adjusted simulation.
        if (view.code == c.view_ts_bias) and varidx.is_var:
            ds_stats_ref = calc_stats(view, stn, varidx, rcp, sim, False, stats, True, clip, c.cat_regrid)[c.stat_mean]
        else:
            ds_stats_ref = calc_stats(view, stn, varidx, RCP(c.ref), sim, False, Stats([c.stat_mean]), True, clip,
                                      c.cat_qqmap)[c.stat_mean]

        # Adjust values.
        for i in range(len(ds_stats)):
            if (view.code == c.view_ts_bias) and varidx.is_var:
                val_ref = ds_stats_ref[i][vi_name]
            else:
                val_ref = float(ds_stats_ref[vi_name].mean().values)
            ds_stats[i][vi_name] = ds_stats[i][vi_name] - val_ref
            ds_stats[i][vi_name][c.attrs_units] = units

    # Convert the array of Datasets into a dictionary.
    stat_str_l = []
    for stat in stats.items:
        if stat.code != c.stat_centile:
            stat_code = stat.code
        else:
            stat_code = stat.centile_as_str
        stat_str_l.append(stat_code)
    ds_stats_dict = dict(zip(stat_str_l, ds_stats))

    return ds_stats_dict


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
            p_csv = cntx.d_stat(cat, vi_code_grp) + vi_name + c.f_ext_csv
            if os.path.exists(p_csv) and (not cntx.opt_force_overwrite):
                continue

            # Containers.
            stn_l, rcp_l, hor_l, stat_l, centile_l, val_l = [], [], [], [], [], []

            # Loop through emission scenarios.
            for rcp in rcps.items:

                # Select years.
                if rcp.code == c.ref:
                    hors = Hors([cntx.per_ref])
                    if cat == c.cat_scen:
                        d = os.path.dirname(cntx.p_obs(stn, vi_code_grp))
                    else:
                        d = cntx.d_idx(vi_code_grp)
                else:
                    hors = Hors(cntx.per_hors)
                    if cat == c.cat_scen:
                        d = cntx.d_scen(c.cat_qqmap, vi_code_grp)
                    else:
                        d = cntx.d_idx(vi_code_grp)

                if not os.path.isdir(d):
                    continue

                idx_desc = vi_code
                if vi_code_grp != vi_code:
                    idx_desc = vi_code_grp + "." + idx_desc
                fu.log("Processing: " + stn + ", " + idx_desc + ", " + rcp.code + "", True)

                # Identify the statistics to be calculated.
                # The order is: mean, min, max, centile_1, centile_2, ..., centile n.
                stats = Stats([c.stat_mean])
                if rcp.code != c.ref:
                    stats.add([c.stat_min, c.stat_max], inplace=True)
                    for centile in cntx.opt_stat_centiles:
                        if centile not in [0, 1]:
                            stats.add(Stat(c.stat_centile, centile), inplace=True)

                # Calculate statistics.
                # The order is the same as the items in 'stats'.
                ds_stats = calc_stats(View(c.view_tbl), stn, varidx, rcp, Sim(c.simxx), False, stats,
                                      (freq == c.freq_YS) and (cat == c.cat_scen), cntx.opt_stat_clip)

                # Loop through statistics.
                for i in range(len(stats.items)):

                    # Extract the current statistic and the associated result.
                    stat = stats.items[i]
                    centile_str = stat.code if stat.code != c.stat_centile else stat.centile_as_str
                    ds_stat = ds_stats[centile_str]

                    # Loop through horizons.
                    for hor in hors.items:

                        # Select period.
                        if cntx.opt_ra:
                            ds_stat_hor = utils.sel_period(ds_stat.squeeze(), [hor.year_1, hor.year_2]).copy(deep=True)
                        else:
                            ds_stat_hor = ds_stat.copy(deep=True)

                        # Extract value.
                        if varidx.is_summable:
                            val = float(ds_stat_hor[vi_name].sum()) / (hor.year_2 - hor.year_1 + 1)
                        else:
                            val = float(ds_stat_hor[vi_name].mean())

                        # Add row.
                        stn_l.append(stn)
                        rcp_l.append(rcp.code)
                        hor_l.append(hor.code)
                        stat_l.append("none" if rcp.code == c.ref else stat.code)
                        centile_l.append(stat.centile)
                        val_l.append(round(val, 6))

            # Save results.
            if len(stn_l) > 0:

                # Build pandas dataframe.
                dict_pd = {"stn": stn_l, ("var" if cat == c.cat_scen else "idx"): [vi_name] * len(stn_l),
                           "rcp": rcp_l, "hor": hor_l, "stat": stat_l, "centile": centile_l, "val": val_l}
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
        fn = varidx.name + "_" + dash_plot.mode_rcp
        # CSV files:
        p_rcp_csv = cntx.d_fig(cat, c.view_ts, vi_code_grp + "_" + c.f_csv) + fn + c.f_ext_csv
        p_sim_csv = p_rcp_csv.replace("_" + dash_plot.mode_rcp, "_" + dash_plot.mode_sim)
        p_rcp_del_csv = p_rcp_csv.replace(c.f_ext_csv, "_delta" + c.f_ext_csv)
        p_sim_del_csv = p_sim_csv.replace(c.f_ext_csv, "_delta" + c.f_ext_csv)
        # PNG files.
        p_rcp_fig = cntx.d_fig(cat, c.view_ts, vi_code_grp) + fn + c.f_ext_png
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
        cntx.stats  = Stats()
        cntx.stats.add(Stat(c.stat_centile, cntx.opt_ts_centiles[0]))
        cntx.stats.add(Stat(c.stat_centile, cntx.opt_ts_centiles[1]))

        # Attempt loading CSV files into dataframes.
        if os.path.exists(p_rcp_csv) and os.path.exists(p_sim_csv) and \
           os.path.exists(p_rcp_del_csv) and os.path.exists(p_sim_del_csv):
            df_rcp = pd.read_csv(p_rcp_csv)
            df_sim = pd.read_csv(p_sim_csv)
            df_rcp_del = pd.read_csv(p_rcp_del_csv)
            df_sim_del = pd.read_csv(p_sim_del_csv)

        # Generate dataframes.
        else:
            df_stn = calc_ts_stn(stn)
            df_rcp, df_sim, df_rcp_del, df_sim_del =\
                dict(df_stn)["rcp"], dict(df_stn)["sim"], dict(df_stn)["rcp_delta"], dict(df_stn)["sim_delta"]

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


def calc_ts_stn(
    stn: str
) -> dict:

    """
    --------------------------------------------------------------------------------------------------------------------
    Prepare dataframes for the generation of time series.

    This function returns a dictionary of instances of pd.DataFrame:
    - "rcp"       : 3 columns per RCP (mean, min or lower centile, max or higher centile).
    - "sim"       : one column per simulation (mean).
    - "rcp_delta" : same as "rcp", but with anomalies instead of absolute values.
    - "sim_delta" : same as "sim", but with anomalies instead of absolute values.

    Parameters
    ----------
    stn: str
        Station.

    Returns
    -------
    dict
        Dictionary of instances of pd.DataFrame.
    --------------------------------------------------------------------------------------------------------------------
    """

    df_rcp, df_sim, df_rcp_delta, df_sim_delta = None, None, None, None

    # Extract relevant information from the context.
    vi_code = cntx.varidx.code
    vi_name = cntx.varidx.name
    vi_code_grp = VI.group(vi_code) if cntx.varidx.is_group else vi_code
    vi_name_grp = VI.group(vi_name) if cntx.varidx.is_group else vi_name

    # Complete horizon (inncluding reference period).
    hor = Hor([min(min(cntx.per_hors)), max(max(cntx.per_hors))])

    # Emission scenarios.
    rcps = RCPs([c.ref] + cntx.rcps.code_l)

    # List required statistics.
    stats = Stats()
    stat_lower = Stat(c.stat_centile, cntx.opt_ts_centiles[0])
    stat_middle = Stat(c.stat_mean)
    stat_upper = Stat(c.stat_centile, cntx.opt_ts_centiles[1])
    stats.add(stat_lower)
    stats.add(stat_middle)
    stats.add(stat_upper)

    for delta in [False, True]:

        # Initialize the structure that will hold the result for the current 'delta' iteration.
        dict_pd = {"year": list(range(hor.year_1, hor.year_2 + 1))}
        df_rcp_x, df_sim_x = pd.DataFrame(dict_pd), pd.DataFrame(dict_pd)
        df_rcp_x[c.ref], df_sim_x[c.ref] = np.nan, np.nan

        # Loop through emission scenarios.
        for rcp in rcps.items:

            # Calculate ensemble statistics.
            if rcp.code != c.ref:

                # Calculate statistics.
                ds_stats = dict(calc_stats(cntx.view, stn, cntx.varidx, rcp, Sim(c.simxx), delta, stats, True,
                                           cntx.opt_stat_clip))

                # Select years.
                ds_stats_lower = ds_stats[stat_lower.centile_as_str]
                ds_stats_lower = utils.sel_period(ds_stats_lower, [hor.year_1, hor.year_2])
                ds_stats_middle = ds_stats[c.stat_mean]
                ds_stats_middle = utils.sel_period(ds_stats_middle, [hor.year_1, hor.year_2])
                ds_stats_upper = ds_stats[stat_upper.centile_as_str]
                ds_stats_upper = utils.sel_period(ds_stats_upper, [hor.year_1, hor.year_2])

                # Record mean, lower and upper annual values (for each year).
                df_rcp_x[rcp.code + "_lower"] = ds_stats_lower[vi_name].values
                df_rcp_x[rcp.code + "_middle"] = ds_stats_middle[vi_name].values
                df_rcp_x[rcp.code + "_upper"] = ds_stats_upper[vi_name].values

            # List files.
            if cntx.varidx.is_var:
                p_ref = cntx.d_stn(vi_code_grp) + vi_name_grp + "_" + stn + c.f_ext_nc
            else:
                p_ref = cntx.d_idx(vi_code_grp) + vi_name_grp + "_ref" + c.f_ext_nc
            if rcp.code == c.ref:
                p_sim_l = [p_ref]
            else:
                if cntx.varidx.is_var:
                    d = cntx.d_scen(c.cat_qqmap, vi_code_grp)
                else:
                    d = cntx.d_idx(vi_code_grp)
                p_sim_l = glob.glob(d + "*_" + ("*" if rcp.code == c.rcpxx else rcp.code) + c.f_ext_nc)

            # Exit if there is no file corresponding to the criteria of if statistics have already been calculated for
            # the current simulation.
            if (len(p_sim_l) == 0) or\
               ((len(p_sim_l) > 0) and not (os.path.isdir(os.path.dirname(p_sim_l[0])))) or\
               (rcp.code == c.rcpxx):
                continue

            # Loop through simulation files.
            for i_sim in range(len(p_sim_l)):

                # Extract the description of the current simulation.
                if "rcp" not in p_sim_l[i_sim]:
                    sim_code = RCP(c.ref).desc
                else:
                    tokens = os.path.basename(p_sim_l[i_sim]).replace(c.f_ext_nc, "").split("_")
                    n_tokens = len(tokens)
                    sim_code = (tokens[n_tokens - 4] + "_" + tokens[n_tokens - 3] + "_" +
                                tokens[n_tokens - 2] + "_" + tokens[n_tokens - 1])

                # Calculate simulation statistics.
                ds_stats_i = dict(calc_stats(cntx.view, stn, cntx.varidx, rcp, Sim(sim_code), delta,
                                             Stats([c.stat_mean]), True, cntx.opt_stat_clip))[c.stat_mean]

                # Select years.
                ds_stats_i = utils.sel_period(ds_stats_i, [hor.year_1, hor.year_2])

                # Record statistics.
                if rcp.code == c.ref:
                    years = utils.extract_date_field(ds_stats_i, "year")
                    vals = list(ds_stats_i[vi_name].values)
                    for i in range(len(vals)):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=SettingWithCopyWarning)
                            df_rcp_x[c.ref][df_rcp_x["year"] == years[i]] = vals[i]
                            df_sim_x[c.ref][df_sim_x["year"] == years[i]] = vals[i]
                elif rcp.code != c.rcpxx:
                    df_sim_x[sim_code] = ds_stats_i[vi_name].values

        # Record results.
        if len(df_sim_x.columns) > 1:

            if not delta:
                df_rcp = df_rcp_x.copy()
                df_sim = df_sim_x.copy()
            else:
                df_rcp_delta = df_rcp_x.copy()
                df_sim_delta = df_sim_x.copy()

    # Convert the array of instances of pd.DataFrame into a dictionary.
    df_dict = dict(zip(["rcp", "sim", "rcp_delta", "sim_delta"], [df_rcp, df_sim, df_rcp_delta, df_sim_delta]))

    return df_dict


def calc_by_freq(
    ds: xr.Dataset,
    var: VarIdx,
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
    var: VarIdx
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
        da_m = ds[var.name].rename({c.dim_rlon: c.dim_longitude, c.dim_rlat: c.dim_latitude})
    elif c.dim_lon in ds.dims:
        da_m = ds[var.name].rename({c.dim_lon: c.dim_longitude, c.dim_lat: c.dim_latitude})
    else:
        da_m = ds[var.name]

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
            if var.is_summable:
                da_mean = da_m.resample(time=time_str).sum().groupby(freq_str).mean()
                da_min  = da_m.resample(time=time_str).sum().groupby(freq_str).min()
                da_max  = da_m.resample(time=time_str).sum().groupby(freq_str).max()
            else:
                da_mean = da_m.resample(time=time_str).mean().groupby(freq_str).mean()
                da_min  = da_m.resample(time=time_str).mean().groupby(freq_str).min()
                da_max  = da_m.resample(time=time_str).mean().groupby(freq_str).max()

            # Create dataset
            da_mean.name = da_min.name = da_max.name = var.name
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

                ds_m[var.name].attrs[c.attrs_units] = ds[var.name].attrs[c.attrs_units]
                ds_l.append(ds_m)

        else:

            # Calculate statistics for each month-year combination.
            # Looping through years is less efficient but required to avoid a problem with incomplete at-a-station
            # datasets.
            arr_all = []
            for m in range(1, 13):
                vals_m = []
                for y in range(per[0], per[1] + 1):
                    if var.is_summable:
                        val_m_y = np.nansum(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    else:
                        val_m_y = np.nanmean(da_m[(da_m["time.year"] == y) & (da_m["time.month"] == m)].values)
                    vals_m.append(val_m_y)
                arr_all.append(vals_m)

            # Create dataset.
            dict_pd = {var.name: arr_all}
            df = pd.DataFrame(dict_pd)
            ds_l = df.to_xarray()
            ds_l.attrs[c.attrs_units] = ds[var.name].attrs[c.attrs_units]

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

    msg = "Calculating maps."
    fu.log(msg, True)

    # Get statistic string.
    def stat_str(
        _stat: Stat
    ):

        if _stat.code in [c.stat_mean, c.stat_min, c.stat_max]:
            _stat_str = _stat.code
        else:
            _stat_str = _stat.centile_as_str

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
        _d_fig = cntx.d_fig(cat, c.view_map, (vi_code_grp + cntx.sep if vi_code_grp != varidx.code else "") +
                            varidx.code + cntx.sep + per_str)
        _stat_str = stat_str(_stat)
        _fn_fig = vi_name + "_" + _rcp.code + "_" + str(_per[0]) + "_" + str(_per[1]) + "_" + _stat_str + c.f_ext_png
        _p_fig = _d_fig + _fn_fig
        if _delta:
            _p_fig = _p_fig.replace(c.f_ext_png, "_delta" + c.f_ext_png)

        # CSV file.
        _d_csv = cntx.d_fig(cat, c.view_map, (vi_code_grp + cntx.sep if vi_code_grp != varidx.code else "") +
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
    if cntx.opt_map_centiles is not None:
        for centile in cntx.opt_map_centiles:
            stats.add(Stat(c.stat_centile, centile))

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
                    df = pd.DataFrame(df, columns=[c.dim_longitude, c.dim_latitude, varidx.code])
                    df = df.sort_values(by=[c.dim_latitude, c.dim_longitude])
                    lat = list(set(df[c.dim_latitude]))
                    lat.sort()
                    lon = list(set(df[c.dim_longitude]))
                    lon.sort()
                    arr = np.reshape(list(df[varidx.code]), (len(lat), len(lon)))
                    da = xr.DataArray(data=arr, dims=[c.dim_latitude, c.dim_longitude],
                                      coords=[(c.dim_latitude, lat), (c.dim_longitude, lon)])
                    da.name = vi_name
                    ds_map = da.to_dataset()

                else:

                    # Calculate map.
                    ds_map = calc_map_rcp(varidx, rcp, per, stat)

                    # Clip to geographic boundaries.
                    if cntx.opt_map_clip and (cntx.p_bounds != ""):
                        ds_map = utils.subset_shape(ds_map)

                # Record current map, statistic and centile, RCP and period.
                arr_ds_maps.append(ds_map)
                arr_items.append([stat.code, stat.centile, rcp.code, per])

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
                cntx.code   = c.platform_script
                cntx.view   = View(c.view_map)
                cntx.lib    = Lib(c.lib_mat)
                cntx.varidx = VarIdx(vi_name)
                cntx.rcp    = rcp
                cntx.hor    = Hor(per)
                cntx.stats  = Stats()
                for s in range(len(cntx.opt_map_centiles)):
                    stats.add(Stat(c.stat_centile, cntx.opt_map_centiles[s]))
                cntx.stat   = stat
                cntx.delta  = Delta(str(j == 1))

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
           (stat.centile_as_str if stat.centile >= 0 else stat.code) + ", " +
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

        # TODO: Warning: this part has never been tested.

        # Get information on stations.
        p_stn = glob.glob(cntx.d_stn() + c.f_ext_csv)[0]
        df = pd.read_csv(p_stn, sep=cntx.f_sep)

        # Collect values for each station and determine overall boundaries.
        fu.log("Collecting emissions scenarios at each station.", True)
        data_lon_l, data_lat_l, data_val_l = [], [], []
        for stn in stns:

            # Get coordinates.
            lon = df[df["station"] == stn][c.dim_lon].values[0]
            lat = df[df["station"] == stn][c.dim_lat].values[0]

            # Calculate statistics.
            ds_stat = calc_stats(c.view_map, stn, varidx, rcp, Sim(c.simxx), False, Stats(stat), False,
                                 cntx.opt_map_clip)[c.stat_mean]
            if ds_stat is None:
                continue

            # Extract data from stations.
            n = ds_stat.dims[c.dim_time]
            for year in range(0, n):

                # Collect data.
                lon = float(lon)
                lat = float(lat)
                val = float(ds_stat[vi_name][0][0][year])
                if math.isnan(val):
                    val = float(0)
                data_lon_l.append(lon)
                data_lat_l.append(lat)
                data_val_l.append(val)

        fu.log("Performing interpolation.", True)

        # Build a Dataset containing the x-y-z locations of data points.
        df_data = pd.Dataframe({vi_name: data_val_l, c.dim_latitude: data_lat_l, c.dim_longitude: data_lon_l})
        df_data = df_data.pivot(index=c.dim_latitude, columns=c.dim_longitude)
        df_data = df_data.droplevel(0, axis=1)
        da_data = xr.DataArray(df_data)
        ds_data = da_data.to_dataset(name=vi_name)

        # Build a list of x-y locations at which interpolation is needed.
        grid_lon_l = np.arange(min(data_lon_l), max(data_lon_l) + cntx.map_resol, cntx.map_resol)
        grid_lat_l = np.arange(min(data_lat_l), max(data_lat_l) + cntx.map_resol, cntx.map_resol)

        # Create a Dataset for the new grid.
        ds_grid = xr.Dataset({c.dim_latitude: ([c.dim_latitude], grid_lat_l),
                              c.dim_longitude: ([c.dim_longitude], grid_lon_l)})

        # Interpolate.
        regridder = xe.Regridder(ds_data, ds_grid, "bilinear")
        da_res = regridder(ds_data[vi_name])

        # Apply and create mask.
        if (cntx.obs_src == c.ens_era5_land) and (vi_name not in [c.v_tas, c.v_tasmin, c.v_tasmax]):
            da_mask = fu.create_mask()
            if da_mask is not None:
                da_res = utils.apply_mask(da_res, da_mask)

        # Create dataset.
        ds_res = da_res.to_dataset(name=vi_name)
        ds_res[vi_name].attrs[c.attrs_units] = da_data.attrs[c.attrs_units]

    # There is no need to interpolate.
    else:

        # Reference period.
        if varidx.is_var:
            p_sim_ref =\
                cntx.d_scen(c.cat_obs, vi_code_grp) + vi_name_grp + "_" + cntx.obs_src + c.f_ext_nc
        else:
            p_sim_ref = cntx.d_idx(vi_code_grp) + vi_name_grp + "_" + c.ref + c.f_ext_nc
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
            if varidx.is_summable:
                ds_res = ds_res * 365
            if varidx.is_var:
                ds_res[vi_name].attrs[c.attrs_units] = units
            else:
                ds_res.attrs[c.attrs_units] = units

        # Future period.
        else:

            # List scenarios or indices for the current RCP.
            if varidx.is_var:
                d = cntx.d_scen(c.cat_qqmap, vi_code_grp)
            else:
                d = cntx.d_idx(vi_code_grp)
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
                    if varidx.is_summable:
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
                    elif stat.code == c.stat_centile:
                        val = np.percentile(vals, stat.centile)

                    # Record value.
                    ds_res[vi_name][i_lat, j_lon] = val

        # Remember units.
        units = ds_res[vi_name].attrs[c.attrs_units] if varidx.is_var else ds_res.attrs[c.attrs_units]

        # Convert units.
        if (vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and\
           (ds_res[vi_name].attrs[c.attrs_units] == c.unit_K):
            ds_res = ds_res - c.d_KC
            ds_res[vi_name].attrs[c.attrs_units] = c.unit_C
        elif varidx.is_summable and (ds_res[vi_name].attrs[c.attrs_units] == c.unit_kg_m2s1):
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
                if cat != c.cat_idx:
                    p_l = list(glob.glob(cntx.d_scen(cat, vi_code_grp) + "*" + c.f_ext_nc))
                else:
                    p_l = list(glob.glob(cntx.d_fig(vi_code_grp) + "*" + c.f_ext_nc))
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
                        d_cat = cntx.d_scen(cat, vi_code) if cat != c.cat_fig else cntx.d_idx(vi_code)
                        n_files_proc_before = len(list(glob.glob(d_cat + "*" + c.f_ext_csv)))

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
                        n_files_proc_after = len(list(glob.glob(d_cat + "*" + c.f_ext_csv)))

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
    if varidx.is_summable and (ds[vi_name].attrs[c.attrs_units] == c.unit_kg_m2s1):
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
    cat_vi = c.cat_scen if varidx.is_var else c.cat_idx
    cat_fig = c.view_cycle_ms if freq == c.freq_MS else c.view_cycle_d
    p_fig = cntx.d_fig(cat_vi, cat_fig, varidx.name) + title + c.f_ext_png
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
        elif varidx.is_summable and (ds[varidx.name].attrs[c.attrs_units] == c.unit_kg_m2s1):
            ds = ds * c.spd
        ds[varidx.name].attrs[c.attrs_units] = units

        # Calculate statistics.
        ds_l = calc_by_freq(ds, varidx, per, freq)

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
    cntx.varidxs = cntx.cluster_vars
    cntx.rcp     = RCP(c.rcpxx)
    cntx.stats   = Stats()
    cntx.stats.add(Stat(c.stat_centile, cntx.opt_cluster_centiles[0]))
    cntx.stats.add(Stat(c.stat_centile, cntx.opt_cluster_centiles[1]))

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
            p_i = cntx.d_fig(c.cat_scen, c.view_ts, var.code + "_" + c.f_csv) + var.code + "_sim" + c.f_ext_csv
            p_csv_ts_l.append(p_i)
        n_cluster_max = len(dash_utils.get_shared_sims(p_csv_ts_l))

        # Loop through all combinations of
        for n_cluster in range(1, n_cluster_max):

            # Paths.
            p_csv = cntx.d_fig(c.cat_scen, c.view_cluster) +\
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
        if varidx.is_summable:
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
    p_ref: str,
    p_sim: str,
    p_fig: str,
    title: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot allowing to compare reference and simulated data.

    Parameters
    ----------
    varidx : VarIdx
        Variable or index.
    p_ref : str
        Path of the NetCDF file containing reference data.
    p_sim : str
        Path of the NetCDF file containing simulation data.
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
        fig = plot.plot_workflow(df, varidx, units, title)
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    if save_csv:
        fu.save_csv(df, p_csv)
