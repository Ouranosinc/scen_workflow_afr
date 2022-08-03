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
import multiprocessing
import numpy as np
import os
import pandas as pd
# Do not delete the line below: 'rioxarray' must be added even if it's not explicitly used in order to have a 'rio'
# variable in DataArrays.
import rioxarray
import skill_metrics as sm
import sys
import warnings
import xarray as xr
import xesmf as xe
from typing import Union, List, Tuple, Optional

# xclim libraries.
from xclim import ensembles

# Workflow libraries.
import wf_file_utils as fu
import wf_plot
import wf_utils
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import dash_plot, dash_stats as dash_stats, dash_utils, cl_varidx as vi
from dashboard.cl_delta import Delta
from dashboard.cl_hor import Hor, Hors
from dashboard.cl_lib import Lib
from dashboard.cl_rcp import RCP, RCPs
from dashboard.cl_sim import Sim
from dashboard.cl_stat import Stat, Stats
from dashboard.cl_varidx import VarIdx
from dashboard.cl_view import View


def calc_stats(
    ds_l: List[xr.Dataset],
    view: View,
    stn: str,
    varidx: VarIdx,
    rcp: RCP,
    sim: Sim,
    hor: Union[None, Hor],
    delta: bool,
    stats: Stats,
    squeeze_coords: bool,
    clip: bool,
    cat_scen: Optional[str] = ""
) -> Union[None, dict]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate statistics using xclim libraries.

    The function returns a dictrionary of Datasets, each of which is associated with a statistic. The
    statistics returned by the function are the following: mean, std (standard deviation), max, min, cXX (a centile,
    where XX is an integer between 01 and 99).

    Parameters
    ----------
    ds_l: List[xr.Dataset]
        List of Datasets
    view: View,
        View = {c.VIEW_TS, c.VIEW_TS_BIAS, c.VIEW_TBL, c.VIEW_MAP}
    stn: str
        Station.
    varidx: VarIdx
        Climate variable or index.
    rcp: RCP
        Emission scenario. All RCPs are considered if rcp.code == c.RCPXX.
    sim: Sim
        Simulation. All simulations are considered if sim.code == c.SIMXX.
    hor: Union[None, Hor]
        Horizon. No constraint is applied if None is specified.
    delta: bool
        If True, calculate climate anomalies.
    stats: Stats
        Statistics.
    squeeze_coords: bool
        If True, squeeze coordinates.l
    clip: bool
        If True, clip according to 'cntx.p_bounds'.
    cat_scen: Optional[str]
        Scenario category = {c.CAT_OBS, c.CAT_RAW, c.CAT_REGRID, c.CAT_QQMAP}

    Returns
    -------
    Union[None, dict]
        Dictionary of Datasets (containing statistics).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract name and group.
    vi_name = str(varidx.name)

    # Collect Datasets.
    if len(ds_l) == 0:

        # List paths to NetCDF files.
        p_sim_l = list(list_netcdf(stn, varidx, "path", rcp, sim))

        # Adjust paths if doing the analysis for bias adjustment time series.
        if (rcp.code != c.REF) and (cat_scen == c.CAT_REGRID):
            for i_sim in range(len(p_sim_l)):
                p_sim_l[i_sim] = p_sim_l[i_sim].replace(cntx.sep + c.CAT_QQMAP, cntx.sep + c.CAT_REGRID).\
                    replace(c.F_EXT_NC, "_4" + c.CAT_QQMAP + c.F_EXT_NC)

        # Exit if there is no file corresponding to the criteria.
        if (len(p_sim_l) == 0) or ((len(p_sim_l) > 0) and not(os.path.isdir(os.path.dirname(p_sim_l[0])))):
            return None

        # Create array of Datasets.
        for i in range(len(p_sim_l)):

            # Open dataset.
            ds_i = fu.open_netcdf(p_sim_l[i])
            ds_i = wf_utils.standardize_netcdf(ds_i, vi_name=vi_name)

            # Add to list of Datasets.
            ds_l.append(ds_i)

    # Subset, resample and adjust units.
    units = ""
    for i in range(len(ds_l)):

        # Select dataset.
        ds_i = ds_l[i]
        units = str(wf_utils.units(ds_i, vi_name))

        # Subset years.
        if hor is not None:
            ds_i = wf_utils.sel_period(ds_i, [hor.year_1, hor.year_2]).copy(deep=True)

        # Subset by location (at a point or on a surface).
        if cntx.opt_ra:
            if cntx.p_bounds == "":
                ds_i = wf_utils.subset_ctrl_pt(ds_i)
            elif clip:
                ds_i = wf_utils.subset_shape(ds_i)

        # Resample to the annual frequency.
        if varidx.is_summable:
            ds_i = ds_i.resample(time=c.FREQ_YS).sum()
        else:
            ds_i = ds_i.resample(time=c.FREQ_YS).mean()

        # Adjust units (1st call set units; 2nd call converts units).
        ds_i = wf_utils.set_units(ds_i, vi_name, units)
        ds_i = wf_utils.set_units(ds_i, vi_name)

        ds_l[i] = ds_i

    # Create ensemble.
    xclim_logger_level = wf_utils.get_logger_level("root")
    wf_utils.set_logger_level("root", logging.CRITICAL)
    ds_ens = ensembles.create_ensemble(ds_l).load()
    ds_ens.close()
    wf_utils.set_logger_level("root", xclim_logger_level)

    # Skip simulation if there is no data.
    if len(ds_ens[c.DIM_TIME]) == 0:
        return None

    # Calculate statistics by year if there are multiple years.
    calc_by_year = (len(ds_l) > 1)

    # Loop through years.
    # The analysis is broken down into years to accomodate large datasets.
    ds_stats_l = []
    year_1 = int(ds_ens.time[0].dt.year)
    year_n = int(ds_ens.time[len(ds_ens.time) - 1].dt.year)
    year_l = list(range(year_1, year_n + 1))
    for y in year_l:

        # Select the current year if there are multiple simulations in the ensemble.
        if calc_by_year:
            ds_ens_y = wf_utils.sel_period(ds_ens, [y, y])
        else:
            ds_ens_y = ds_ens

        # Calculate statistics (mean, std, max, min).
        ds_stats_basic = None
        if (c.STAT_MEAN in stats.code_l) or (c.STAT_STD in stats.code_l) or\
           (c.STAT_MIN in stats.code_l) or (c.STAT_MAX in stats.code_l):
            ds_stats_basic = ensembles.ensemble_mean_std_max_min(ds_ens_y)

        # Calculate centiles.
        ds_stats_centile = None
        if c.STAT_CENTILE in stats.code_l:
            ds_stats_centile = ensembles.ensemble_percentiles(ds_ens_y, values=stats.centile_l, split=False)

        # Loop through statistics.
        ds_stats_y_l = []
        for stat in stats.items:

            # Select the current statistic.
            if stat.code != c.STAT_CENTILE:
                ds_stats_y = ds_stats_basic
            else:
                ds_stats_y = ds_stats_centile.sel(percentiles=stat.centile)
                if c.DIM_PERCENTILES in ds_stats_y.dims:
                    ds_stats_y = ds_stats_y.squeeze(dim=c.DIM_PERCENTILES)

            # Rename the variable of interest and drop the other variables.
            if stat.code in [c.STAT_MEAN, c.STAT_STD, c.STAT_MAX, c.STAT_MIN]:
                vi_name_stat = vi_name + "_" + stat.code
                ds_stats_y = ds_stats_y.rename({vi_name_stat: vi_name})
                ds_stats_y = ds_stats_y[[vi_name]]

            # There are no anomalies to calculate for the reference dataset.
            if (rcp.code == c.REF) and delta:
                ds_stats_y[vi_name] = xr.zeros_like(ds_stats_y[vi_name])

            # Calculate the mean value for each year (i.e., all coordinates combined).
            if cntx.opt_ra and squeeze_coords:
                ds_stats_y = ds_stats_y.mean(dim=wf_utils.coord_names(ds_stats_y))

            # Put units back in.
            ds_stats_y = wf_utils.set_units(ds_stats_y, vi_name, units)

            # Record this result.
            ds_stats_y_l.append(ds_stats_y)

        # Merge with previous years.
        if y == year_l[0]:
            ds_stats_l = ds_stats_y_l
            if not calc_by_year:
                break
        else:
            for i in range(len(ds_stats_l)):
                ds_stats_l[i] = xr.Dataset.merge(ds_stats_l[i], ds_stats_y_l[i])
                ds_stats_l[i] = wf_utils.set_units(ds_stats_l[i], vi_name, units)

    if delta:

        # All anomalies are zero for the reference set.
        if rcp.is_ref:
            for s in range(len(ds_stats_l)):
                ds_stats_l[s][vi_name] = xr.zeros_like(ds_stats_l[s][vi_name])

        # Calculate anomalies.
        else:

            # Bias time series: adjusted simulation - non-adjusted simulation.
            if (view.code == c.VIEW_TS_BIAS) and varidx.is_var:
                ds_stats_ref = calc_stats(ds_l=[], view=view, stn=stn, varidx=varidx, rcp=rcp, sim=sim,
                                          hor=None, delta=False, stats=stats, squeeze_coords=True, clip=clip,
                                          cat_scen=c.CAT_REGRID)[c.STAT_MEAN]
            else:
                ds_stats_ref = calc_stats(ds_l=[], view=view, stn=stn, varidx=varidx, rcp=RCP(c.REF), sim=Sim(c.REF),
                                          hor=None, delta=False, stats=Stats([c.STAT_MEAN]), squeeze_coords=True,
                                          clip=clip, cat_scen=c.CAT_QQMAP)[c.STAT_MEAN]
            units = ds_stats_ref[vi_name].attrs[c.ATTRS_UNITS]

            # Adjust values.
            if (view.code == c.VIEW_TS_BIAS) and varidx.is_var:
                val_ref = ds_stats_ref[vi_name]
            else:
                val_ref = float(ds_stats_ref[vi_name].mean().values)
            for s in range(len(ds_stats_l)):
                ds_stats_l[s][vi_name] = ds_stats_l[s][vi_name] - val_ref
                ds_stats_l[s][vi_name][c.ATTRS_UNITS] = units

    # Convert the arrays of Datasets into dictionaries.
    stat_code_l = []
    for stat in stats.items:
        if stat.code != c.STAT_CENTILE:
            stat_code = stat.code
        else:
            stat_code = stat.centile_as_str
        stat_code_l.append(stat_code)

    ds_stats_dict = dict(zip(stat_code_l, ds_stats_l))

    return ds_stats_dict


def calc_stats_ref_sims(
    stn: str,
    view: View,
    varidx: VarIdx,
    hor: Optional[Union[None, Hor]] = None,
    delta: Optional[bool] = False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate statistics for the reference data and all simulations.

    Parameters
    ----------
    stn: str
        Station name.
    view: View
        View = {c.VIEW_TBL, c.VIEW_TS}.
    varidx: VarIdx
        Variables or index.
    hor: Optional[Union[None, Hor]]
        Horizon
    delta: Optional[bool]
        If True, calculate anomalies.

    Returns
    -------
    List[List[str, dict]]
        Array of [simulation code, dictionary of [statistics]].
    --------------------------------------------------------------------------------------------------------------------
    """

    # Data frequency.
    freq = c.FREQ_D if varidx.is_var else c.FREQ_YS

    # List simulations associated with NetCDF files.
    sim_code_l = list(list_netcdf(stn, varidx, "sim_code"))

    # View.
    if view.code == c.VIEW_TBL:
        squeeze_coords = (freq == c.FREQ_YS) and varidx.is_var
    elif view.code in [c.VIEW_TS, c.VIEW_TS_BIAS]:
        squeeze_coords = True
    else:
        squeeze_coords = False

    # Statistics.
    stats = Stats([c.STAT_MEAN])
    if view.code == c.VIEW_MAP:
        if cntx.opt_map_centiles is not None:
            for centile in cntx.opt_map_centiles:
                stats.add(Stat(c.STAT_CENTILE, centile))

    # Calculate statistics for the reference data and all simulations.
    ds_stats_l = []
    for sim_code_i in sim_code_l:

        fu.log("Processing: " + cntx.obs_src + ", " + varidx.code + ", " + sim_code_i, True)

        rcp_code = Sim(sim_code_i).rcp.code

        stats_dict =\
            dict(calc_stats(ds_l=[], view=view, stn=stn, varidx=varidx, rcp=RCP(rcp_code), sim=Sim(sim_code_i),
                            hor=hor, delta=delta, stats=stats, squeeze_coords=squeeze_coords,
                            clip=cntx.opt_tbl_clip))

        ds_stats_l.append([sim_code_i, stats_dict])

    return ds_stats_l


def list_netcdf(
    stn: str,
    varidx: VarIdx,
    out_format: str,
    rcp: Optional[RCP] = RCP(""),
    sim: Optional[Sim] = Sim(""),
    include_ref: Optional[bool] = True
) -> List[str]:

    """
    --------------------------------------------------------------------------------------------------------------------
    List paths to reference data and simulations.

    Parameters
    ----------
    stn: str
        Station name.
    varidx: VarIdx
        Variables or index.
    out_format: str
        Output format = {"path", sim_code"}
    rcp: Optional[RCP]
        Emission scenario.
        If a value is not provided, the reference data and all simulations are considered.
        If the value is c.RCPXX, all simulations are considered.
    sim: Optional[Sim]
        Simulation.
        If a value is not provided, the reference data and all simulations are considered.
        If the value is c.SIMXX, all simulations are considered.
    include_ref: Optional[bool]
        If True, include the reference data.

    Returns
    -------
    List[str]
        List of paths to reference data and simulations.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable or index code and group.
    vi_code = varidx.code
    vi_name = varidx.name
    vi_code_grp = vi.group(vi_code) if varidx.is_group else vi_code
    vi_name_grp = vi.group(vi_name) if varidx.is_group else vi_name

    # Collect paths to simulation files.
    ref_in_rcp_sim = (rcp.code in ["", c.REF]) and (sim.code in ["", c.REF])
    if varidx.is_var:
        p_ref = cntx.d_scen(c.CAT_OBS, vi_code_grp) + vi_name_grp + "_" + stn + c.F_EXT_NC
        include_ref = include_ref and os.path.exists(p_ref) and ref_in_rcp_sim
        d = cntx.d_scen(c.CAT_QQMAP, vi_code_grp)
    else:
        p_ref = cntx.d_idx(vi_code_grp) + vi_name_grp + "_ref" + c.F_EXT_NC
        include_ref = include_ref and os.path.exists(p_ref) and ref_in_rcp_sim
        d = cntx.d_idx(vi_code_grp)
    p_l = []
    for p in glob.glob(d + "*" + c.F_EXT_NC):
        if ((rcp.code in ["", c.RCPXX]) or (rcp.code in p)) and ((sim.code in ["", c.SIMXX]) or (sim.code in p)):
            p_l.append(p)

    # Return paths.
    if out_format == "path":

        # Add reference data.
        if include_ref:
            p_l = [p_ref] + p_l

        # Return paths.
        return p_l

    # Return simulations codes.
    else:

        # Extract simulation codes.
        sim_code_l = []
        for p in p_l:
            sim_code_i = os.path.basename(p).replace(varidx.name + "_", "").replace(c.F_EXT_NC, "")
            if sim_code_i != c.REF:
                sim_code_l.append(sim_code_i)
        sim_code_l.sort()

        # Add reference data.
        if include_ref:
            sim_code_l = [c.REF] + sim_code_l

        return sim_code_l


def calc_taylor(
    vi_code_l: List[str],
    i_vi_proc: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Taylor diagram.

    Parameters
    ----------
    vi_code_l: List[str],
        Variables or index codes.
    i_vi_proc: int
        Rank of variable or index to process.
    --------------------------------------------------------------------------------------------------------------------
    """

    varidx = VarIdx(vi_code_l[i_vi_proc])

    # Resampling frequency.
    freq = c.FREQ_MS if varidx.is_var else c.FREQ_YS

    cat = c.CAT_SCEN if varidx.is_var else c.CAT_IDX
    if (cat == c.CAT_SCEN) or (not varidx.is_group):
        vi_code_l = [varidx.code]
        vi_name_l = [varidx.name]
    else:
        vi_code_l = vi.explode_idx_l([varidx.code])
        vi_name_l = vi.explode_idx_l([varidx.name])

    # Adjust the dataset by filling missing values (interpolating), removing February 29th, selecting reference period,
    # clipping and squeezing.
    def adjust_ds(
        _ds: xr.Dataset,
        _vi_name: str
    ) -> xr.Dataset:

        # Interpolating along time dimnension to fill missing data.
        if freq == c.FREQ_D:
            _ds_days = wf_utils.remove_feb29(_ds).resample(time=c.FREQ_YS).count(dim=c.DIM_TIME, keep_attrs=True)
            if int(_ds_days[_vi_name].min()) != int(_ds_days[_vi_name].max()):
                if VarIdx(_vi_name).is_summable:
                    _ds = _ds.resample(time=freq).sum(dim=c.DIM_TIME, keep_attrs=True)
                else:
                    _ds = _ds.resample(time=freq).mean(dim=c.DIM_TIME, keep_attrs=True)
                _ds = _ds.interpolate_na(dim=c.DIM_TIME)

        # Subset (in time and space).
        _ds = wf_utils.remove_feb29(_ds)
        _ds = wf_utils.sel_period(_ds, cntx.per_ref)
        if cntx.opt_ra and cntx.opt_taylor_clip and cntx.p_bounds != "":
            _ds = wf_utils.subset_shape(_ds)
        _ds = wf_utils.squeeze_lon_lat(_ds)

        # Daily frequency.
        # TODO: Get rid of the following warning : 'numpy.datetime64' object has no attribute 'year'
        if freq == c.FREQ_D:
            _ds = wf_utils.convert_to_365_calendar(_ds)
            _ds[c.DIM_TIME] = wf_utils.reset_calendar(_ds)

        # Yearly or monthly frequency.
        else:
            if VarIdx(_vi_name).is_summable:
                _ds = _ds.resample(time=freq).sum(dim=c.DIM_TIME, keep_attrs=True)
            else:
                _ds = _ds.resample(time=freq).mean(dim=c.DIM_TIME, keep_attrs=True)

        # Adjust units.
        if ((_vi_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]) and
            (((c.ATTRS_UNITS in _ds[_vi_name].attrs) and (c.UNIT_C not in _ds[_vi_name].attrs[c.ATTRS_UNITS])) or
             ((c.ATTRS_UNITS not in _ds[_vi_name].attrs) and (float(_ds[_vi_name].max()) > 100)))):
            _ds[_vi_name] = _ds[_vi_name] - c.d_KC
            _ds[_vi_name].attrs[c.ATTRS_UNITS] = c.UNIT_C
        elif (varidx.is_summable and
              (((c.ATTRS_UNITS in _ds[_vi_name].attrs) and (c.UNIT_mm not in _ds[_vi_name].attrs[c.ATTRS_UNITS])) or
               ((c.ATTRS_UNITS not in _ds[_vi_name].attrs) and (float(_ds[_vi_name].max()) < 1)))):
            _ds[_vi_name] = _ds[_vi_name] * c.SPD
            _ds[_vi_name].attrs[c.ATTRS_UNITS] = c.UNIT_mm

        return _ds

    # Loop through stations.
    stns = (cntx.stns if not cntx.opt_ra else [cntx.obs_src])
    for stn in stns:

        # Loop through variables or indices.
        # There should be a single item in the list, unless 'vi_code' is a group of indices.
        for i_vi in range(len(vi_name_l)):
            vi_code = vi_code_l[i_vi]
            vi_name = vi_name_l[i_vi]
            vi_code_grp = vi.group(vi_code) if varidx.is_group else vi_code

            # Loop through categories of simulation data.
            cat_l = [c.CAT_REGRID, c.CAT_QQMAP] if varidx.is_var else [c.CAT_IDX]
            for cat in cat_l:

                # Paths.
                p_fig = cntx.d_fig(c.VIEW_TAYLOR, vi_code_grp) + vi_name + ("_" + cat if varidx.is_var else "") +\
                    c.F_EXT_PNG
                p_csv = p_fig.replace(cntx.sep + vi_code_grp + cntx.sep,
                                      cntx.sep + vi_code_grp + "_" + c.F_CSV + cntx.sep).\
                    replace(c.F_EXT_PNG, c.F_EXT_CSV)

                # Determine if the analysis is required.
                save_fig = (cntx.opt_force_overwrite or
                            ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_taylor_format)))
                save_csv = (cntx.opt_force_overwrite or
                            ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_taylor_format)))
                if not (save_fig or save_csv):
                    return

                # Arrays that will hold statistics.
                sim_code_l, sdev_l, crmsd_l, ccoef_l = [], [], [], []

                # Load data.
                if os.path.exists(p_csv):
                    df = pd.read_csv(p_csv)

                # Prepare data.
                else:

                    # Status message.
                    vi_code_display = vi_code_grp + "." + vi_code if vi_code_grp != vi_code else vi_code
                    msg = "Processing: " + stn + ", " + vi_code_display + ", " + cat
                    fu.log(msg)

                    # List simulations associated with NetCDF files.
                    sim_code_l = list(list_netcdf(stn, varidx, "sim_code"))
                    if c.REF in sim_code_l:
                        sim_code_l.remove(c.REF)
                    sim_code_l.sort()

                    # Load reference data.
                    if varidx.is_var:
                        p_ref = cntx.d_scen(c.CAT_OBS, vi_name) + vi_name + "_" + cntx.obs_src + c.F_EXT_NC
                    else:
                        p_ref = cntx.d_idx(vi_code_grp) + vi_name + "_" + c.REF + c.F_EXT_NC
                    ref_l = []
                    if os.path.exists(p_ref):

                        # Load NetCDF (reference data).
                        ds_ref = fu.open_netcdf(p_ref)

                        # Select the days associated with the reference data, discard February 29th, and squeeze.
                        ds_ref = adjust_ds(ds_ref, vi_name)
                        ref_l = ds_ref[vi_name].values
                        if len([i for i in list(range(len(ref_l))) if not np.isnan(ref_l[i])]) == 0:
                            continue

                    # Loop through simulations.
                    sim_code_sel_l = []
                    for sim_code in sim_code_l:

                        # Load NetCDF (simulation data).
                        if varidx.is_var:
                            p_sim = cntx.d_scen(cat, vi_name) + vi_name + "_" + sim_code + c.F_EXT_NC
                            if cat == c.CAT_REGRID:
                                p_sim = p_sim.replace(c.F_EXT_NC, "_ref_4qqmap" + c.F_EXT_NC)
                        else:
                            p_sim = cntx.d_idx(vi_code_grp) + vi_name + "_" + sim_code + c.F_EXT_NC
                        ds_sim = fu.open_netcdf(p_sim)

                        # Select the days associated with the reference data, discard February 29th, and squeeze.
                        ds_sim = adjust_ds(ds_sim, vi_name)
                        sim_l = ds_sim[vi_name].values
                        if (len([i for i in list(range(len(sim_l))) if not np.isnan(sim_l[i])]) == 0) or\
                           (len(ref_l) != len(sim_l)):
                            continue

                        # Select the values that are not equal to np.nan (in the reference and simulation datasets).
                        sel = ref_l * sim_l
                        sel = [i for i in list(range(len(sel))) if not(np.isnan(sel[i]))]

                        # Calculate and store statistics.
                        # The first array element (e.g. taylor_stats[0]) corresponds to the reference series while
                        # the second and subsequent elements (e.g. taylor_stats[1:]) are those for the predicted
                        # series.
                        taylor_stats = sm.taylor_statistics([sim_l[i] for i in sel], [ref_l[i] for i in sel], "data")
                        for j in range(2):
                            if (len(sdev_l) == 0) or (j == 1):
                                sim_code_sel_l.append(sim_code)
                                sdev_l.append(taylor_stats["sdev"][j])
                                crmsd_l.append(taylor_stats["crmsd"][j])
                                ccoef_l.append(taylor_stats["ccoef"][j])

                    # Adjust precision.
                    sim_code_l = ["Référence"] + sim_code_sel_l
                    sdev_l = list(dash_plot.adjust_precision(sdev_l, n_dec_max=2, output_type="float"))
                    crmsd_l = list(dash_plot.adjust_precision(crmsd_l, n_dec_max=2, output_type="float"))
                    ccoef_l = list(dash_plot.adjust_precision(ccoef_l, n_dec_max=2, output_type="float"))

                    # Convert to a DataFrame.
                    dict_pd = {"sim_code": sim_code_l, "sdev": sdev_l, "crmsd": crmsd_l, "ccoef": ccoef_l}
                    df = pd.DataFrame(dict_pd)

                # Save CSV file.
                if save_csv and (p_csv != ""):
                    fu.save_csv(df, p_csv)

                # Generate figure.
                if save_fig:
                    fig = dash_plot.gen_taylor_plot(df)
                    fu.save_plot(fig, p_fig)


def calc_tbl(
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

    cat = c.CAT_SCEN if varidx.is_var else c.CAT_IDX
    if (cat == c.CAT_SCEN) or (not varidx.is_group):
        vi_code_l = [varidx.code]
        vi_name_l = [varidx.name]
    else:
        vi_code_l = vi.explode_idx_l([varidx.code])
        vi_name_l = vi.explode_idx_l([varidx.name])

    # List emission scenarios.
    rcps = RCPs([c.REF] + cntx.rcps.code_l)
    if rcps.count > 2:
        rcps.add(c.RCPXX)

    # Identify the statistics to be calculated.
    stats_ref, stats_rcp = Stats([c.STAT_MEAN]), Stats([c.STAT_MEAN])
    stats_rcp.add([c.STAT_MIN, c.STAT_MAX], inplace=True)
    for centile in cntx.opt_tbl_centiles:
        if centile not in [0, 100]:
            stats_rcp.add(Stat(c.STAT_CENTILE, centile), inplace=True)

    # Loop through stations.
    stns = (cntx.stns if not cntx.opt_ra else [cntx.obs_src])
    for stn in stns:

        # Loop through variables or indices.
        # There should be a single item in the list, unless 'vi_code' is a group of indices.
        for i_vi in range(len(vi_name_l)):
            vi_code = vi_code_l[i_vi]
            vi_name = vi_name_l[i_vi]
            vi_code_grp = vi.group(vi_code) if varidx.is_group else vi_code

            # Status message.
            vi_code_display = vi_code_grp + "." + vi_code if vi_code_grp != vi_code else vi_code
            msg = "Processing: " + stn + ", " + vi_code_display

            # Skip iteration if the file already exists.
            p_csv = cntx.d_tbl(vi_code_grp) + vi_name + c.F_EXT_CSV
            if os.path.exists(p_csv) and (not cntx.opt_force_overwrite):
                fu.log(msg + "(not required)", True)
                continue

            fu.log(msg, True)

            # Try reading values from time series.
            df_stats = None
            p = cntx.d_fig(c.VIEW_TS, vi_code + "_" + c.F_CSV) + vi_name + "_sim" + c.F_EXT_CSV
            if os.path.exists(p):
                df_stats = pd.read_csv(p)

            # Calculate statistics for the reference data and all simulations.
            ds_stats_l = []
            if df_stats is None:
                ds_stats_l = calc_stats_ref_sims(stn, View(c.VIEW_TBL), varidx)

            # Containers.
            stn_l, rcp_l, hor_l, stat_l, centile_l, val_l = [], [], [], [], [], []

            # Loop through emission scenarios.
            for rcp in rcps.items:

                # Select years.
                if rcp.code == c.REF:
                    hors = Hors([cntx.per_ref])
                else:
                    hors = Hors(cntx.per_hors)

                # Loop through statistics.
                stats = stats_ref if rcp.code == c.REF else stats_rcp
                for stat in stats.items:

                    # Loop through horizons.
                    for hor in hors.items:

                        # Collect values for the simulations included in 'rcp'.
                        val_hor_l = []
                        if df_stats is not None:
                            df_stats_hor =\
                                df_stats[(df_stats["year"] >= hor.year_1) & (df_stats["year"] <= hor.year_2)]
                            for column in df_stats_hor.columns:
                                if (rcp.code in column) or ((rcp.code == c.RCPXX) and ("rcp" in column)):
                                    val_hor_l.append(np.nanmean(df_stats_hor[column]))
                        else:
                            for i_sim in range(len(ds_stats_l)):

                                # RCP code and statistics associated with current simulation.
                                rcp_code = Sim(ds_stats_l[i_sim][0]).rcp.code
                                ds_stats = ds_stats_l[i_sim][1][c.STAT_MEAN]

                                if (rcp_code != rcp.code) and not (("rcp" in rcp_code) and (rcp.code == c.RCPXX)):
                                    continue

                                # Select period and extract value.
                                if cntx.opt_ra:
                                    ds_stats_hor = wf_utils.sel_period(ds_stats.squeeze(),
                                                                       [hor.year_1, hor.year_2]).copy(deep=True)
                                else:
                                    ds_stats_hor = ds_stats.copy(deep=True)

                                # Add value.
                                val_hor_l.append(float(ds_stats_hor[vi_name].mean()))

                        if len(val_hor_l) == 0:
                            continue

                        # Calculate mean value.
                        if stat.code == c.STAT_MEAN:
                            val = np.mean(val_hor_l)
                        elif stat.code == c.STAT_MIN:
                            val = np.min(val_hor_l)
                        elif stat.code == c.STAT_MAX:
                            val = np.max(val_hor_l)
                        else:
                            val = np.quantile(val_hor_l, q=stat.centile/100)

                        # Add row.
                        stn_l.append(stn)
                        rcp_l.append(rcp.code)
                        hor_l.append(hor.code)
                        stat_l.append("none" if rcp.code == c.REF else stat.code)
                        centile_l.append(stat.centile)
                        val_l.append(round(val, 6))

            # Save results.
            if len(stn_l) > 0:

                # Build pandas dataframe.
                dict_pd = {"stn": stn_l, ("var" if cat == c.CAT_SCEN else "idx"): [vi_name] * len(stn_l),
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
    vi_code_grp = str(vi.group(varidx.code)) if varidx.is_group else varidx.code

    # Loop through stations.
    stns = cntx.stns if not cntx.opt_ra else [cntx.obs_src]
    for stn in stns:

        # Status message.
        msg = varidx.code
        if varidx.is_group:
            msg = vi_code_grp + "." + msg
        fu.log("Processing: " + stn + ", " + msg, True)

        # Path of files to be created.
        fn = varidx.name + "_" + dash_plot.MODE_RCP
        # CSV files:
        p_rcp_csv = cntx.d_fig(view_code, vi_code_grp + "_" + c.F_CSV) + fn + c.F_EXT_CSV
        p_sim_csv = p_rcp_csv.replace("_" + dash_plot.MODE_RCP, "_" + dash_plot.MODE_SIM)
        p_rcp_del_csv = p_rcp_csv.replace(c.F_EXT_CSV, "_delta" + c.F_EXT_CSV)
        p_sim_del_csv = p_sim_csv.replace(c.F_EXT_CSV, "_delta" + c.F_EXT_CSV)
        # PNG files.
        p_rcp_fig = cntx.d_fig(view_code, vi_code_grp) + fn + c.F_EXT_PNG
        p_sim_fig = p_rcp_fig.replace("_" + dash_plot.MODE_RCP + c.F_EXT_PNG, "_" + dash_plot.MODE_SIM + c.F_EXT_PNG)
        p_rcp_del_fig = p_rcp_fig.replace(c.F_EXT_PNG, "_delta" + c.F_EXT_PNG)
        p_sim_del_fig = p_sim_fig.replace(c.F_EXT_PNG, "_delta" + c.F_EXT_PNG)

        # Skip if no work required.
        save_csv = ((not os.path.exists(p_rcp_csv) or
                     not os.path.exists(p_sim_csv) or
                     not os.path.exists(p_rcp_del_csv) or
                     not os.path.exists(p_sim_del_csv)) and
                    (c.F_CSV in cntx.opt_ts_format))
        save_fig = ((not os.path.exists(p_rcp_fig) or
                     not os.path.exists(p_sim_fig) or
                     not os.path.exists(p_rcp_del_fig) or
                     not os.path.exists(p_sim_del_fig)) and
                    (c.F_PNG in cntx.opt_ts_format))
        if (not save_csv) and (not save_fig) and (not cntx.opt_force_overwrite):
            continue

        # Update context.
        cntx.code   = c.PLATFORM_SCRIPT
        cntx.view   = View(view_code)
        cntx.lib    = Lib(c.LIB_MAT)
        cntx.varidx = VarIdx(varidx.code)
        cntx.rcp    = RCP(c.RCPXX)
        cntx.sim    = Sim("")
        cntx.stats  = Stats()
        cntx.stats.add(Stat(c.STAT_CENTILE, cntx.opt_ts_centiles[0]))
        cntx.stats.add(Stat(c.STAT_CENTILE, cntx.opt_ts_centiles[1]))

        # Attempt loading CSV files into dataframes.
        if os.path.exists(p_rcp_csv) and os.path.exists(p_sim_csv) and \
           os.path.exists(p_rcp_del_csv) and os.path.exists(p_sim_del_csv):
            df_rcp = pd.read_csv(p_rcp_csv)
            df_sim = pd.read_csv(p_sim_csv)
            df_rcp_del = pd.read_csv(p_rcp_del_csv)
            df_sim_del = pd.read_csv(p_sim_del_csv)

        # Generate dataframes.
        else:
            df_stn = calc_ts_stn(stn, varidx)
            df_rcp, df_sim, df_rcp_del, df_sim_del =\
                dict(df_stn)["rcp"], dict(df_stn)["sim"], dict(df_stn)["rcp_delta"], dict(df_stn)["sim_delta"]

        # CSV file format ------------------------------------------------------------------------------------------

        if c.F_CSV in cntx.opt_ts_format:
            if (not os.path.exists(p_rcp_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_rcp, p_rcp_csv)
            if (not os.path.exists(p_sim_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_sim, p_sim_csv)
            if (not os.path.exists(p_rcp_del_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_rcp_del, p_rcp_del_csv)
            if (not os.path.exists(p_sim_del_csv)) or cntx.opt_force_overwrite:
                fu.save_csv(df_sim_del, p_sim_del_csv)

        # PNG file format ------------------------------------------------------------------------------------------

        if c.F_PNG in cntx.opt_ts_format:

            # Loop through modes (rcp|sim).
            for mode in [dash_plot.MODE_RCP, dash_plot.MODE_SIM]:

                # Loop through delta.
                for delta in ["False", "True"]:
                    cntx.delta = Delta(delta)

                    # Select dataset and output file name.
                    if mode == dash_plot.MODE_RCP:
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
    stn: str,
    varidx: VarIdx
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
    varidx: VarIdx
        Variable or index.

    Returns
    -------
    dict
        Dictionary of instances of pd.DataFrame.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract relevant information from the context.
    vi_name = varidx.name

    # Complete horizon (inncluding reference period).
    hor = Hor([min(min(cntx.per_hors)), max(max(cntx.per_hors))])

    # Initialize the structure that will hold the result for the current 'delta' iteration.
    dict_pd = {"year": list(range(hor.year_1, hor.year_2 + 1))}
    df_rcp, df_sim, df_rcp_delta, df_sim_delta =\
        pd.DataFrame(dict_pd), pd.DataFrame(dict_pd), pd.DataFrame(dict_pd), pd.DataFrame(dict_pd)

    # List required statistics.
    stats = Stats()
    stat_lower = Stat(c.STAT_CENTILE, cntx.opt_ts_centiles[0])
    stat_middle = Stat(c.STAT_MEAN)
    stat_upper = Stat(c.STAT_CENTILE, cntx.opt_ts_centiles[1])
    stats.add(stat_lower)
    stats.add(stat_middle)
    stats.add(stat_upper)

    # Calculate statistics for the reference data and each simulation.
    fu.log("Processing: " + cntx.obs_src + ", " + varidx.code + ", absolute values", True)
    ds_stats_abs_l = calc_stats_ref_sims(stn, cntx.view, cntx.varidx, delta=False)
    fu.log("Processing: " + cntx.obs_src + ", " + varidx.code + ", delta values", True)
    ds_stats_del_l = calc_stats_ref_sims(stn, cntx.view, cntx.varidx, delta=True)

    # Individual time series.
    for i_sim in range(len(ds_stats_abs_l)):

        # Extract simulation code and description.
        sim_code = ds_stats_abs_l[i_sim][0]
        ds_stats_abs_i = ds_stats_abs_l[i_sim][1][c.STAT_MEAN]
        ds_stats_del_i = ds_stats_del_l[i_sim][1][c.STAT_MEAN]

        # Subset years.
        ds_stats_abs_i = wf_utils.sel_period(ds_stats_abs_i, [hor.year_1, hor.year_2])
        ds_stats_del_i = wf_utils.sel_period(ds_stats_del_i, [hor.year_1, hor.year_2])

        # Extract values.
        vals_abs = list(ds_stats_abs_i[vi_name].values)
        vals_del = list(ds_stats_del_i[vi_name].values)

        # Record statistics.
        if sim_code == c.REF:
            n_nan = (hor.year_2 - hor.year_1 + 1 - len(vals_abs))
            df_rcp[sim_code] = vals_abs + [np.nan] * n_nan
            df_sim[sim_code] = vals_abs + [np.nan] * n_nan
            df_rcp_delta[sim_code] = [0.0] * len(vals_abs) + [np.nan] * n_nan
            df_sim_delta[sim_code] = [0.0] * len(vals_abs) + [np.nan] * n_nan
        else:
            df_sim[sim_code] = vals_abs
            df_sim_delta[sim_code] = vals_del

    # Loop through emission scenarios.
    for rcp in RCPs(cntx.rcps.code_l).items:

        for val_type in ["abs", "del"]:

            if val_type == "abs":
                ds_stats_l = ds_stats_abs_l
            else:
                ds_stats_l = ds_stats_del_l

            # Collect statistics.
            ds_stats_rcp_l = []
            for i_sim in range(len(ds_stats_l)):
                if rcp.code in ds_stats_l[i_sim][0]:
                    ds_stats_rcp_l.append(ds_stats_l[i_sim][1][c.STAT_MEAN])

            # Calculate ensemble statistics.
            xclim_logger_level = wf_utils.get_logger_level("root")
            wf_utils.set_logger_level("root", logging.CRITICAL)
            ds_ens = ensembles.create_ensemble(ds_stats_rcp_l)
            wf_utils.set_logger_level("root", xclim_logger_level)
            ds_stats_basic = ensembles.ensemble_mean_std_max_min(ds_ens)
            ds_stats_centile = ensembles.ensemble_percentiles(ds_ens, values=stats.centile_l, split=False)

            # Select years.
            da_stats_lower = ds_stats_centile.sel(percentiles=stat_lower.centile)[vi_name]
            da_stats_lower = wf_utils.sel_period(da_stats_lower, [hor.year_1, hor.year_2])
            da_stats_middle = ds_stats_basic[vi_name + "_" + c.STAT_MEAN]
            da_stats_middle = wf_utils.sel_period(da_stats_middle, [hor.year_1, hor.year_2])
            da_stats_upper = ds_stats_centile.sel(percentiles=stat_upper.centile)[vi_name]
            da_stats_upper = wf_utils.sel_period(da_stats_upper, [hor.year_1, hor.year_2])

            # Record mean, lower and upper annual values.
            if val_type == "abs":
                df_rcp[rcp.code + "_lower"] = da_stats_lower.values
                df_rcp[rcp.code + "_middle"] = da_stats_middle.values
                df_rcp[rcp.code + "_upper"] = da_stats_upper.values
            else:
                df_rcp_delta[rcp.code + "_lower"] = da_stats_lower.values
                df_rcp_delta[rcp.code + "_middle"] = da_stats_middle.values
                df_rcp_delta[rcp.code + "_upper"] = da_stats_upper.values

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
    if c.DIM_RLON in ds.dims:
        da_m = ds[var.name].rename({c.DIM_RLON: c.DIM_LONGITUDE, c.DIM_RLAT: c.DIM_LATITUDE})
    elif c.DIM_LON in ds.dims:
        da_m = ds[var.name].rename({c.DIM_LON: c.DIM_LONGITUDE, c.DIM_LAT: c.DIM_LATITUDE})
    else:
        da_m = ds[var.name]

    # Grouping frequency.
    freq_str = "time.month" if freq == c.FREQ_MS else "time.dayofyear"
    time_str = "M" if freq == c.FREQ_MS else "1D"

    # Summarize data per month.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        if cntx.opt_ra:
            da_m = da_m.mean(dim={c.DIM_LONGITUDE, c.DIM_LATITUDE})
        if freq != c.FREQ_MS:

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

                ds_m[var.name].attrs[c.ATTRS_UNITS] = ds[var.name].attrs[c.ATTRS_UNITS]
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
            ds_l.attrs[c.ATTRS_UNITS] = ds[var.name].attrs[c.ATTRS_UNITS]

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

    fu.log("Processing: " + cntx.obs_src + ", " + varidx.code, True)

    # Extract variable name, group and RCPs.
    varidx = VarIdx(varidx.code)
    vi_name = str(varidx.name)
    vi_code_grp = vi.group(varidx.code) if varidx.is_group else varidx.code
    rcps = RCPs([c.REF] + cntx.rcps.code_l + [c.RCPXX])

    # Get statistic string.
    def stat_str(
        _stat: Stat
    ):

        if _stat.code in [c.STAT_MEAN, c.STAT_MIN, c.STAT_MAX]:
            _stat_str = _stat.code
        else:
            _stat_str = _stat.centile_as_str

        return _stat_str

    # Get the path of the CSV and PNG files.
    def p_csv_fig(
        _rcp: RCP,
        _hor: Hor,
        _stat: Stat,
        _delta: Optional[bool] = False
    ) -> Tuple[str, str]:

        # PNG file.
        _d_fig = cntx.d_fig(c.VIEW_MAP, (vi_code_grp + cntx.sep if vi_code_grp != varidx.code else "") +
                            varidx.code + cntx.sep + _hor.code)
        _stat_str = stat_str(_stat)
        _fn_fig = vi_name + "_" + _rcp.code + "_" + str(_hor.year_1) + "_" + str(_hor.year_2) + "_" + _stat_str +\
            c.F_EXT_PNG
        _p_fig = _d_fig + _fn_fig
        if _delta:
            _p_fig = _p_fig.replace(c.F_EXT_PNG, "_delta" + c.F_EXT_PNG)

        # CSV file.
        _d_csv = cntx.d_fig(c.VIEW_MAP, (vi_code_grp + cntx.sep if vi_code_grp != varidx.code else "") +
                            varidx.code + "_csv" + cntx.sep + _hor.code)
        _fn_csv = _fn_fig.replace(c.F_EXT_PNG, c.F_EXT_CSV)
        _p_csv = _d_csv + _fn_csv
        if _delta:
            _p_csv = _p_csv.replace(c.F_EXT_CSV, "_delta" + c.F_EXT_CSV)

        return _p_csv, _p_fig

    # Prepare data -----------------------------------------------------------------------------------------------------

    # Calculate the overall minimum and maximum values (considering all maps for the current 'varidx').
    # There is one z-value per period.
    n_hor = len(cntx.per_hors)
    z_min_abs, z_max_abs = [np.nan] * n_hor, [np.nan] * n_hor
    z_min_del, z_max_del = [np.nan] * n_hor, [np.nan] * n_hor

    # Calculated datasets and item description.
    arr_ds_maps = []
    arr_items = []

    # Identify statistics to calculate.
    stats = Stats([c.STAT_MEAN])
    if cntx.opt_map_centiles is not None:
        for centile in cntx.opt_map_centiles:
            stats.add(Stat(c.STAT_CENTILE, centile))

    # Reference map (to calculate deltas).
    ds_map_ref = None

    # This will hold relevant NetCDF files.
    ds_l = []

    def load_netcdf_files():

        p_l = list_netcdf(cntx.obs_src, varidx, "path")
        for p in p_l:

            # Extract simulation code.
            sim_code_i = c.REF if "rcp" not in p else os.path.basename(p).replace(varidx.name + "_", "").\
                replace(c.F_EXT_NC, "")

            fu.log("Processing: " + cntx.obs_src + ", " + varidx.code + ", " + sim_code_i, True)

            # Load NetCDF and sort/rename dimensions.
            ds_i = fu.open_netcdf(p)
            ds_i = wf_utils.standardize_netcdf(ds_i, vi_name=vi_name)
            units = wf_utils.units(ds_i, vi_name)

            # Subset years.
            ds_i = wf_utils.sel_period(ds_i, [cntx.per_ref[0], max(max(cntx.per_hors))]).copy(deep=True)

            # Resample to the annual frequency.
            if varidx.is_summable:
                ds_i = ds_i.resample(time=c.FREQ_YS).sum()
            else:
                ds_i = ds_i.resample(time=c.FREQ_YS).mean()

            # Adjust units.
            ds_i[vi_name].attrs[c.ATTRS_UNITS] = units
            ds_i = wf_utils.set_units(ds_i, vi_name)

            # Add Dataset to list.
            ds_l.append([sim_code_i, ds_i])

    # Loop through horizons.
    hors = Hors(cntx.per_hors)
    for h in range(hors.count):
        hor = hors.items[h]
        hor_year_l = [hor.year_1, hor.year_2]

        # Loop through categories of emission scenarios.
        for rcp in rcps.items:

            # Skip if looking at the reference data and future period.
            if (rcp.code == c.REF) and (hor_year_l != cntx.per_ref):
                continue

            fu.log("Processing: " + cntx.obs_src + ", " + varidx.code + ", " + hor.code + ", " + rcp.code, True)

            ds_stats_rcp_l = []

            # Loop through statistics.
            for stat in stats.items:

                # Skip if looking at the reference data and a statistic other than the mean.
                if (rcp.code == c.REF) and (stat.code != c.STAT_MEAN):
                    continue

                # Statistics code as a string (ex: "mean", "c010").
                stat_code_as_str = stat.centile_as_str if stat.is_centile else stat.code

                # Path of CSV and PNG files.
                p_csv, p_fig = p_csv_fig(rcp, hor, stat)

                if os.path.exists(p_csv) and not cntx.opt_force_overwrite:

                    # Load data.
                    df = pd.read_csv(p_csv)

                    # Convert to DataArray.
                    df = pd.DataFrame(df, columns=[c.DIM_LONGITUDE, c.DIM_LATITUDE, "val"])
                    df = df.sort_values(by=[c.DIM_LATITUDE, c.DIM_LONGITUDE])
                    lat = list(set(df[c.DIM_LATITUDE]))
                    lat.sort()
                    lon = list(set(df[c.DIM_LONGITUDE]))
                    lon.sort()
                    arr = np.reshape(list(df["val"]), (len(lat), len(lon)))
                    da = xr.DataArray(data=arr, dims=[c.DIM_LATITUDE, c.DIM_LONGITUDE],
                                      coords=[(c.DIM_LATITUDE, lat), (c.DIM_LONGITUDE, lon)])
                    da.name = vi_name
                    ds_map = da.to_dataset()

                else:

                    # Load NetCDF files.
                    if len(ds_l) == 0:
                        load_netcdf_files()

                    # Select simulations.
                    ds_stats_hor_l = []
                    for i_sim in range(len(ds_l)):
                        rcp_code = Sim(ds_l[i_sim][0]).rcp.code
                        if (rcp_code == rcp.code) or (("rcp" in rcp_code) and (rcp.code == c.RCPXX)):
                            ds_stats_hor_l.append(ds_l[i_sim][1])

                    # Calculate statistics.
                    if len(ds_stats_rcp_l) == 0:
                        ds_stats_rcp_l = dict(calc_stats(ds_l=ds_stats_hor_l, view=c.VIEW_MAP, stn=cntx.obs_src,
                                                         varidx=varidx, rcp=rcp, sim=Sim(c.SIMXX), hor=hor, delta=False,
                                                         stats=stats, squeeze_coords=False, clip=True))

                    # Select the DataArray corresponding to the current statistics.
                    ds_map = ds_stats_rcp_l[stat_code_as_str]

                    # Calculate mean value (over time).
                    ds_map = ds_map.mean(dim=c.DIM_TIME)

                # Record current map, statistic and centile, RCP and period.
                arr_ds_maps.append(ds_map)
                arr_items.append([stat, rcp, hor])

                # Calculate reference map.
                if (ds_map_ref is None) and (stat.code == c.STAT_MEAN) and (rcp.code == c.REF) and\
                   (hor_year_l == cntx.per_ref):
                    ds_map_ref = ds_map

                # Extract values.
                vals_abs = ds_map[vi_name].values
                vals_del = None
                if hor_year_l != cntx.per_ref:
                    vals_del = ds_map[vi_name].values - ds_map_ref[vi_name].values

                # Record minimum and maximum values.
                z_min, z_max = np.nanmin(vals_abs), np.nanmax(vals_abs)
                z_min_abs[h] = z_min if np.isnan(z_min_abs[h]) else np.nanmin(np.array([z_min, z_min_abs[h]]))
                z_max_abs[h] = z_max if np.isnan(z_max_abs[h]) else np.nanmax(np.array([z_max, z_max_abs[h]]))
                if vals_del is not None:
                    z_min, z_max = np.nanmin(vals_del), np.nanmax(vals_del)
                    z_min_del[h] = z_min if np.isnan(z_min_del[h]) else np.nanmin(np.array([z_min, z_min_del[h]]))
                    z_max_del[h] = z_max if np.isnan(z_max_del[h]) else np.nanmax(np.array([z_max, z_max_del[h]]))

    # Generate maps ----------------------------------------------------------------------------------------------------

    # Loop through maps.
    for i in range(len(arr_ds_maps)):

        # Current map.
        ds_map = arr_ds_maps[i]

        # Current RCP and horizon.
        stat     = arr_items[i][0]
        rcp      = arr_items[i][1]
        hor      = arr_items[i][2]
        hor_as_l = [hor.year_1, hor.year_2]
        i_hor    = cntx.per_hors.index([hor.year_1, hor.year_2])

        # Perform twice (for values, then for deltas).
        for j in range(2):

            # Skip if delta map generation if the option is disabled or if it's the reference period.
            if (j == 1) and ((hor_as_l == cntx.per_ref) or (not cntx.opt_map_delta)):
                continue

            # Extract dataset (calculate delta if required).
            if j == 0:
                da_map = ds_map[vi_name]
            else:
                da_map = ds_map[vi_name] - ds_map_ref[vi_name]

            # Calculate minimum and maximum values.
            z_min = z_min_abs[i_hor] if j == 0 else z_min_del[i_hor]
            z_max = z_max_abs[i_hor] if j == 0 else z_max_del[i_hor]

            # PNG and CSV formats ----------------------------------------------------------------------------------

            # Path of output files.
            p_csv, p_fig = p_csv_fig(rcp, hor, stat, j == 1)

            # Create context.
            cntx.code   = c.PLATFORM_SCRIPT
            cntx.view   = View(c.VIEW_MAP)
            cntx.lib    = Lib(c.LIB_MAT)
            cntx.varidx = VarIdx(vi_name)
            cntx.rcp    = rcp
            cntx.hor    = hor
            cntx.stats  = Stats()
            for s in range(len(cntx.opt_map_centiles)):
                stats.add(Stat(c.STAT_CENTILE, cntx.opt_map_centiles[s]))
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
                ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_map_format))
            save_csv = cntx.opt_force_overwrite or\
                ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_map_format))

            # Create dataframe.
            arr_lon, arr_lat, arr_val = [], [], []
            for m in range(len(da_map.longitude.values)):
                for n in range(len(da_map.latitude.values)):
                    arr_lon.append(da_map.longitude.values[m])
                    arr_lat.append(da_map.latitude.values[n])
                    arr_val.append(da_map.values[n, m])
            dict_pd = {c.DIM_LONGITUDE: arr_lon, c.DIM_LATITUDE: arr_lat, "val": arr_val}
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
            p_tif = p_fig.replace(varidx.code + cntx.sep, varidx.code + "_" + c.F_TIF + cntx.sep).\
                replace(c.F_EXT_PNG, c.F_EXT_TIF)

            if (c.F_TIF in cntx.opt_map_format) and ((not os.path.exists(p_tif)) or cntx.opt_force_overwrite):

                # TODO: da_tif.rio.reproject is now crashing. It was working in July 2021.

                # Increase resolution.
                da_tif = da_map.copy()
                if cntx.opt_map_resolution > 0:
                    lat_vals = np.arange(min(da_tif.latitude), max(da_tif.latitude), cntx.opt_map_resolution)
                    lon_vals = np.arange(min(da_tif.longitude), max(da_tif.longitude), cntx.opt_map_resolution)
                    da_tif = da_tif.rename({c.DIM_LATITUDE: c.DIM_LAT, c.DIM_LONGITUDE: c.DIM_LON})
                    da_grid = xr.Dataset(
                        {c.DIM_LAT: ([c.DIM_LAT], lat_vals), c.DIM_LON: ([c.DIM_LON], lon_vals)})
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        da_tif = xe.Regridder(da_tif, da_grid, "bilinear")(da_tif)

                # Project data.
                da_tif.rio.set_crs("EPSG:4326")
                if cntx.opt_map_spat_ref != "EPSG:4326":
                    da_tif.rio.set_spatial_dims(c.DIM_LON, c.DIM_LAT, inplace=True)
                    da_tif = da_tif.rio.reproject(cntx.opt_map_spat_ref)
                    da_tif.values[da_tif.values == -9999] = np.nan
                    da_tif = da_tif.rename({"y": c.DIM_LAT, "x": c.DIM_LON})

                # Save.
                d = os.path.dirname(p_tif)
                if not (os.path.isdir(d)):
                    os.makedirs(d)
                da_tif.rio.to_raster(p_tif)


def conv_nc_csv(
    cat: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert NetCDF to CSV files.

    Parameters
    ----------
    cat : str
        Category: c.CAT_STN  = observations or reanalysis
                  c.CAT_SCEN = climate scenarios
                  c.CAT_IDX  = climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through stations.
    stns = cntx.stns if not cntx.opt_ra else [cntx.obs_src]
    for stn in stns:

        # Identify categories.
        if cat == c.CAT_STN:
            cat_l = [c.CAT_STN]
        elif cat == c.CAT_SCEN:
            cat_l = [c.CAT_OBS, c.CAT_RAW, c.CAT_REGRID, c.CAT_QQMAP]
        else:
            cat_l = [c.CAT_IDX]

        # Loop through categories.
        for cat in cat_l:

            # Loop through variables or indices.
            vi_code_l = cntx.varidxs.code_l if cat != c.CAT_IDX else vi.explode_idx_l(cntx.varidxx.code_l)
            for vi_code in vi_code_l:

                # Extract variable/index information.
                varidx = VarIdx(vi_code)

                # List NetCDF files.
                p_l = list(list_netcdf(stn, varidx, "path"))
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

                        # Directory.
                        if cat == c.CAT_STN:
                            d_cat = cntx.d_stn(vi_code)
                        elif cat == c.CAT_SCEN:
                            d_cat = cntx.d_scen(cat, vi_code)
                        else:
                            d_cat = cntx.d_idx(vi_code)

                        # Calculate the number of files processed (before conversion).
                        n_files_proc_before = len(list(glob.glob(d_cat + "*" + c.F_EXT_CSV)))

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
                        n_files_proc_after = len(list(glob.glob(d_cat + "*" + c.F_EXT_CSV)))

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
    vi_code_grp = vi.group(vi_code) if varidx.is_group else vi_code

    # Paths.
    p = p_l[i_file]
    p_csv = p.replace(c.F_EXT_NC, c.F_EXT_CSV).\
        replace(cntx.sep + vi_code_grp + "_", cntx.sep + vi_name + "_").\
        replace(cntx.sep + vi_code_grp + cntx.sep, cntx.sep + vi_code_grp + "_" + c.F_CSV + cntx.sep)

    if os.path.exists(p_csv) and (not cntx.opt_force_overwrite):
        if cntx.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Explode lists of codes and names.
    idx_names_exploded = vi.explode_idx_l(cntx.idxs.name_l)

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
        lon_l, lat_l = wf_utils.coords(ds, True)

    # Extract values.
    # Calculate average values (only if the analysis is based on observations at a station).
    val_l = list(ds[vi_name].values)
    if not cntx.opt_ra:
        if vi_name not in cntx.idxs.name_l:
            for i in range(n_time):
                val_l[i] = val_l[i].mean()
        else:
            if c.REF in p:
                for i in range(n_time):
                    val_l[i] = val_l[i][0][0]
            else:
                val_l = list(val_l[0][0])

    # Convert values to more practical units (if required).
    if varidx.is_summable and (ds[vi_name].attrs[c.ATTRS_UNITS] == c.UNIT_kg_m2s1):
        for i in range(n_time):
            val_l[i] = val_l[i] * c.SPD
    elif (vi_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]) and\
         (ds[vi_name].attrs[c.ATTRS_UNITS] == c.UNIT_K):
        for i in range(n_time):
            val_l[i] = val_l[i] - c.d_KC

    # Build pandas dataframe.
    if not cntx.opt_ra:
        dict_pd = {c.DIM_TIME: time_l, vi_name: val_l}
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
        dict_pd = {c.DIM_TIME: time_l_1d, c.DIM_LON: lon_l_1d, c.DIM_LAT: lat_l_1d, vi_name: val_l_1d}
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
        Frequency = {c.FREQ_D, c.FREQ_MS}
    title: str
        Plot title.
    i_trial: int
        Iteration number. The purpose is to attempt doing the analysis again. It happens once in a while that the
        dictionary is missing values, which results in the impossibility to build a dataframe and save it.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Paths.
    cat_fig = c.VIEW_CYCLE_MS if freq == c.FREQ_MS else c.VIEW_CYCLE_D
    p_fig = cntx.d_fig(cat_fig, varidx.name + cntx.sep + cntx.hor.desc) + title + c.F_EXT_PNG
    p_csv = p_fig.replace(cntx.sep + varidx.name + cntx.sep, cntx.sep + varidx.name + "_" + c.F_CSV + cntx.sep).\
        replace(c.F_EXT_PNG, c.F_EXT_CSV)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_cycle_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_cycle_format)))
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
            ds = wf_utils.sel_period(ds, per)
            if len(ds[c.DIM_TIME]) == 0:
                return
            if freq == c.FREQ_D:
                ds = wf_utils.remove_feb29(ds)

        # Convert units.
        units = ds[varidx.name].attrs[c.ATTRS_UNITS]
        if (varidx.name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]) and \
           (ds[varidx.name].attrs[c.ATTRS_UNITS] == c.UNIT_K):
            ds = ds - c.d_KC
        elif varidx.is_summable and (ds[varidx.name].attrs[c.ATTRS_UNITS] == c.UNIT_kg_m2s1):
            ds = ds * c.SPD
        ds[varidx.name].attrs[c.ATTRS_UNITS] = units

        # Calculate statistics.
        ds_l = calc_by_freq(ds, varidx, per, freq)

        n = 12 if freq == c.FREQ_MS else 365

        # Remove February 29th.
        if (freq == c.FREQ_D) and (len(ds_l[0][varidx.name]) > 365):
            for i in range(3):
                ds_l[i] = ds_l[i].rename_dims({"dayofyear": c.DIM_TIME})
                ds_l[i] = ds_l[i][varidx.name][ds_l[i][c.DIM_TIME] != 59].to_dataset()
                ds_l[i][c.DIM_TIME] = wf_utils.reset_calendar(ds_l[i], cntx.per_ref[0], cntx.per_ref[0], c.FREQ_D)
                ds_l[i][varidx.name].attrs[c.ATTRS_UNITS] = ds[varidx.name].attrs[c.ATTRS_UNITS]

        # Create dataframe.
        if freq == c.FREQ_D:
            dict_pd = {
                ("month" if freq == c.FREQ_MS else "day"): range(1, n + 1),
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
        if freq == c.FREQ_D:
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
        msg_err = "Unable to save " + ("cycle_d" if (freq == c.FREQ_D) else "cycle_m") +\
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
    cntx.view    = View(c.VIEW_CLUSTER)
    cntx.lib     = Lib(c.LIB_MAT)
    cntx.varidxs = cntx.cluster_vars
    cntx.rcp     = RCP(c.RCPXX)
    cntx.stats   = Stats()
    cntx.stats.add(Stat(c.STAT_CENTILE, cntx.opt_cluster_centiles[0]))
    cntx.stats.add(Stat(c.STAT_CENTILE, cntx.opt_cluster_centiles[1]))

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
            p_i = cntx.d_fig(c.VIEW_TS, var.code + "_" + c.F_CSV) + var.code + "_sim" + c.F_EXT_CSV
            p_csv_ts_l.append(p_i)
        n_cluster_max = len(dash_utils.get_shared_sims(p_csv_ts_l))

        # Loop through all combinations of
        for n_cluster in range(1, n_cluster_max):

            # Paths.
            p_csv = cntx.d_fig(c.VIEW_CLUSTER) + vars_str + "_" + c.F_CSV + cntx.sep +\
                vars_str + "_" + str(n_cluster) + c.F_EXT_CSV
            p_fig = p_csv.replace("_" + c.F_CSV, "").replace(c.F_EXT_CSV, c.F_EXT_PNG)

            # Determine if the analysis is required.
            analysis_enabled = cntx.opt_cluster
            save_fig = (cntx.opt_force_overwrite or
                        ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_cluster_format)))
            save_csv = (cntx.opt_force_overwrite or
                        ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_cluster_format)))
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
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.F_CSV + cntx.sep). \
        replace(c.F_EXT_PNG, c.F_EXT_CSV)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_diagnostic_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_diagnostic_format)))
    if not (save_fig or save_csv):
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Load datasets.
        da_ref = fu.open_netcdf(p_ref)[vi_name]
        if c.DIM_LONGITUDE in da_ref.dims:
            da_ref = da_ref.rename({c.DIM_LONGITUDE: c.DIM_RLON, c.DIM_LATITUDE: c.DIM_RLAT})
        da_sim = fu.open_netcdf(p_sim)[vi_name]
        da_sim_adj = fu.open_netcdf(p_sim_adj)[vi_name]

        # Select control point.
        if cntx.opt_ra:
            subset_ctrl_pt = False
            if c.DIM_RLON in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.rlon) > 1) or (len(da_ref.rlat) > 1)
            elif c.DIM_LON in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.lon) > 1) or (len(da_ref.lat) > 1)
            elif c.DIM_LONGITUDE in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.longitude) > 1) or (len(da_ref.latitude) > 1)
            if subset_ctrl_pt:
                if cntx.p_bounds == "":
                    da_ref = wf_utils.subset_ctrl_pt(da_ref)
                    da_sim = wf_utils.subset_ctrl_pt(da_sim)
                    da_sim_adj = wf_utils.subset_ctrl_pt(da_sim_adj)
                else:
                    da_ref = wf_utils.squeeze_lon_lat(da_ref)
                    da_sim = wf_utils.squeeze_lon_lat(da_sim)
                    da_sim_adj = wf_utils.squeeze_lon_lat(da_sim_adj)

        # Conversion coefficient.
        coef = 1
        delta_ref, delta_sim, delta_sim_adj = 0, 0, 0
        if varidx.is_summable:
            coef = c.SPD * 365
        elif vi_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
            if da_ref.units == c.UNIT_K:
                delta_ref = -c.d_KC
            if da_sim.units == c.UNIT_K:
                delta_sim = -c.d_KC
            if da_sim_adj.units == c.UNIT_K:
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
                   c.CAT_OBS: list(da_ref_mean.values) + [np.nan] * (n_sim_adj - n_ref),
                   c.CAT_SIM_ADJ: list(da_sim_adj_mean.values),
                   c.CAT_SIM: list(da_sim_mean.values)}
        df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:
        fig = wf_plot.plot_postprocess(df, varidx, title)
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
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.F_CSV + cntx.sep). \
        replace(c.F_EXT_PNG, c.F_EXT_CSV)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_diagnostic_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_diagnostic_format)))
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
            if c.DIM_RLON in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.rlon) > 1) or (len(da_ref.rlat) > 1)
            elif c.DIM_LON in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.lon) > 1) or (len(da_ref.lat) > 1)
            elif c.DIM_LONGITUDE in da_ref.dims:
                subset_ctrl_pt = (len(da_ref.longitude) > 1) or (len(da_ref.latitude) > 1)
            if subset_ctrl_pt:
                if cntx.p_bounds == "":
                    da_ref = wf_utils.subset_ctrl_pt(da_ref)
                    da_sim = wf_utils.subset_ctrl_pt(da_sim)
                else:
                    da_ref = wf_utils.squeeze_lon_lat(da_ref)
                    da_sim = wf_utils.squeeze_lon_lat(da_sim)
            else:
                da_ref = da_ref.squeeze()
                da_sim = da_sim.squeeze()

        # Convert date format if the need is.
        if da_ref.time.dtype == c.DTYPE_OBJ:
            da_ref[c.DIM_TIME] = wf_utils.reset_calendar(da_ref)

        # Units.
        units = [da_ref.units, da_sim.units]

        # Create dataframe.
        n_ref = len(list(da_ref.values))
        n_sim = len(list(da_sim.values))
        dict_pd = {"year": list(range(1, n_sim + 1)),
                   c.CAT_OBS: list(da_ref.squeeze().values) + [np.nan] * (n_sim - n_ref),
                   c.CAT_SIM: list(da_sim.values)}
        df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:
        fig = wf_plot.plot_workflow(df, varidx, units, title)
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    if save_csv:
        fu.save_csv(df, p_csv)
