# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions related to climate indices.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import datetime
import functools
import glob
import math
import multiprocessing
import numpy as np
import os.path
import sys
import xarray as xr
import warnings
from typing import Tuple, List, Optional, Union

# xclim libraries.
import xclim.indices as indices
import xclim.indices.generic as indices_gen
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units
from xclim.indices import run_length as rl
from xclim.indices.generic import select_time

# Workflow libraries.
import file_utils as fu
import statistics as stats
import utils
from def_constant import const as c
from def_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard.def_rcp import RCP, RCPs
from dashboard.def_varidx import VarIdx


def gen():

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate climate indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through indices.
    for idx in cntx.idxs.items:

        # Emission scenarios.
        rcps = RCPs([c.ref] + cntx.rcps.code_l)

        # Data preparation ---------------------------------------------------------------------------------------------

        fu.log("Selecting variables and indices.", True)

        # Select required variables.
        vi_code_l = []

        # Temperature.
        if idx.name in [c.i_tnx, c.i_tng, c.i_tropical_nights, c.i_tng_months_below, c.i_heat_wave_max_length,
                        c.i_heat_wave_total_length, c.i_tgg, c.i_etr, c.i_tn_days_below]:
            vi_code_l.append(c.v_tasmin)

        if idx.name in [c.i_tx90p, c.i_tx_days_above, c.i_hot_spell_frequency, c.i_hot_spell_max_length, c.i_txg,
                        c.i_txx, c.i_wsdi, c.i_heat_wave_max_length, c.i_heat_wave_total_length, c.i_tgg,
                        c.i_etr]:
            vi_code_l.append(c.v_tasmax)

        # Precipitation.
        if idx.name in [c.i_rx1day, c.i_rx5day, c.i_cwd, c.i_cdd, c.i_sdii, c.i_prcptot, c.i_r10mm, c.i_r20mm,
                        c.i_rnnmm, c.i_wet_days, c.i_dry_days, c.i_rain_season_start, c.i_rain_season_end,
                        c.i_rain_season_prcptot, c.i_rain_season_length, c.i_dry_spell_total_length, c.i_rain_season]:

            if idx.name != c.i_rain_season_length:
                vi_code_l.append(c.v_pr)

            if idx.name in [c.i_rain_season_end, c.i_rain_season]:
                if c.v_evspsblpot in cntx.vars.code_l:
                    vi_code_l.append(c.v_evspsblpot)
                elif c.v_evspsbl in cntx.vars.code_l:
                    vi_code_l.append(c.v_evspsbl)
                else:
                    vi_code_l.append("nan")

            if idx.name in [c.i_rain_season_end, c.i_rain_season_length, c.i_rain_season_prcptot]:
                vi_code_l.append(idx.code.replace(idx.name, c.i_rain_season_start))

            if idx.name == c.i_rain_season:
                vi_code_l.append("nan")

            if idx.name in [c.i_rain_season_end, c.i_rain_season]:
                vi_code_l.append(str(idx.params[len(idx.params) - 1]))

            if idx.name in [c.i_rain_season_length, c.i_rain_season_prcptot]:
                vi_code_l.append(idx.code.replace(idx.name, c.i_rain_season_end))

        # Temperature-precipitation.
        if idx.name == c.i_drought_code:
            vi_code_l.append(c.v_tas)
            vi_code_l.append(c.v_pr)

        # Wind.
        if idx.name == c.i_wg_days_above:
            vi_code_l.append(c.v_uas)
            vi_code_l.append(c.v_vas)

        elif idx.name == c.i_wx_days_above:
            vi_code_l.append(c.v_sfcwindmax)

        # Loop through stations.
        stns = cntx.stns if not cntx.opt_ra else [cntx.obs_src]
        for stn in stns:

            # Verify if this variable or index is available for the current station.
            fu.log("Verifying data availability (based on directories).", True)
            vi_code_l_avail = True
            for vi_code_i in vi_code_l:
                if vi_code_i != "nan":
                    ens = VarIdx(vi_code_i).ens
                    if ((ens == c.ens_cordex) and not os.path.isdir(cntx.d_scen(c.cat_qqmap, vi_code_i))) or\
                       ((ens != c.ens_cordex) and not os.path.isdir(cntx.d_idx(vi_code_i))):
                        vi_code_l_avail = False
                        break
            if not vi_code_l_avail:
                continue

            # Create mask.
            da_mask = None
            if stn == c.ens_era5_land:
                da_mask = fu.create_mask()

            # Loop through emissions scenarios.
            for rcp in rcps.items:

                fu.log("Processing: " + idx.code + ", " + stn + ", " + str(rcp.desc) + "", True)

                # List simulation files for the first variable. As soon as there is no file for one variable, the
                # analysis for the current RCP needs to abort.
                fu.log("Collecting simulation files.", True)
                varidx_0 = VarIdx(vi_code_l[0])
                if rcp.code == c.ref:
                    if varidx_0.is_var:
                        p_sim_l = cntx.d_scen(c.cat_obs, vi_code_l[0]) + varidx_0.name + "_" + stn + c.f_ext_nc
                    else:
                        p_sim_l = cntx.d_idx(vi_code_l[0]) + varidx_0.name + "_ref" + c.f_ext_nc
                    if type(p_sim_l) is str:
                        p_sim_l = [p_sim_l]
                else:
                    if varidx_0.is_var:
                        d = cntx.d_scen(c.cat_qqmap, vi_code_l[0])
                    else:
                        d = cntx.d_idx(vi_code_l[0])
                    p_sim_l = glob.glob(d + "*_" + rcp.code + c.f_ext_nc)
                if not p_sim_l:
                    continue

                # Remove simulations that are included in the exceptions lists.
                p_sim_filter = []
                for p in p_sim_l:
                    found = False
                    # List of simulation exceptions.
                    for e in cntx.sim_excepts:
                        if e.replace(c.f_ext_nc, "") in p:
                            found = True
                            break
                    # List of variable-simulation exceptions.
                    for e in cntx.var_sim_excepts:
                        if e.replace(c.f_ext_nc, "") in p:
                            found = True
                            break
                    # Add simulation.
                    if not found:
                        p_sim_filter.append(p)
                p_sim_l = p_sim_filter

                # Ensure that simulations are available for other variables than the first one.
                fu.log("Verifying data availability (based on NetCDF files).", True)
                if len(vi_code_l) > 1:
                    p_sim_fix_l = []
                    for p_sim_i in p_sim_l:
                        missing = False
                        for vi_code_j in vi_code_l[1:]:
                            if vi_code_j != "nan":
                                p_sim_j = VarIdx(vi_code_l[0]).equi_path(p_sim_i, vi_code_j, stn, rcp.code)
                                if not os.path.exists(p_sim_j):
                                    missing = True
                                    break
                        if not missing:
                            p_sim_fix_l.append(p_sim_i)
                    p_sim_l = p_sim_fix_l

                # Calculation ------------------------------------------------------------------------------------------

                fu.log("Calculating climate indices", True)

                n_sim = len(p_sim_l)
                d_idx = cntx.d_idx(idx.code)

                # Scalar mode.
                if cntx.n_proc == 1:
                    for i_sim in range(n_sim):
                        gen_single(idx, vi_code_l, p_sim_l, stn, rcp, da_mask, i_sim)

                # Parallel processing mode.
                else:

                    # Loop until all simulations have been processed.
                    while True:

                        # Calculate the number of processed files (before generation).
                        # This verification is based on the index NetCDF file.
                        n_sim_proc_before = len(list(glob.glob(d_idx + "*" + c.f_ext_nc)))

                        # Scalar processing mode.
                        scalar_required = False
                        if idx.name == c.i_prcptot:
                            scalar_required = not str(idx.params[0]).isdigit()
                        if (cntx.n_proc == 1) or scalar_required:
                            for i_sim in range(n_sim):
                                gen_single(idx, vi_code_l, p_sim_l, stn, rcp, da_mask, i_sim)

                        # Parallel processing mode.
                        else:

                            try:
                                fu.log("Splitting work between " + str(cntx.n_proc) + " threads.", True)
                                pool = multiprocessing.Pool(processes=min(cntx.n_proc, n_sim))
                                func = functools.partial(gen_single, idx, vi_code_l, p_sim_l, stn, rcp, da_mask)
                                pool.map(func, list(range(n_sim)))
                                pool.close()
                                pool.join()
                                fu.log("Fork ended.", True)
                            except Exception as e:
                                fu.log(str(e))
                                pass

                        # Calculate the number of processed files (after generation).
                        n_sim_proc_after = len(list(glob.glob(d_idx + "*" + c.f_ext_nc)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cntx.n_proc == 1) or (n_sim_proc_before == n_sim_proc_after):
                            break


def gen_single(
    idx: VarIdx,
    vi_code_l: [str],
    p_sim_l: [str],
    stn: str,
    rcp: RCP,
    da_mask: xr.DataArray,
    i_sim: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts observations to NetCDF.

    Parameters
    ----------
    idx : VarIdx
        Climate index.
    vi_code_l : [str]
        List of climate variables or indices.
    p_sim_l : [str]
        List of simulation files.
    stn : str
        Station name.
    rcp : RCP
        RCP emission scenario.
    da_mask : xr.DataArray
        Mask.
    i_sim : int
        Rank of simulation in 'p_sim'.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract variable name
    varidx_0 = VarIdx(vi_code_l[0])
    vi_name_0 = str(varidx_0.name)

    # Name of NetCDF file to generate.
    if rcp.code == c.ref:
        p_idx = cntx.d_idx(idx.code) + idx.name + "_ref" + c.f_ext_nc
    else:
        p_idx = cntx.d_idx(idx.code) + os.path.basename(p_sim_l[i_sim]).replace(vi_name_0, idx.name)

    # Exit loop if the file already exists (simulations files only; not reference file).
    if (rcp.code != c.ref) and os.path.exists(p_idx) and (not cntx.opt_force_overwrite):
        if cntx.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Load datasets (one per variable or index).
    ds_vi_l: List[xr.Dataset] = []
    for i_varidx in range(0, len(vi_code_l)):
        vi_code_i = vi_code_l[i_varidx]
        varidx_i = VarIdx(idx.name)
        if vi_code_i == "nan":
            continue

        try:
            # Open dataset.
            p_sim_j = VarIdx(vi_code_l[0]).equi_path(p_sim_l[i_sim], vi_code_i, stn, rcp.code)
            ds = fu.open_netcdf(p_sim_j)

            # Remove February 29th and select reference period.
            if (rcp.code == c.ref) and (varidx_i.ens == c.ens_cordex):
                ds = utils.remove_feb29(ds)
                ds = utils.sel_period(ds, cntx.per_ref)

            # Adjust temperature units.
            if varidx_i.name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
                if ds[vi_code_i].attrs[c.attrs_units] == c.unit_K:
                    ds[vi_code_i] = ds[vi_code_i] - c.d_KC
                elif rcp.code == c.ref:
                    ds[vi_code_i][c.attrs_units] = c.unit_C
                ds[vi_code_i].attrs[c.attrs_units] = c.unit_C

        except KeyError as e:
            ds = None

        # Add dataset.
        ds_vi_l.append(ds)

    # Calculate the 90th percentile of tasmax for the reference period.
    da_tx90p = None
    if (idx.name == c.i_wsdi) and (rcp.code == c.ref):
        da_tx90p = xr.DataArray(percentile_doy(ds_vi_l[0][c.v_tasmax], per=0.9))

    # Merge threshold value and unit, if required. Ex: "0.0 C" for temperature.
    params_str = []
    for i in range(len(idx.params)):
        vi_param = idx.params[i]

        # Convert thresholds (percentile to absolute value) ------------------------------------------------

        if (idx.name == c.i_tx90p) or ((idx.name == c.i_wsdi) and (i == 0)):
            vi_param = "90p"
        if (idx.name in [c.i_tx_days_above, c.i_tng_months_below, c.i_tx90p, c.i_prcptot, c.i_tropical_nights]) or\
           ((idx.name in [c.i_hot_spell_frequency, c.i_hot_spell_max_length, c.i_wsdi]) and (i == 0)) or \
           ((idx.name in [c.i_heat_wave_max_length, c.i_heat_wave_total_length]) and (i <= 1)) or \
           ((idx.name in [c.i_wg_days_above, c.i_wx_days_above]) and (i == 0)):

            if "p" in str(vi_param):
                vi_param = float(vi_param.replace("p", ""))
                if rcp.code == c.ref:
                    # Calculate percentile.
                    if (idx.name in [c.i_tx90p, c.i_hot_spell_frequency, c.i_hot_spell_max_length, c.i_wsdi]) or\
                       (i == 1):
                        vi_param = ds_vi_l[i][c.v_tasmax].quantile(float(vi_param) / 100.0).values.ravel()[0]
                    elif idx.name in [c.i_heat_wave_max_length, c.i_heat_wave_total_length]:
                        vi_param = ds_vi_l[i][c.v_tasmin].quantile(float(vi_param) / 100.0).values.ravel()[0]
                    elif idx.name == c.i_prcptot:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            da_i = ds_vi_l[i][c.v_pr].resample(time=c.freq_YS).sum(dim=c.dim_time)
                        dims = utils.coord_names(ds_vi_l[i])
                        vi_param = da_i.mean(dim=dims).quantile(float(vi_param) / 100.0).values.ravel()[0] * c.spd
                    elif idx.name in [c.i_wg_days_above, c.i_wx_days_above]:
                        if idx.name == c.i_wg_days_above:
                            da_uas = ds_vi_l[0][c.v_uas]
                            da_vas = ds_vi_l[1][c.v_vas]
                            da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas)
                        else:
                            da_vv = ds_vi_l[0][c.v_sfcwindmax]
                        vi_param = da_vv.quantile(float(vi_param) / 100.0).values.ravel()[0]
                    # Round value and save it.
                    vi_param = float(round(vi_param, 2))
                    cntx.idxs.params[cntx.idxs.name_l.index(idx.name)][i] = vi_param
                else:
                    vi_param = cntx.idxs.params[cntx.idxs.name_l.index(idx.name)][i]

        # Combine threshold and unit -----------------------------------------------------------------------

        if (idx.name in [c.i_tx90p, c.i_tropical_nights]) or\
           ((idx.name in [c.i_hot_spell_frequency, c.i_hot_spell_max_length, c.i_wsdi,
                          c.i_tn_days_below, c.i_tx_days_above]) and (i == 0)) or \
           ((idx.name in [c.i_heat_wave_max_length, c.i_heat_wave_total_length]) and (i <= 1)):
            vi_ref = str(vi_param) + " " + c.unit_C
            vi_fut = str(vi_param + c.d_KC) + " " + c.unit_K
            params_str.append(vi_ref if (rcp.code == c.ref) else vi_fut)

        elif idx.name in [c.i_cwd, c.i_cdd, c.i_r10mm, c.i_r20mm, c.i_rnnmm, c.i_wet_days, c.i_dry_days, c.i_sdii]:
            params_str.append(str(vi_param) + " mm/day")

        elif (idx.name in [c.i_wg_days_above, c.i_wx_days_above]) and (i == 1):
            params_str.append(str(vi_param) + " " + c.unit_m_s)

        elif not ((idx.name in [c.i_wg_days_above, c.i_wx_days_above]) and (i == 4)):
            params_str.append(str(vi_param))

    # Exit loop if the file already exists (reference file only).
    if not ((rcp.code == c.ref) and os.path.exists(p_idx) and (not cntx.opt_force_overwrite)):

        # Will hold data arrays and units.
        da_idx_l    = []
        idx_units_l = []
        idx_name_l  = []

        # Temperature --------------------------------------------------------------------------------------------------

        if idx.name in [c.i_tx_days_above, c.i_tx90p]:

            # Collect required datasets and parameters.
            da_tasmax = ds_vi_l[0][c.v_tasmax]
            thresh = params_str[0]
            start_date = end_date = ""
            if len(params_str) > 1:
                start_date = str(params_str[1]).replace("nan", "")
                end_date = str(params_str[2]).replace("nan", "")

            # Calculate index.
            da_idx = tx_days_above(da_tasmax, thresh, start_date, end_date)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        if idx.name == c.i_tn_days_below:

            # Collect required datasets and parameters.
            da_tasmin = ds_vi_l[0][c.v_tasmin]
            thresh = params_str[0]
            start_date = end_date = ""
            if len(params_str) > 1:
                start_date = str(params_str[1]).replace("nan", "")
                end_date = str(params_str[2]).replace("nan", "")

            # Calculate index.
            da_idx = tn_days_below(da_tasmin, thresh, start_date, end_date)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name == c.i_tng_months_below:

            # Collect required datasets and parameters.
            da_tasmin = ds_vi_l[0][c.v_tasmin]
            param_tasmin = float(params_str[0])
            if da_tasmin.attrs[c.attrs_units] != c.unit_C:
                param_tasmin += c.d_KC

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.tn_mean(da_tasmin, freq=c.freq_MS))
                da_idx = xr.DataArray(indices_gen.threshold_count(da_idx, "<", param_tasmin, c.freq_YS))
            da_idx = da_idx.astype(float)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name in [c.i_hot_spell_frequency, c.i_hot_spell_max_length, c.i_wsdi]:

            # Collect required datasets and parameters.
            da_tasmax = ds_vi_l[0][c.v_tasmax]
            param_tasmax = params_str[0]
            param_ndays = int(float(params_str[1]))

            # Calculate index.
            if idx.name == c.i_hot_spell_frequency:
                da_idx = xr.DataArray(
                    indices.hot_spell_frequency(da_tasmax, param_tasmax, param_ndays).values)
            elif idx.name == c.i_hot_spell_max_length:
                da_idx = xr.DataArray(
                    indices.hot_spell_max_length(da_tasmax, param_tasmax, param_ndays).values)
            else:
                da_idx = xr.DataArray(
                    indices.warm_spell_duration_index(da_tasmax, da_tx90p, param_ndays).values)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name in [c.i_heat_wave_max_length, c.i_heat_wave_total_length]:

            # Collect required datasets and parameters.
            da_tasmin = ds_vi_l[0][c.v_tasmin]
            da_tasmax = ds_vi_l[1][c.v_tasmax]
            param_tasmin = params_str[0]
            param_tasmax = params_str[1]
            window = int(float(params_str[2]))

            # Calculate index.
            if idx.name == c.i_heat_wave_max_length:
                da_idx = xr.DataArray(
                    heat_wave_max_length(da_tasmin, da_tasmax, param_tasmin, param_tasmax, window).values)
            else:
                da_idx = xr.DataArray(heat_wave_total_length(
                    da_tasmin, da_tasmax, param_tasmin, param_tasmax, window).values)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name in [c.i_txg, c.i_txx]:

            # Collect required datasets and parameters.
            da_tasmax = ds_vi_l[0][c.v_tasmax]

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx.name == c.i_txg:
                    da_idx = xr.DataArray(indices.tx_mean(da_tasmax))
                else:
                    da_idx = xr.DataArray(indices.tx_max(da_tasmax))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_C)

        elif idx.name in [c.i_tnx, c.i_tng, c.i_tropical_nights]:

            # Collect required datasets and parameters.
            da_tasmin = ds_vi_l[0][c.v_tasmin]

            # Calculate index.
            if idx.name in [c.i_tnx, c.i_tng]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    if idx.name == c.i_tnx:
                        da_idx = xr.DataArray(indices.tn_max(da_tasmin))
                        idx_units = c.unit_C
                    else:
                        da_idx = xr.DataArray(indices.tn_mean(da_tasmin))
                        idx_units = c.unit_C
            else:
                param_tasmin = params_str[0]
                da_idx = xr.DataArray(indices.tropical_nights(da_tasmin, param_tasmin))
                idx_units = c.unit_1

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(idx_units)

        elif idx.name in [c.i_tgg, c.i_etr]:

            # Collect required datasets and parameters.
            da_tasmin = ds_vi_l[0][c.v_tasmin]
            da_tasmax = ds_vi_l[1][c.v_tasmax]

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx.name == c.i_tgg:
                    da_idx = xr.DataArray(indices.tg_mean(indices.tas(da_tasmin, da_tasmax)))
                else:
                    da_idx = xr.DataArray(indices.tx_max(da_tasmax) - indices.tn_min(da_tasmin))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_C)

        elif idx.name == c.i_drought_code:

            # Collect required datasets and parameters.
            da_tas = ds_vi_l[0][c.v_tas]
            da_pr  = ds_vi_l[1][c.v_pr]
            da_lon, da_lat = utils.coords(ds_vi_l[0])

            # Calculate index
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.drought_code(da_tas, da_pr, da_lat)).resample(time=c.freq_YS).mean()

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        # Precipitation ------------------------------------------------------------------------------------------------

        elif idx.name in [c.i_rx1day, c.i_rx5day, c.i_prcptot]:

            # Collect required datasets and parameters.
            da_pr = ds_vi_l[0][c.v_pr]

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx.name == c.i_rx1day:
                    da_idx = xr.DataArray(indices.max_1day_precipitation_amount(da_pr, c.freq_YS))
                elif idx.name == c.i_rx5day:
                    da_idx = xr.DataArray(indices.max_n_day_precipitation_amount(da_pr, 5, c.freq_YS))
                else:
                    start_date = end_date = ""
                    if len(params_str) == 3:
                        start_date = str(params_str[1]).replace("nan", "")
                        end_date = str(params_str[2]).replace("nan", "")
                    da_idx = xr.DataArray(precip_accumulation(da_pr, start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(da_idx.attrs[c.attrs_units])

        elif idx.name in [c.i_cwd, c.i_cdd, c.i_r10mm, c.i_r20mm, c.i_rnnmm, c.i_wet_days, c.i_dry_days, c.i_sdii]:

            # Collect required datasets and parameters.
            da_pr = ds_vi_l[0][c.v_pr]
            param_pr = params_str[0]

            # Calculate index.
            if idx.name in c.i_cwd:
                da_idx = xr.DataArray(
                    indices.maximum_consecutive_wet_days(da_pr, param_pr, c.freq_YS))
            elif idx.name in c.i_cdd:
                da_idx = xr.DataArray(
                    indices.maximum_consecutive_dry_days(da_pr, param_pr, c.freq_YS))
            elif idx.name in [c.i_r10mm, c.i_r20mm, c.i_rnnmm, c.i_wet_days]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    da_idx = xr.DataArray(indices.wetdays(da_pr, param_pr, c.freq_YS))
            elif idx.name == c.i_dry_days:
                da_idx = xr.DataArray(indices.dry_days(da_pr, param_pr, c.freq_YS))
            else:
                da_idx = xr.DataArray(indices.daily_pr_intensity(da_pr, param_pr))
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name == c.i_rain_season_start:

            # Collect required datasets and parameters.
            da_pr      = ds_vi_l[0][c.v_pr]
            thresh_wet = params_str[0] + " mm"
            window_wet = int(params_str[1])
            thresh_dry = params_str[2] + " mm"
            dry_days   = int(params_str[3])
            window_dry = int(params_str[4])
            start_date = str(params_str[5]).replace("nan", "")
            end_date   = str(params_str[6]).replace("nan", "")

            # Calculate index.
            da_idx =\
                xr.DataArray(rain_season_start(da_pr, thresh_wet, window_wet, thresh_dry, dry_days, window_dry,
                                               start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name == c.i_rain_season_end:

            # Collect required datasets and parameters.
            da_pr  = ds_vi_l[0][c.v_pr]
            da_etp = None
            if ds_vi_l[1] is not None:
                if c.v_evspsblpot in cntx.vars.code_l:
                    da_etp = ds_vi_l[1][c.v_evspsblpot]
                elif c.v_evspsbl in cntx.vars.code_l:
                    da_etp = ds_vi_l[1][c.v_evspsbl]
            da_start = None
            if (len(ds_vi_l) > 2) and (ds_vi_l[2] is not None):
                da_start = ds_vi_l[2][c.i_rain_season_start]
            da_start_next = None
            if (len(ds_vi_l) > 3) and (ds_vi_l[3] is not None):
                da_start_next = ds_vi_l[3][c.i_rain_season_start]
            op         = params_str[0]
            thresh     = params_str[1] + " mm"
            window     = -1 if str(params_str[2]) == "nan" else int(params_str[2])
            etp_rate   = ("0" if str(params_str[3]) == "nan" else params_str[3]) + " mm"
            start_date = str(params_str[4]).replace("nan", "")
            end_date   = str(params_str[5]).replace("nan", "")

            # Calculate index.
            da_idx = xr.DataArray(rain_season_end(da_pr, da_etp, da_start, da_start_next, op, thresh, window, etp_rate,
                                                  start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name == c.i_rain_season_length:

            # Collect required datasets and parameters.
            da_start = ds_vi_l[0][c.i_rain_season_start]
            da_end   = ds_vi_l[1][c.i_rain_season_end]

            # Calculate index.
            da_idx = rain_season_length(da_start, da_end)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        elif idx.name == c.i_rain_season_prcptot:

            # Collect required datasets and parameters.
            da_pr = ds_vi_l[0][c.v_pr]
            da_start = ds_vi_l[1][c.i_rain_season_start]
            da_end = ds_vi_l[2][c.i_rain_season_end]

            # Calculate index.
            da_idx = xr.DataArray(rain_season_prcptot(da_pr, da_start, da_end))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(VarIdx(idx.name).unit)

        elif idx.name == c.i_rain_season:

            # Collect required datasets and parameters.
            da_pr = ds_vi_l[0][c.v_pr]
            # Rain start:
            s_thresh_wet = params_str[0] + " mm"
            s_window_wet = int(params_str[1])
            s_thresh_dry = params_str[2] + " mm"
            s_dry_days   = int(params_str[3])
            s_window_dry = int(params_str[4])
            s_start_date = str(params_str[5]).replace("nan", "")
            s_end_date   = str(params_str[6]).replace("nan", "")
            # Rain end:
            da_etp = None
            if ds_vi_l[1] is not None:
                if c.v_evspsblpot in cntx.vars.code_l:
                    da_etp = ds_vi_l[1][c.v_evspsblpot]
                elif c.v_evspsbl in cntx.vars.code_l:
                    da_etp = ds_vi_l[1][c.v_evspsbl]
            da_start_next = None
            if ds_vi_l[3] is not None:
                da_start_next = ds_vi_l[3][c.i_rain_season_start]
            e_op         = params_str[7]
            e_thresh     = params_str[8] + " mm"
            e_window     = -1 if str(params_str[9]) == "nan" else int(params_str[9])
            e_etp_rate   = ("0" if str(params_str[10]) == "nan" else params_str[10]) + " mm"
            e_start_date = str(params_str[11]).replace("nan", "")
            e_end_date   = str(params_str[12]).replace("nan", "")

            # Calculate indices.
            da_start, da_end, da_length, da_prcptot =\
                rain_season(da_pr, da_etp, da_start_next, s_thresh_wet, s_window_wet, s_thresh_dry, s_dry_days,
                            s_window_dry, s_start_date, s_end_date, e_op, e_thresh, e_window, e_etp_rate,
                            e_start_date, e_end_date)

            # Add to list.
            da_idx_l = [da_start, da_end, da_length, da_prcptot]
            idx_units_l = [c.unit_1, c.unit_1, c.unit_1, VarIdx(c.i_rain_season_prcptot).unit]
            idx_name_l = [c.i_rain_season_start, c.i_rain_season_end, c.i_rain_season_length, c.i_rain_season_prcptot]

        elif idx.name == c.i_dry_spell_total_length:

            # Collect required datasets and parameters.
            da_pr = ds_vi_l[0][c.v_pr]
            thresh = params_str[0] + " mm"
            window = int(params_str[1])
            op = params_str[2]
            start_date = end_date = ""
            if len(params_str) == 5:
                start_date = str(params_str[3])
                end_date = str(params_str[4])

            # Calculate index.
            da_idx = xr.DataArray(dry_spell_total_length(da_pr, thresh, window, op, "YS", start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        # Wind ---------------------------------------------------------------------------------------------------------

        elif idx.name in [c.i_wg_days_above, c.i_wx_days_above]:

            # Collect required datasets and parameters.
            param_vv     = float(params_str[0])
            param_vv_neg = params_str[1]
            param_dd     = float(params_str[2])
            param_dd_tol = float(params_str[3])
            start_date = end_date = ""
            if len(params_str) == 6:
                start_date = str(params_str[4]).replace("nan", "")
                end_date = str(params_str[5]).replace("nan", "")
            if idx.name == c.i_wg_days_above:
                da_uas = ds_vi_l[0][c.v_uas]
                da_vas = ds_vi_l[1][c.v_vas]
                da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas, param_vv_neg)
            else:
                da_vv = ds_vi_l[0][c.v_sfcwindmax]
                da_dd = None

            # Calculate index.
            da_idx =\
                xr.DataArray(w_days_above(da_vv, da_dd, param_vv, param_dd, param_dd_tol, start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(c.unit_1)

        if len(idx_name_l) == 0:
            idx_name_l = [idx.name]

        # Loop through data arrays.
        ds_idx = None
        for i in range(len(da_idx_l)):
            da_idx = da_idx_l[i]
            idx_units = idx_units_l[i]

            # Assign units.
            da_idx.attrs[c.attrs_units] = idx_units

            # Convert to float. This is required to ensure that 'nan' values are not transformed into integers.
            da_idx = da_idx.astype(float)

            # Rename dimensions and attribute a name.
            da_idx = utils.rename_dimensions(da_idx)
            da_idx.name = idx.name

            # Interpolate (to remove nan values).
            if np.isnan(da_idx).astype(int).max() > 0:
                da_idx = utils.interpolate_na_fix(da_idx)

            # Sort dimensions to fit input data.
            utils.standardize_netcdf(da_idx, template=ds_vi_l[0][varidx_0.name])

            # Apply mask.
            if da_mask is not None:
                da_idx = utils.apply_mask(da_idx, da_mask)

            # Create dataset.
            if i == 0:
                ds_idx = da_idx.to_dataset(name=idx_name_l[i])
                ds_idx.attrs[c.attrs_units] = idx_units
                ds_idx.attrs[c.attrs_sname] = idx.name
                ds_idx.attrs[c.attrs_lname] = idx.name
                ds_idx = utils.copy_coords(ds_vi_l[0], ds_idx)

            # Add data array.
            ds_idx[idx_name_l[i]] = utils.copy_coords(ds_vi_l[0][varidx_0.name], da_idx)

        # Adjust calendar.
        ds_idx = ds_idx.squeeze()
        years = utils.extract_date_field(ds_vi_l[0], "year")
        ds_idx[c.dim_time] = utils.reset_calendar(ds_idx, min(years), max(years), c.freq_YS)

        # Save result to NetCDF file.
        desc = cntx.sep + idx.name + cntx.sep + os.path.basename(p_idx)
        fu.save_netcdf(ds_idx, p_idx, desc=desc)

    # Convert percentile threshold values for climate indices. This is sometimes required in time series.
    if (rcp.code == c.ref) and (idx.name == c.i_prcptot):
        ds_idx = fu.open_netcdf(p_idx)
        da_idx = ds_idx.mean(dim=[c.dim_longitude, c.dim_latitude])[idx.name]
        param_pr = params_str[0]
        if "p" in str(param_pr):
            param_pr = float(param_pr.replace("p", ""))
            cntx.idxs.params[cntx.idx.code_l.index(idx.code)][0] = \
                float(round(da_idx.quantile(float(param_pr) / 100.0).values.ravel()[0], 2))

    if cntx.n_proc > 1:
        fu.log("Work done!", True)


def precip_accumulation(
    da_pr: xr.DataArray,
    start_date: str,
    end_date: str
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent function in xclim.indices.

    Parameters
    ----------
    da_pr: xr.DataArray
        Precipitation data.
    start_date: str
        First day of year on which season can start ("mm-dd").
    end_date: str
        Last day of year on which season can start ("mm-dd")
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert start and end dates to doy.
    start_doy = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday

    # Subset based on day of year.
    da_pr = utils.subset_doy(da_pr, start_doy, end_doy)

    # Calculate index.
    da_idx = xr.DataArray(indices.precip_accumulation(da_pr, freq=c.freq_YS))

    return da_idx


def tx_days_above(
    da_tasmax: xr.DataArray,
    thresh: str,
    start_date: str,
    end_date: str
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent function in xclim.indices.

    Parameters
    ----------
    da_tasmax: xr.DataArray
        Maximum temperature data.
    thresh: str
        Maximum temperature threshold value.
    start_date: str
        First day of year where season can start ("mm-dd").
    end_date: str
        Last day of year where season can start ("mm-dd")..
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert start and end dates to doy.
    start_doy = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday

    # Subset based on day of year.
    da_tasmax = utils.subset_doy(da_tasmax, start_doy, end_doy)

    # Calculate index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_idx = xr.DataArray(indices.tx_days_above(da_tasmax, thresh).values)

    return da_idx


def tn_days_below(
    da_tasmin: xr.DataArray,
    thresh: str,
    start_date: str,
    end_date: str
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent function in xclim.indices.

    Parameters
    ----------
    da_tasmin: xr.DataArray
        Minimum temperature data.
    thresh: str
        Minimum temperature threshold value.
    start_date: str
        First day of year where season can start ("mm-dd").
    end_date: str
        Last day of year where season can start ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert start and end dates to doy.
    start_doy = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday

    # Subset based on day of year.
    da_tasmin = utils.subset_doy(da_tasmin, start_doy, end_doy)

    # Calculate index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_idx = xr.DataArray(indices.tn_days_below(da_tasmin, thresh).values)

    return da_idx


def heat_wave_max_length(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    param_tasmin: str = "22.0 degC",
    param_tasmax: str = "30 degC",
    window: int = 3,
    freq: str = c.freq_YS
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent non-working function in xclim.indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    param_tasmin = convert_units_to(param_tasmin, tasmin)
    param_tasmax = convert_units_to(param_tasmax, tasmax)

    # Adjust calendars.
    if tasmin.time.dtype != tasmax.time.dtype:
        tasmin[c.dim_time] = tasmin[c.dim_time].astype("datetime64[ns]")
        tasmax[c.dim_time] = tasmax[c.dim_time].astype("datetime64[ns]")

    # Call the xclim function if time dimension is the same.
    n_tasmin = len(tasmin[c.dim_time])
    n_tasmax = len(tasmax[c.dim_time])

    if n_tasmin == n_tasmax:
        return indices.heat_wave_max_length(tasmin, tasmax, param_tasmin, param_tasmax, window, freq)

    # Calculate manually.
    else:

        cond = tasmin
        if n_tasmax > n_tasmin:
            cond = tasmax
        for t in cond.time:
            if (t.values in tasmin[c.dim_time]) and (t.values in tasmax[c.dim_time]):
                cond[cond[c.dim_time] == t] =\
                    (tasmin[tasmin[c.dim_time] == t] > param_tasmin) &\
                    (tasmax[tasmax[c.dim_time] == t] > param_tasmax)
            else:
                cond[cond[c.dim_time] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)
        max_l = group.map(rl.longest_run, dim=c.dim_time)

        return max_l.where(max_l >= window, 0)


def heat_wave_total_length(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    param_tasmin: str = "22.0 degC",
    param_tasmax: str = "30 degC",
    window: int = 3,
    freq: str = c.freq_YS
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent non-working function in xclim.indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    param_tasmin = convert_units_to(param_tasmin, tasmin)
    param_tasmax = convert_units_to(param_tasmax, tasmax)

    # Adjust calendars.
    if tasmin.time.dtype != tasmax.time.dtype:
        tasmin[c.dim_time] = tasmin[c.dim_time].astype("datetime64[ns]")
        tasmax[c.dim_time] = tasmax[c.dim_time].astype("datetime64[ns]")

    # Call the xclim function if time dimension is the same.
    n_tasmin = len(tasmin[c.dim_time])
    n_tasmax = len(tasmax[c.dim_time])

    if n_tasmin == n_tasmax:
        return indices.heat_wave_total_length(tasmin, tasmax, param_tasmin, param_tasmax, window, freq)

    # Calculate manually.
    else:

        cond = tasmin
        if n_tasmax > n_tasmin:
            cond = tasmax
        for t in cond.time:
            if (t.values in tasmin[c.dim_time]) and (t.values in tasmax[c.dim_time]):
                cond[cond[c.dim_time] == t] =\
                    (tasmin[tasmin[c.dim_time] == t] > param_tasmin) &\
                    (tasmax[tasmax[c.dim_time] == t] > param_tasmax)
            else:
                cond[cond[c.dim_time] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)

        return group.map(rl.windowed_run_count, args=(window,), dim=c.dim_time)


@declare_units(
    pr="[precipitation]",
    thresh="[length]"
)
def dry_spell_total_length(
    pr: xr.DataArray,
    thresh: str = "1.0 mm",
    window: int = 3,
    op: str = "sum",
    freq: str = "YS",
    start_date: str = "",
    end_date: str = ""
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Return the total number of days in dry periods of n days and more, during which the maximum or accumulated
    precipitation on a window of n days is under the threshold.

    Parameters
    ----------
    pr: xr.DataArray:
        Daily precipitation.
    thresh: str
        Accumulated precipitation value under which a period is considered dry.
    window: int
        Number of days where the maximum or accumulated precipitation is under threshold.
    op: {"max", "sum"}
        Reduce operation..
    freq: str
      Resampling frequency.
    start_date: str
        First day of year to consider ("mm-dd").
    end_date: str
        Last day of year to consider ("mm-dd").

    Returns
    -------
    xarray.DataArray
      The {freq} total number of days in dry periods of minimum {window} days.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")
    thresh = convert_units_to(thresh, pram)

    # Eliminate negative values.
    pram = xr.where(pram < (thresh if op == "max" else 0), 0, pram)
    pram.attrs["units"] = "mm"

    # Identify dry days.
    dry = None
    for i in range(2):
        pram_i = pram if dry is None else pram.sortby("time", ascending=False)
        if op == "max":
            mask_i = xr.DataArray(pram_i.rolling(time=window).max() < thresh)
        else:
            mask_i = xr.DataArray(pram_i.rolling(time=window).sum() < thresh)
        dry_i = (mask_i.rolling(time=window).sum() >= 1).shift(time=-(window - 1))
        if dry is None:
            dry = dry_i
        else:
            dry_i = dry_i.sortby("time", ascending=True)
            dt = (dry.time - dry.time[0]).dt.days
            dry = xr.where(dt > len(dry.time) - window - 1, dry_i, dry)

    # Identify days that are between 'start_date' and 'start_date'.
    doy_start = 1
    if start_date != "":
        doy_start = datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    doy_end = 365
    if end_date != "":
        doy_end = datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    bisextile_year_fix = xr.where(pram.time.dt.year % 4 == 0, 1, 0)
    doy_start = ([doy_start] * len(pram.time)) if doy_start < 60 else (doy_start + bisextile_year_fix)
    doy_end = ([doy_end] * len(pram.time)) if doy_end < 60 else (doy_end + bisextile_year_fix)
    doy = xr.where(doy_end >= doy_start,
                   (pram.time.dt.dayofyear >= doy_start) & (pram.time.dt.dayofyear <= doy_end),
                   (pram.time.dt.dayofyear <= doy_end) | (pram.time.dt.dayofyear >= doy_start))

    # Calculate the number of dry days per year.
    out = (dry & doy).astype(float).resample(time=freq).sum("time")

    return to_agg_units(out, pram, "count")


@declare_units(pr="[precipitation]", thresh="[length]")
def dry_spell_total_length_20220120(
    pr: xr.DataArray,
    thresh: str = "1.0 mm",
    window: int = 3,
    op: str = "sum",
    freq: str = "YS",
    **indexer,
) -> xr.DataArray:

    """
    Total length of dry spells

    Total number of days in dry periods of a minimum length, during which the maximum or
    accumulated precipitation within a window of the same length is under a threshold.

    Parameters
    ----------
    pr : xarray.DataArray
      Daily precipitation.
    thresh : str
      Accumulated precipitation value under which a period is considered dry.
    window : int
      Number of days where the maximum or accumulated precipitation is under threshold.
    op : {"max", "sum"}
      Reduce operation.
    freq : str
      Resampling frequency.
    indexer :
      Indexing parameters to compute the indicator on a temporal subset of the data.
      It accepts the same arguments as :py:func:`xclim.indices.generic.select_time`.
      Indexing is done after finding the dry days, but before finding the spells.

    Returns
    -------
    xarray.DataArray
      The {freq} total number of days in dry periods of minimum {window} days.

    Notes
    -----
    The algorithm assumes days before and after the timeseries are "wet", meaning that
    the condition for being considered part of a dry spell is stricter on the edges. For
    example, with `window=3` and `op='sum'`, the first day of the series is considered
    part of a dry spell only if the accumulated precipitation within the first 3 days is
    under the threshold. In comparison, a day in the middle of the series is considered
    part of a dry spell if any of the three 3-day periods of which it is part are
    considered dry (so a total of five days are included in the computation, compared to only 3.)
    """
    pram = rate2amount(pr, out_units="mm")
    thresh = convert_units_to(thresh, pram)

    pram_pad = pram.pad(time=(0, window))
    mask = getattr(pram_pad.rolling(time=window), op)() < thresh
    dry = (mask.rolling(time=window).sum() >= 1).shift(time=-(window - 1))
    dry = dry.isel(time=slice(0, pram.time.size)).astype(float)

    out = select_time(dry, **indexer).resample(time=freq).sum("time")
    return to_agg_units(out, pram, "count")


@declare_units(
    pr="[precipitation]",
    s_thresh_wet="[length]",
    s_thresh_dry="[length]",
    e_thresh="[length]"
)
def rain_season(
    pr: xr.DataArray,
    etp: xr.DataArray = None,
    start_next: xr.DataArray = None,
    s_thresh_wet: str = "25.0 mm",
    s_window_wet: int = 3,
    s_thresh_dry: str = "1.0 mm",
    s_dry_days: int = 7,
    s_window_dry: int = 30,
    s_start_date: str = "",
    s_end_date: str = "",
    e_op: str = "max",
    e_thresh: str = "5.0 mm",
    e_window: int = 20,
    e_etp_rate: str = "",
    e_start_date: str = "",
    e_end_date: str = ""
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate rain season start, end, length and accumulated precipitation.

    Parameters
    ----------
    pr: xr.DataArray
        Daily precipitation.
    etp: xr.DataArray
        Daily evapotranspiration.
    start_next: xr.DataArray
        First day of the next rain season.
    s_thresh_wet: str
        Accumulated precipitation threshold associated with {s_window_wet}.
    s_window_wet: int
        Number of days when accumulated precipitation is above {s_thresh_wet}.
    s_thresh_dry: str
        Daily precipitation threshold associated with {s_window_dry].
    s_dry_days: int
        Maximum number of dry days in {s_window_tot}.
    s_window_dry: int
        Number of days, after {s_window_wet}, during which daily precipitation is not greater than or equal to
        {s_thresh_dry} for {s_dry_days} consecutive days.
    s_start_date: str
        First day of year when season can start ("mm-dd").
    s_end_date: str
        Last day of year when season can start ("mm-dd").
    e_op: str
        Resampling operator = {"max", "sum", "etp}
        If "max": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season ends when no daily precipitation greater than {e_thresh} have occurred over a period of
            {e_window} days.
        If "sum": based on a total amount of precipitation received during the last days of the rain season.
            The rain season ends when the total amount of precipitation is less than {e_thresh} over a period of
            {e_window} days.
        If "etp": calculation is based on the period required for a water column of height {e_thresh] to evaporate,
            considering that any amount of precipitation received during that period must evaporate as well. If {etp} is
            not available, the evapotranspiration rate is assumed to be {e_etp_rate}.
    e_thresh: str
        Maximum or accumulated precipitation threshold associated with {e_window}.
        If {e_op} == "max": maximum daily precipitation  during a period of {e_window} days.
        If {e_op} == "sum": accumulated precipitation over {e_window} days.
        If {e_op} == "etp": height of water column that must evaporate.
    e_window: int
        If {e_op} in ["max", "sum"]: number of days used to verify if the rain season is ending.
    e_etp_rate: str
        If {e_op} == "etp": evapotranspiration rate.
        Otherwise: not used.
    e_start_date: str
        First day of year at or after which the season can end ("mm-dd").
    e_end_date: str
        Last day of year at or before which the season can end ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    def rename_dimensions(da: xr.DataArray, lat_name: str = "latitude", lon_name: str = "longitude") -> xr.DataArray:
        if ("location" not in da.dims) and ((lat_name not in da.dims) or (lon_name not in da.dims)):
            if "dim_0" in list(da.dims):
                da = da.rename({"dim_0": "time"})
                da = da.rename({"dim_1": lat_name, "dim_2": lon_name})
            elif ("lat" in list(da.dims)) or ("lon" in list(da.dims)):
                da = da.rename({"lat": lat_name, "lon": lon_name})
            elif ("rlat" in list(da.dims)) or ("rlon" in list(da.dims)):
                da = da.rename({"rlat": lat_name, "rlon": lon_name})
            elif (lat_name not in list(da.dims)) and (lon_name not in list(da.dims)):
                if lat_name == "latitude":
                    da = da.expand_dims(latitude=1)
                if lon_name == "longitude":
                    da = da.expand_dims(longitude=1)
        return da

    def reorder_dimensions(da: xr.DataArray) -> xr.DataArray:
        if "location" not in da.dims:
            da = da.transpose("time", "latitude", "longitude")
        else:
            da = da.transpose("location", "time")
        return da

    # Rename dimensions and reorder dimensions.
    pr = reorder_dimensions(rename_dimensions(pr))
    if etp is not None:
        etp = reorder_dimensions(rename_dimensions(etp))

    # Calculate rain season start.
    start = xr.DataArray(rain_season_start(pr, s_thresh_wet, s_window_wet, s_thresh_dry, s_dry_days, s_window_dry,
                                           s_start_date, s_end_date))

    # Calculate rain season end.
    end = xr.DataArray(rain_season_end(pr, etp, start, start_next, e_op, e_thresh, e_window, e_etp_rate,
                                       e_start_date, e_end_date))

    # Calculate rain season length.
    length = xr.DataArray(rain_season_length(start, end))

    # Calculate rain quantity.
    prcptot = xr.DataArray(rain_season_prcptot(pr, start, end))

    return start, end, length, prcptot


@declare_units(
    pr="[precipitation]",
    thresh_wet="[length]",
    thresh_dry="[length]"
)
def rain_season_start(
    pr: xr.DataArray,
    thresh_wet: str = "25.0 mm",
    window_wet: int = 3,
    thresh_dry: str = "1.0 mm",
    dry_days: int = 7,
    window_dry: int = 30,
    start_date: str = "",
    end_date: str = ""
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Detect the first day of the rain season.

    Rain season starts on the first day of a sequence of {window_wet} days with accumulated precipitation greater than
    or equal to {thresh_wet} that is followed by a period of {window_dry} days with fewer than {dry_days} consecutive
    days with less than {thresh_dry} daily precipitation. The search is constrained by {start_date} and {end_date}."


    Parameters
    ----------
    pr: xr.DataArray
        Precipitation data.
    thresh_wet: str
        Accumulated precipitation threshold associated with {window_wet}.
    window_wet: int
        Number of days when accumulated precipitation is above {thresh_wet}.
    thresh_dry: str
        Daily precipitation threshold associated with {window_dry}.
    dry_days: int
        Maximum number of dry days in {window_dry}.
    window_dry: int
        Number of days, after {window_wet}, during which daily precipitation is not greater than or equal to
        {thresh_dry} for {dry_days} consecutive days.
    start_date: str
        First day of year when season can start ("mm-dd").
    end_date: str
        Last day of year when season can start ("mm-dd").

    Returns
    -------
    xr.DataArray, [dimensionless]
        Rain season start (day of year).

    Examples
    --------
    Successful season start:
        . . . . 10 10 10 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
                 ^
    False start:
        . . . . 10 10 10 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
    Not even a start:
        . . . .  8  8  8 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
    given the numbers correspond to daily precipitation, based on default parameter values.

    References
    ----------
    This index was suggested by:
    Sivakumar, M.V.K. (1988). Predicting rainy season potential from the onset of rains in Southern Sahelian and
    Sudanian climatic zones of West Africa. Agricultural and Forest Meteorology, 42(4): 295-305.
    https://doi.org/10.1016/0168-1923(88)90039-1
    and by:
    Dodd, D.E.S. & Jolliffe, I.T. (2001) Early detection of the start of the wet season in semiarid tropical climates of
    Western Africa. Int. J. Climatol., 21, 1251‑1262. https://doi.org/10.1002/joc.640
    This corresponds to definition no. 2, which is a simplification of an index mentioned in:
    Jolliffe, I.T. & Sarria-Dodd, D.E. (1994) Early detection of the start of the wet season in tropical climates. Int.
    J. Climatol., 14: 71-76. https://doi.org/10.1002/joc.3370140106
    which is based on:
    Stern, R.D., Dennett, M.D., & Garbutt, D.J. (1981) The start of the rains in West Africa. J. Climatol., 1: 59-68.
    https://doi.org/10.1002/joc.3370010107
    --------------------------------------------------------------------------------------------------------------------
    """

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")
    thresh_wet = convert_units_to(thresh_wet, pram)
    thresh_dry = convert_units_to(thresh_dry, pram)

    # Eliminate negative values.
    pram = xr.where(pram < 0, 0, pram)
    pram.attrs["units"] = "mm"

    # Assign search boundaries.
    start_doy = 1
    if start_date != "":
        start_doy = datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365
    if end_date != "":
        end_doy = datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if (start_date == "") and (end_date != ""):
        start_doy = 1 if end_doy == 365 else end_doy + 1
    elif (start_date != "") and (end_date == ""):
        end_doy = 365 if start_doy == 1 else start_doy - 1

    # Flag the first day of each sequence of {window_wet} days with a total of {thresh_wet} in precipitation
    # (assign True).
    wet = xr.DataArray(pram.rolling(time=window_wet).sum() >= thresh_wet)\
        .shift(time=-(window_wet - 1), fill_value=False)

    # Identify dry days (assign 1).
    dry_day = xr.where(pram < thresh_dry, 1, 0)

    # Identify each day that is not followed by a sequence of {window_dry} days within a period of {window_tot} days,
    # starting after {window_wet} days (assign True).
    dry_seq = None
    for i in range(window_dry - dry_days - 1):
        dry_day_i = dry_day.shift(time=-(i + window_wet), fill_value=False)
        dry_seq_i = xr.DataArray(dry_day_i.rolling(time=dry_days).sum() >= dry_days)\
            .shift(time=-(dry_days - 1), fill_value=False)
        if i == 0:
            dry_seq = dry_seq_i.copy()
        else:
            dry_seq = dry_seq | dry_seq_i
    no_dry_seq = (dry_seq.astype(bool) == 0)

    # Flag days between {start_date} and {end_date} (or the opposite).
    if end_doy >= start_doy:
        doy = (pram.time.dt.dayofyear >= start_doy) & (
            pram.time.dt.dayofyear <= end_doy
        )
    else:
        doy = (pram.time.dt.dayofyear <= end_doy) | (
            pram.time.dt.dayofyear >= start_doy
        )

    # Obtain the first day of each year when conditions apply.
    start = (wet & no_dry_seq & doy).resample(time="YS").\
        map(rl.first_run, window=1, dim="time", coord="dayofyear")
    start = xr.where((start < 1) | (start > 365), np.nan, start)
    start.attrs["units"] = "1"

    return start


@declare_units(
    pr="[precipitation]",
    thresh="[length]",
    etp_rate="[length]"
)
def rain_season_end(
    pr: xr.DataArray,
    etp: xr.DataArray = None,
    start: xr.DataArray = None,
    start_next: xr.DataArray = None,
    op: str = "max",
    thresh: str = "5.0 mm",
    window: int = 20,
    etp_rate: str = "0.0 mm",
    start_date: str = "",
    end_date: str = ""
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Detect the last day of the rain season.

    Three methods are available:
    - If {op}=="max", season ends when no daily precipitation is greater than {thresh} over a period of {window} days.
    - If {op}=="sum", season ends when cumulated precipitation over a period of {window} days is smaller than {thresh}.
    - If {op}=="etp", season ends after a water column of height {thresh} has evaporated at daily rate specified in
      {etp} or {etp_rate}, considering that the cumulated precipitation during this period must also evaporate.
    Search is constrained by {start_date} and {end_date}.

    Parameters
    ----------
    pr: xr.DataArray
        Daily precipitation.
    etp: xr.DataArray
        Daily evapotranspiration.
    start: xr.DataArray
        First day of the current rain season.
    start_next: xr.DataArray
        First day of the next rain season.
    op: str
        Resampling operator = {"max", "sum", "etp}
        If "max": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season ends when no daily precipitation greater than {thresh} have occurred over a period of
            {window} days.
        If "sum": based on a total amount of precipitation received during the last days of the rain season.
            The rain season ends when the total amount of precipitation is less than {thresh} over a period of
            {window} days.
        If "etp": calculation is based on the period required for a water column of height {thresh] to evaporate,
            considering that any amount of precipitation received during that period must evaporate as well. If {etp} is
            not available, the evapotranspiration rate is assumed to be {etp_rate}.
    thresh: str
        Maximum or accumulated precipitation threshold associated with {window}.
        If {op} == "max": maximum daily precipitation  during a period of {window} days.
        If {op} == "sum": accumulated precipitation over {window} days.
        If {op} == "etp": height of water column that must evaporate.
    window: int
        If {op} in ["max", "sum"]: number of days used to verify if the rain season is ending.
    etp_rate: str
        If {op} == "etp": evapotranspiration rate.
        Otherwise: not used.
    start_date: str
        First day of year at or after which the season can end ("mm-dd").
    end_date: str
        Last day of year at or before which the season can end ("mm-dd").

    Returns
    -------
    xr.DataArray, [dimensionless]
        Rain season end (day of year).

    Examples
    --------
    Successful season end with {op} == "max":
        . . . . 5 0 1 0 1 0 1 0 1 0 1 0 1 0 4 0 1 0 1 0 1 . . . (pr)
                ^
    Successful season end with {op} == "sum":
        . 5 5 5 5 0 1 0 2 0 3 0 4 0 3 0 2 0 1 0 1 0 1 0 1 . . . (pr)
                ^
    Successful season end with {op} == "etp":
        . 5 5 5 5 3 3 3 3 5 0 0 1 1 0 5 . . (pr)
        . 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 . . (etp_rate)
                                  ^
    given the numbers correspond to daily precipitation or evapotranspiration, based on default parameter values.

    References
    ----------
    The algorithm corresponding to {op} = "max", referred to as the agronomic criterion, is suggested by:
    Somé, L. & Sivakumar, M.V.k. (1994). Analyse de la longueur de la saison culturale en fonction de la date de début
    des pluies au Burkina Faso. Compte rendu des travaux no 1: Division du sol et Agroclimatologie. INERA, Burkina Faso,
    43 pp.
    It can be applied to a country such as Ivory Coast, which has a bimodal regime near its coast.
    The algorithm corresponding to {op} = "etp" is applicable to Sahelian countries with a monomodal regime, as
    mentioned by Food Security Cluster (May 23rd, 2016):
    https://fscluster.org/mali/document/les-previsions-climatiques-de-2016-du
    This includes countries such as Burkina Faso, Senegal, Mauritania, Gambia, Guinea and Bissau.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get the dimension name associated with a coordinate name.
    def dim_name(
        da: xr.DataArray,
        coord_name: str
    ) -> str:
        dims = list(da.dims)
        index = [idx for idx, s in enumerate(dims) if coord_name in s]
        return "" if len(index) == 0 else dims[index[0]]

    # Copy dimension names from a DataArray of variables (da_b) to a DataArray of climate indices (da_a).
    def copy_dim_names(
        da_a: xr.DataArray,
        da_b: xr.DataArray
    ) -> xr.DataArray:
        lon_b, lat_b = dim_name(da_b, "lon"), dim_name(da_b, "lat")
        lon_a, lat_a = dim_name(da_a, "lon"), dim_name(da_a, "lat")
        if (lon_b not in ["", lon_a]) and (lon_a != ""):
            da_a = da_a.rename({lon_a: lon_b})
        if (lat_b not in ["", lat_a]) and (lat_a != ""):
            da_a = da_a.rename({lat_a: lat_b})
        return da_a

    # Copy dimension names.
    if start is not None:
        start = copy_dim_names(start, pr)
    if start_next is not None:
        start_next = copy_dim_names(start_next, pr)

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")
    etpam = None
    if etp is not None:
        etpam = rate2amount(etp, out_units="mm")
    thresh = convert_units_to(thresh, pram)
    etp_rate = convert_units_to(etp_rate, etpam if etpam is not None else pram)

    # Eliminate negative values.
    pram = xr.where(pram < 0, 0, pram)
    pram.attrs["units"] = "mm"
    if etpam is not None:
        etpam = xr.where(etpam < 0, 0, etpam)
        etpam.attrs["units"] = "mm"

    # Assign search boundaries.
    start_doy = 1
    if start_date != "":
        start_doy = datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365
    if end_date != "":
        end_doy = datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if (start_date == "") and (end_date != ""):
        start_doy = 1 if end_doy == 365 else end_doy + 1
    elif (start_date != "") and (end_date == ""):
        end_doy = 365 if start_doy == 1 else start_doy - 1

    # Flag days between {start_date} and {end_date} (or the opposite).
    dayofyear = pram.time.dt.dayofyear.astype(float)
    if end_doy >= start_doy:
        doy = (dayofyear >= start_doy) & (dayofyear <= end_doy)
    else:
        doy = (dayofyear <= end_doy) | (dayofyear >= start_doy)

    end = None

    if op == "etp":

        # Calculate the minimum length of the period.
        window_min = math.ceil(thresh / etp_rate) if (etp_rate > 0) else 0
        window_max =\
            (end_doy - start_doy + 1 if (end_doy >= start_doy) else (365 - start_doy + 1 + end_doy)) - window_min
        if etp_rate == 0:
            window_min = window_max

        # Window must be varied until it's size allows for complete evaporation.
        for window_i in list(range(window_min, window_max + 1)):

            # Flag the day before each sequence of {dt} days that results in evaporating a water column, considering
            # precipitation falling during this period (assign 1).
            if etpam is None:
                dry_seq = xr.DataArray(
                    (pram.rolling(time=window_i).sum() + thresh) <= (window_i * etp_rate)
                )
            else:
                dry_seq = xr.DataArray(
                    (pram.rolling(time=window_i).sum() + thresh) <= etpam.rolling(time=window_i).sum()
                )

            # Obtain the first day of each year when conditions apply.
            end_i = (dry_seq & doy).resample(time="YS").\
                map(rl.first_run, window=1, dim="time", coord="dayofyear")

            # Update the cells that were not assigned yet.
            if end is None:
                end = end_i.copy()
            else:
                sel = np.isnan(end) & ((np.isnan(end_i).astype(int) == 0) | (end_i < end))
                end = xr.where(sel, end_i, end)

            # Exit loop if all cells were assigned a value.
            window = window_i
            if np.isnan(end).astype(int).sum() == 0:
                break

    else:

        # Shift datasets to simplify the analysis.
        dt = 0 if end_doy >= start_doy else start_doy - 1
        pram_shift = pram.copy().shift(time=-dt, fill_value=False)
        doy = doy.shift(time=-dt, fill_value=False)

        # Determine if it rains (assign 1) or not (assign 0).
        wet = xr.where(pram_shift < thresh, 0, 1) if op == "max" else xr.where(pram_shift == 0, 0, 1)

        # Flag each day (assign 1) before a sequence of:
        # {window} days with no amount reaching {thresh}:
        if op == "max":
            dry_seq = xr.DataArray(wet.rolling(time=window).sum() == 0)
        # {window} days with a total amount reaching {thresh}:
        else:
            dry_seq = xr.DataArray(pram_shift.rolling(time=window).sum() < thresh)
        dry_seq = dry_seq.shift(time=-window, fill_value=False)

        # Obtain the first day of each year when conditions apply.
        end = (dry_seq & doy).resample(time="YS").\
            map(rl.first_run, window=1, dim="time", coord="dayofyear")

        # Shift result to the right.
        end += dt
        if end.max() > 365:
            transfer = xr.ufuncs.maximum(end - 365, 0).shift(time=1, fill_value=np.nan)
            end = xr.where(end > 365, np.nan, end)
            end = xr.where(np.isnan(transfer).astype(bool) == 0, transfer, end)

        # Rain season can't end on (or after) the first day of the last moving {window}, because we ignore the weather
        # past the end of the dataset.
        end = xr.where((end > 365 - window) & (end == end[len(end.time) - 1]), np.nan, end)

    # Rain season can't end unless the last day is rainy or the window comprises rainy days.
    def rain_near_end(loc: str = "") -> xr.DataArray:
        if loc == "":
            end_loc = end
            pram_loc = pram
        else:
            end_loc = end[end.location == loc].squeeze()
            pram_loc = pram[pram.location == loc].squeeze()
        day = pram_loc.astype(int)
        for t in range(len(day["time"])):
            day[t] = t
        n_days = 0
        for t in range(len(end_loc.time)):
            n_days_t = int(xr.DataArray(pram_loc.time.dt.year == end_loc[t].time.dt.year).astype(int).sum())
            # if not np.isnan(end_loc[t]):
            if not end_loc[t].isnull().all():
                # pos_end = int(end_loc[t]) + n_days - 1
                pos_end = end_loc[t].astype(int) + n_days - 1
                if op in ["max", "sum"]:
                    pos_win_1 = pos_end
                    # pos_win_2 = min(pos_win_1 + window, n_days_t + n_days)
                    pos_win_2 = xr.ufuncs.minimum(pos_win_1 + window, n_days_t + n_days)
                else:
                    pos_win_2 = pos_end
                    # pos_win_1 = max(0, pos_end - window)
                    pos_win_1 = xr.ufuncs.maximum(0, pos_end - window)
                # pos_range = [min(pos_win_1, pos_win_2), max(pos_win_1, pos_win_2)]
                pos_range = [xr.ufuncs.minimum(pos_win_1, pos_win_2), xr.ufuncs.maximum(pos_win_1, pos_win_2)]
                # Condition A: the windows comprises rainy days.
                # cond_a = pram_loc.isel(time=slice(pos_range[0], pos_range[1])).sum(dim="time") > 0
                mask_a = xr.DataArray((day >= pos_range[0]) & (day <= pos_range[1])).astype(int)
                cond_a = xr.DataArray((pram_loc * mask_a).resample(time="YS").sum(dim="time") > 0).astype(int)
                # Condition B: the last day is rainy.
                # cond_b = pram_loc.isel(time=pos_end) > 0
                mask_b = xr.DataArray(day == pos_end).astype(int)
                cond_b = xr.DataArray((pram_loc * mask_b).resample(time="YS").sum(dim="time") > 0).astype(int)
                # if (cond_a | cond_b) == False:
                #     end_loc[t] = np.nan
                end_loc[t] = xr.where(cond_a + cond_b == 0, np.nan, end_loc)[t]
            n_days += n_days_t
        return end_loc
    if "location" not in pram.dims:
        end = rain_near_end()
    else:
        locations = list(pram.location.values)
        for i in range(len(locations)):
            end[end.location == locations[i]] = rain_near_end(locations[i])

    # Adjust or discard rain end values that are not compatible with the current or next season start values.
    # If the season ends before or on start day, discard rain end.
    if start is not None:
        sel = np.array(start.isnull().astype(int) == 0) &\
              np.array(end.isnull().astype(int) == 0) &\
              (end.values <= start.values)
        end = xr.where(sel, np.nan, end)

    # If the season ends after or on start day of the next season, the end day of the current season becomes the day
    # before the next season.
    if start_next is not None:
        sel = np.array(start_next.isnull().astype(int) == 0) &\
              np.array(end.isnull().astype(int) == 0) &\
              (end.values >= start_next.values)
        end = xr.where(sel, start_next - 1, end)
        end = xr.where(end < 1, 365, end)
    end.attrs["units"] = "1"

    return end


def rain_season_length(
    start: xr.DataArray,
    end: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the length of the rain season.

    Parameters
    ----------
    start : xr.DataArray
        Rain season start (first day of year).
    end: xr.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xr.DataArray, [dimensionless]
        Rain season length (days/freq).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Start and end dates in the same calendar year.
    if start.mean() <= end.mean():
        length = end - start + 1

    # Start and end dates not in the same year (left shift required).
    else:
        length = xr.DataArray(xr.ones_like(start) * 365) - start + end.shift(time=-1, fill_value=np.nan) + 1

    # Eliminate negative values. This is a safety measure as this should not happen.
    length = xr.where(length < 0, 0, length)
    length.attrs["units"] = "days"

    return length


@declare_units(
    pr="[precipitation]"
)
def rain_season_prcptot(
    pr: xr.DataArray,
    start: xr.DataArray,
    end: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine precipitation amount during rain season.

    Parameters
    ----------
    pr: xr.DataArray
        Daily precipitation.
    start: xr.DataArray
        Rain season start (first day of year).
    end: xr.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xr.DataArray
        Rain season accumulated precipitation (mm/year).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get the dimension name associated with a coordinate name.
    def dim_name(
        da: xr.DataArray,
        coord_name: str
    ) -> str:
        dims = list(da.dims)
        index = [idx for idx, s in enumerate(dims) if coord_name in s]
        return "" if len(index) == 0 else dims[index[0]]

    # Copy dimension names from a DataArray of variables (da_b) to a DataArray of climate indices (da_a).
    def copy_dim_names(
        da_a: xr.DataArray,
        da_b: xr.DataArray
    ) -> xr.DataArray:
        lon_b, lat_b = dim_name(da_b, "lon"), dim_name(da_b, "lat")
        lon_a, lat_a = dim_name(da_a, "lon"), dim_name(da_a, "lat")
        if (lon_b not in ["", lon_a]) and (lon_a != ""):
            da_a = da_a.rename({lon_a: lon_b})
        if (lat_b not in ["", lat_a]) and (lat_a != ""):
            da_a = da_a.rename({lat_a: lat_b})
        return da_a

    # Adjust dimension names.
    start = copy_dim_names(start, pr)
    end = copy_dim_names(end, pr)

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")

    # Initialize the array that will contain results.
    prcptot = xr.zeros_like(start) * np.nan

    # Calculate the sum between two dates for a given year.
    def calc_sum(
        loc: str,
        year: int,
        start_doy: Union[int, xr.DataArray],
        end_doy: Union[int, xr.DataArray]
    ) -> xr.DataArray:
        sel = (pram.time.dt.year == year) & \
              (pram.time.dt.dayofyear >= start_doy) &\
              (pram.time.dt.dayofyear <= end_doy)
        if loc != "":
            sel = sel & (pram["location"] == loc)
        return xr.where(sel, pram, 0).sum(dim="time")

    # Calculate the index.
    def calc_idx(
        loc: str = ""
    ) -> xr.DataArray:
        if loc == "":
            prcptot_loc = prcptot
            start_loc = start
            end_loc = end
        else:
            prcptot_loc = prcptot[prcptot.location == loc].squeeze()
            start_loc = start[start.location == loc].squeeze()
            end_loc = end[end.location == loc].squeeze()

        end_shift_loc = None
        for y in range(len(start.time.dt.year)):
            year = int(start.time.dt.year[y])

            # Start and end dates in the same calendar year.
            if start_loc.mean() <= end_loc.mean():
                mask = xr.DataArray(start_loc[y].isnull().astype(int) +
                                    end_loc[y].isnull().astype(int) == 0).astype(int)
                mask = xr.where(mask == 0, np.nan, mask)
                sum_y = calc_sum(loc, year, start_loc[y].astype(int), end_loc[y].astype(int)) * mask

            # Start and end dates not in the same year (left shift required).
            else:
                end_shift_loc = end_loc.shift(time=-1, fill_value=np.nan) if y == 0 else end_shift_loc
                mask = xr.DataArray(start_loc[y].isnull().astype(int) +
                                    end_shift_loc[y].isnull().astype(int) == 0).astype(int)
                mask = xr.where(mask == 0, np.nan, mask)
                sum_y = (calc_sum(loc, year, start_loc[y].astype(int), 365) +
                         calc_sum(loc, year, 1, end_shift_loc[y].astype(int))) * mask

            # Update result.
            if sum_y is not None:
                prcptot_loc[y] = sum_y if loc == "" else float(sum_y[sum_y["location"] == loc])

        return prcptot_loc

    if "location" not in pram.dims:
        prcptot = calc_idx()
    else:
        locations = list(pram.location.values)
        for i in range(len(locations)):
            prcptot[prcptot.location == locations[i]] = calc_idx(locations[i])

    prcptot = prcptot // 1
    prcptot.attrs["units"] = "mm"

    return prcptot


def w_days_above(
    da_vv: xr.DataArray,
    da_dd: xr.DataArray,
    param_vv: float,
    param_dd: float,
    param_dd_tol: float,
    start_date: str,
    end_date: str
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate the number of days with a strong wind, potentially from a specified direction.

    Parameters
    ----------
    da_vv: xr.DataArray
        Wind speed (m s-1).
    da_dd: xr.DataArray
        Direction for which the wind is coming from (degrees).
    param_vv: float
        Parameter related to 'da_wind' (m s-1).
    param_dd: float
        Parameter related to 'da_dd' (degrees).
    param_dd_tol: float
        Parameter tolerance related to 'da_dd' (degrees).
    start_date: str
        Minimum day of year at or after which the season ends ("mm-dd").
    end_date: str
        Maximum day of year at or before which the season ends ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert start and end dates to day of year.
    start_doy = 1 if start_date == "" else utils.doy_str_to_doy(start_date)
    end_doy = 365 if end_date == "" else utils.doy_str_to_doy(end_date)

    # Condition #1: Subset based on day of year.
    da_vv = utils.subset_doy(da_vv, start_doy, end_doy)
    da_dd = utils.subset_doy(da_dd, start_doy, end_doy)

    # Condition #1: Wind speed.
    da_cond1 = da_vv > param_vv

    # Condition #2: Wind direction.
    if da_dd is None:
        da_cond2 = True
    else:
        da_cond2 = (da_dd - param_dd <= param_dd_tol) if param_dd is not None else True

    # Combine conditions.
    da_conds = da_cond1 & da_cond2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_w_days_above = da_conds.resample(time=c.freq_YS).sum(dim=c.dim_time)

    return da_w_days_above


def gen_per_idx(
    func_name: str,
    view_code: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate diagnostic and cycle plots.

    func_name: str
        Name of function to be called.
    view_code: Optional[str]
        View code.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Number of indices to process.
    n_idx = len(cntx.idxs.code_l)

    # Scalar mode.
    if (cntx.n_proc == 1) or (n_idx < 2):
        for i_idx in range(n_idx):
            if func_name == "stats.calc_map":
                stats.calc_map(cntx.idxs.code_l, i_idx)
            elif func_name == "stats.calc_ts":
                stats.calc_ts(view_code, cntx.idxs.code_l, i_idx)
            else:
                stats.calc_stat_tbl(cntx.idxs.code_l, i_idx)

    # Parallel processing mode.
    else:

        for i in range(math.ceil(n_idx / cntx.n_proc)):

            # Select indices to process in the current loop.
            i_first = i * cntx.n_proc
            n_proc = min(cntx.n_proc, n_idx)
            i_last = i_first + n_proc - 1
            idx_codes = cntx.idxs.code_l[i_first:(i_last + 1)]

            try:
                fu.log("Splitting work between " + str(n_proc) + " threads.", True)
                pool = multiprocessing.Pool(processes=n_proc)
                if func_name == "stats.calc_map":
                    func = functools.partial(stats.calc_map, idx_codes)
                elif func_name == "stats.calc_ts":
                    func = functools.partial(stats.calc_ts, view_code, idx_codes)
                else:
                    func = functools.partial(stats.calc_stat_tbl, idx_codes)
                pool.map(func, list(range(len(idx_codes))))
                pool.close()
                pool.join()
                fu.log("Fork ended.", True)

            except Exception as e:
                fu.log(str(e))
                pass


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    not_req = " (not required)"

    # Indices ----------------------------------------------------------------------------------------------------------

    # Calculate indices.
    fu.log("=")
    msg = "Step #6   Calculating indices"
    if cntx.opt_idx:
        fu.log(msg)
        gen()
    else:
        fu.log(msg + not_req)

    # Statistics -------------------------------------------------------------------------------------------------------

    fu.log("=")
    msg = "Step #7a  Calculating statistics (indices)"
    if cntx.opt_stat[1]:
        fu.log(msg)
        gen_per_idx("stats.calc_stat_tbl")
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #7b  Converting NetCDF to CSV files (indices)"
    if cntx.export_nc_to_csv[1] and not cntx.opt_ra:
        fu.log(msg)
        fu.log("-")
        stats.conv_nc_csv(c.cat_idx)
    else:
        fu.log(msg + not_req)

    # Plots ------------------------------------------------------------------------------------------------------------

    # Generate plots.
    fu.log("-")
    msg = "Step #8b  Generating time series (indices)"
    if cntx.opt_ts[1] and (len(cntx.opt_ts_format) > 0):
        fu.log(msg)
        gen_per_idx("stats.calc_ts", c.view_ts)
    else:
        fu.log(msg + not_req)

    # Generate maps.
    fu.log("-")
    msg = "Step #8d  Generating maps (indices)"
    if cntx.opt_ra and cntx.opt_map[1] and (len(cntx.opt_map_format) > 0):
        fu.log(msg)
        gen_per_idx("stats.calc_map")
    else:
        fu.log(msg + " (not required)")


if __name__ == "__main__":
    run()
