# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions related to climate indices.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import datetime
import functools
import glob
import math
import multiprocessing
import numpy as np
import os.path
import statistics
import utils
import xarray as xr
import xclim.indices as indices
import xclim.indices.generic as indices_gen
import warnings
from typing import Tuple, List
from xclim.indices import run_length as rl
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to, declare_units, rate2amount, to_agg_units


def generate(
    idx_code: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a time series.

    Parameters:
    idx_code : str
        Index code (contains 'idx_name' and an identifier).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Obtain complete index name, index name and index parameters.
    idx_name   = cfg.extract_idx(idx_code)
    idx_params = cfg.idx_params[cfg.idx_codes.index(idx_code)]

    # Emission scenarios.
    rcps = [cfg.rcp_ref] + cfg.rcps

    # Data preparation -------------------------------------------------------------------------------------------------

    utils.log("Selecting variables and indices.", True)

    # ==================================================================================================================
    # TODO.CUSTOMIZATION.INDEX.BEGIN
    # Select variables and indices.
    # ==================================================================================================================

    # Select variables.
    varidx_name_l = []

    # Temperature.
    if idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropical_nights, cfg.idx_tng_months_below,
                    cfg.idx_heat_wave_max_length, cfg.idx_heat_wave_total_length, cfg.idx_tgg, cfg.idx_etr,
                    cfg.idx_tn_days_below]:
        varidx_name_l.append(cfg.var_cordex_tasmin)

    if idx_name in [cfg.idx_tx90p, cfg.idx_tx_days_above, cfg.idx_hot_spell_frequency, cfg.idx_hot_spell_max_length,
                    cfg.idx_txg, cfg.idx_txx, cfg.idx_wsdi, cfg.idx_heat_wave_max_length,
                    cfg.idx_heat_wave_total_length, cfg.idx_tgg, cfg.idx_etr]:
        varidx_name_l.append(cfg.var_cordex_tasmax)

    # Precipitation.
    if idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_cwd, cfg.idx_cdd, cfg.idx_sdii, cfg.idx_prcptot,
                    cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wet_days, cfg.idx_dry_days,
                    cfg.idx_rain_season_start, cfg.idx_rain_season_end, cfg.idx_rain_season_prcptot,
                    cfg.idx_dry_spell_total_length, cfg.idx_rain_season]:
        varidx_name_l.append(cfg.var_cordex_pr)

        if idx_name in [cfg.idx_rain_season_end, cfg.idx_rain_season]:
            if cfg.var_cordex_evspsblpot in cfg.variables_cordex:
                varidx_name_l.append(cfg.var_cordex_evspsblpot)
            elif cfg.var_cordex_evspsbl in cfg.variables_cordex:
                varidx_name_l.append(cfg.var_cordex_evspsbl)
            else:
                varidx_name_l.append("nan")

        if idx_name in [cfg.idx_rain_season_end, cfg.idx_rain_season_length, cfg.idx_rain_season_prcptot]:
            varidx_name_l.append(idx_code.replace(idx_name, cfg.idx_rain_season_start))

        if idx_name == cfg.idx_rain_season:
            varidx_name_l.append("nan")

        if idx_name in [cfg.idx_rain_season_end, cfg.idx_rain_season]:
            varidx_name_l.append(str(idx_params[len(idx_params) - 1]))

        if idx_name in [cfg.idx_rain_season_length, cfg.idx_rain_season_prcptot]:
            varidx_name_l.append(idx_code.replace(idx_name, cfg.idx_rain_season_end))

    # Temperature-precipitation.
    if idx_name == cfg.idx_drought_code:
        varidx_name_l.append(cfg.var_cordex_tas)
        varidx_name_l.append(cfg.var_cordex_pr)

    # Wind.
    if idx_name == cfg.idx_wg_days_above:
        varidx_name_l.append(cfg.var_cordex_uas)
        varidx_name_l.append(cfg.var_cordex_vas)

    elif idx_name == cfg.idx_wx_days_above:
        varidx_name_l.append(cfg.var_cordex_sfcwindmax)

    # ==================================================================================================================
    # TODO.CUSTOMIZATION.INDEX.END
    # ==================================================================================================================

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Verify if this variable or index is available for the current station.
        utils.log("Verifying data availability (based on directories).", True)
        varidx_name_l_avail = True
        for varidx_name in varidx_name_l:
            if (varidx_name != "nan") and \
               (((cfg.extract_idx(varidx_name) in cfg.variables_cordex) and
                 not os.path.isdir(cfg.get_d_scen(stn, cfg.cat_qqmap, varidx_name))) or
                ((cfg.extract_idx(varidx_name) not in cfg.variables_cordex) and
                 not os.path.isdir(cfg.get_d_idx(stn, varidx_name)))):
                varidx_name_l_avail = False
                break
        if not varidx_name_l_avail:
            continue

        # Create mask.
        da_mask = None
        if stn == cfg.obs_src_era5_land:
            da_mask = utils.create_mask()

        # Loop through emissions scenarios.
        for rcp in rcps:

            utils.log("Processing: " + idx_code + ", " + stn + ", " + cfg.get_rcp_desc(rcp) + "", True)

            # List simulation files for the first variable. As soon as there is no file for one variable, the analysis
            # for the current RCP needs to abort.
            utils.log("Collecting simulation files.", True)
            if rcp == cfg.rcp_ref:
                if cfg.extract_idx(varidx_name_l[0]) in cfg.variables_cordex:
                    p_sim = cfg.get_d_stn(varidx_name_l[0]) +\
                            cfg.extract_idx(varidx_name_l[0]) + "_" + stn + cfg.f_ext_nc
                else:
                    p_sim = cfg.get_d_idx(cfg.obs_src, varidx_name_l[0]) +\
                            cfg.extract_idx(varidx_name_l[0]) + "_ref" + cfg.f_ext_nc
                if os.path.exists(p_sim) and (type(p_sim) is str):
                    p_sim = [p_sim]
            else:
                if cfg.extract_idx(varidx_name_l[0]) in cfg.variables_cordex:
                    d = cfg.get_d_scen(stn, cfg.cat_qqmap, varidx_name_l[0])
                else:
                    d = cfg.get_d_idx(stn, varidx_name_l[0])
                p_sim = glob.glob(d + "*_" + rcp + cfg.f_ext_nc)
            if not p_sim:
                continue

            # Remove simulations that are included in the exceptions lists.
            p_sim_filter = []
            for p in p_sim:
                found = False
                # List of simulation exceptions.
                for e in cfg.sim_excepts:
                    if e.replace(cfg.f_ext_nc, "") in p:
                        found = True
                        break
                # List of variable-simulation exceptions.
                for e in cfg.var_sim_excepts:
                    if e.replace(cfg.f_ext_nc, "") in p:
                        found = True
                        break
                # Add simulation.
                if not found:
                    p_sim_filter.append(p)
            p_sim = p_sim_filter

            # Ensure that simulations are available for other variables than the first one.
            utils.log("Verifying data availability (based on NetCDF files).", True)
            if len(varidx_name_l) > 1:
                p_sim_fix = []
                for p_sim_i in p_sim:
                    missing = False
                    for varidx_name_j in varidx_name_l[1:]:
                        if varidx_name_j != "nan":
                            p_sim_j = cfg.get_equivalent_idx_path(p_sim_i, varidx_name_l[0], varidx_name_j, stn, rcp)
                            if not os.path.exists(p_sim_j):
                                missing = True
                                break
                    if not missing:
                        p_sim_fix.append(p_sim_i)
                p_sim = p_sim_fix

            # Calculation ---------------------------------------------------------------------------------------------

            utils.log("Calculating climate indices", True)

            n_sim = len(p_sim)
            d_idx = cfg.get_d_idx(stn, idx_name)

            # Scalar mode.
            if cfg.n_proc == 1:
                for i_sim in range(n_sim):
                    generate_single(idx_code, idx_params, varidx_name_l, p_sim, stn, rcp, da_mask, i_sim)

            # Parallel processing mode.
            else:

                # Loop until all simulations have been processed.
                while True:

                    # Calculate the number of processed files (before generation).
                    # This verification is based on the index NetCDF file.
                    n_sim_proc_before = len(list(glob.glob(d_idx + "*" + cfg.f_ext_nc)))

                    # Scalar processing mode.
                    scalar_required = False
                    if idx_name == cfg.idx_prcptot:
                        scalar_required = not str(idx_params[0]).isdigit()
                    if (cfg.n_proc == 1) or scalar_required:
                        for i_sim in range(n_sim):
                            generate_single(idx_code, idx_params, varidx_name_l, p_sim, stn, rcp, da_mask, i_sim)

                    # Parallel processing mode.
                    else:

                        try:
                            utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=min(cfg.n_proc, n_sim))
                            func = functools.partial(generate_single, idx_code, idx_params, varidx_name_l, p_sim,
                                                     stn, rcp, da_mask)
                            pool.map(func, list(range(n_sim)))
                            pool.close()
                            pool.join()
                            utils.log("Fork ended.", True)
                        except Exception as e:
                            utils.log(str(e))
                            pass

                    # Calculate the number of processed files (after generation).
                    n_sim_proc_after = len(list(glob.glob(d_idx + "*" + cfg.f_ext_nc)))

                    # If no simulation has been processed during a loop iteration, this means that the work is done.
                    if (cfg.n_proc == 1) or (n_sim_proc_before == n_sim_proc_after):
                        break


def generate_single(
    idx_code: str,
    idx_params,
    varidx_name_l: [str],
    p_sim: [str],
    stn: str,
    rcp: str,
    da_mask: xr.DataArray,
    i_sim: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts observations to NetCDF.

    Parameters
    ----------
    idx_code : str
        Climate index code.
    idx_params :
        Climate index parameters.
    varidx_name_l : [str]
        List of climate variables or indices.
    p_sim : [str]
        List of simulation files.
    stn : str
        Station name.
    rcp : str
        RCP emission scenario.
    da_mask : xr.DataArray
        Mask.
    i_sim : int
        Rank of simulation in 'p_sim'.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Name of NetCDF file to generate.
    idx_name = cfg.extract_idx(idx_code)
    if rcp == cfg.rcp_ref:
        p_idx = cfg.get_d_idx(stn, idx_code) + idx_name + "_ref" + cfg.f_ext_nc
    else:
        p_idx = cfg.get_d_scen(stn, cfg.cat_idx, idx_code) +\
                os.path.basename(p_sim[i_sim]).replace(cfg.extract_idx(varidx_name_l[0]), idx_name)

    # Exit loop if the file already exists (simulations files only; not reference file).
    if (rcp != cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite):
        if cfg.n_proc > 1:
            utils.log("Work done!", True)
        return

    # Load datasets (one per variable or index).
    ds_varidx_l: List[xr.Dataset] = []
    for i_varidx in range(0, len(varidx_name_l)):
        varidx_name = varidx_name_l[i_varidx]

        try:
            # Open dataset.
            p_sim_j =\
                cfg.get_equivalent_idx_path(p_sim[i_sim], varidx_name_l[0], cfg.get_idx_group(varidx_name), stn, rcp)
            ds = utils.open_netcdf(p_sim_j)

            # Remove February 29th and select reference period.
            if (rcp == cfg.rcp_ref) and (varidx_name in cfg.variables_cordex):
                ds = utils.remove_feb29(ds)
                ds = utils.sel_period(ds, cfg.per_ref)

            # Adjust temperature units.
            if cfg.extract_idx(varidx_name) in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                if ds[varidx_name].attrs[cfg.attrs_units] == cfg.unit_K:
                    ds[varidx_name] = ds[varidx_name] - cfg.d_KC
                elif rcp == cfg.rcp_ref:
                    ds[varidx_name][cfg.attrs_units] = cfg.unit_C
                ds[varidx_name].attrs[cfg.attrs_units] = cfg.unit_C

        except:
            ds = None

        # Add dataset.
        ds_varidx_l.append(ds)

    # ======================================================================================================
    # TODO.CUSTOMIZATION.INDEX.BEGIN
    # Calculate indices.
    # ======================================================================================================

    # Calculate the 90th percentile of tasmax for the reference period.
    da_tx90p = None
    if (idx_name == cfg.idx_wsdi) and (rcp == cfg.rcp_ref):
        da_tx90p = xr.DataArray(percentile_doy(ds_varidx_l[0][cfg.var_cordex_tasmax], per=0.9))

    # Merge threshold value and unit, if required. Ex: "0.0 C" for temperature.
    idx_params_str = []
    for i in range(len(idx_params)):
        idx_param = idx_params[i]

        # Convert thresholds (percentile to absolute value) ------------------------------------------------

        if (idx_name == cfg.idx_tx90p) or ((idx_name == cfg.idx_wsdi) and (i == 0)):
            idx_param = "90p"
        if (idx_name in [cfg.idx_tx_days_above, cfg.idx_tng_months_below, cfg.idx_tx90p, cfg.idx_prcptot,
                         cfg.idx_tropical_nights]) or\
           ((idx_name in [cfg.idx_hot_spell_frequency, cfg.idx_hot_spell_max_length, cfg.idx_wsdi]) and (i == 0)) or \
           ((idx_name in [cfg.idx_heat_wave_max_length, cfg.idx_heat_wave_total_length]) and (i <= 1)) or \
           ((idx_name in [cfg.idx_wg_days_above, cfg.idx_wx_days_above]) and (i == 0)):

            if "p" in str(idx_param):
                idx_param = float(idx_param.replace("p", "")) / 100.0
                if rcp == cfg.rcp_ref:
                    # Calculate percentile.
                    if (idx_name in [cfg.idx_tx90p, cfg.idx_hot_spell_frequency, cfg.idx_hot_spell_max_length,
                                     cfg.idx_wsdi]) or (i == 1):
                        idx_param = ds_varidx_l[i][cfg.var_cordex_tasmax].quantile(idx_param).values.ravel()[0]
                    elif idx_name in [cfg.idx_heat_wave_max_length, cfg.idx_heat_wave_total_length]:
                        idx_param = ds_varidx_l[i][cfg.var_cordex_tasmin].quantile(idx_param).values.ravel()[0]
                    elif idx_name == cfg.idx_prcptot:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            da_i = ds_varidx_l[i][cfg.var_cordex_pr].resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)
                        dims = utils.get_coord_names(ds_varidx_l[i])
                        idx_param = da_i.mean(dim=dims).quantile(idx_param).values.ravel()[0] * cfg.spd
                    elif idx_name in [cfg.idx_wg_days_above, cfg.idx_wx_days_above]:
                        if idx_name == cfg.idx_wg_days_above:
                            da_uas = ds_varidx_l[0][cfg.var_cordex_uas]
                            da_vas = ds_varidx_l[1][cfg.var_cordex_vas]
                            da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas)
                        else:
                            da_vv = ds_varidx_l[0][cfg.var_cordex_sfcwindmax]
                        idx_param = da_vv.quantile(idx_param).values.ravel()[0]
                    # Round value and save it.
                    idx_param = float(round(idx_param, 2))
                    cfg.idx_params[cfg.idx_names.index(idx_name)][i] = idx_param
                else:
                    idx_param = cfg.idx_params[cfg.idx_names.index(idx_name)][i]

        # Combine threshold and unit -----------------------------------------------------------------------

        if (idx_name in [cfg.idx_tx90p, cfg.idx_tropical_nights]) or\
           ((idx_name in [cfg.idx_hot_spell_frequency, cfg.idx_hot_spell_max_length, cfg.idx_wsdi,
                          cfg.idx_tn_days_below, cfg.idx_tx_days_above]) and (i == 0)) or \
           ((idx_name in [cfg.idx_heat_wave_max_length, cfg.idx_heat_wave_total_length]) and (i <= 1)):
            idx_ref = str(idx_param) + " " + cfg.unit_C
            idx_fut = str(idx_param + cfg.d_KC) + " " + cfg.unit_K
            idx_params_str.append(idx_ref if (rcp == cfg.rcp_ref) else idx_fut)

        elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                          cfg.idx_wet_days, cfg.idx_dry_days, cfg.idx_sdii]:
            idx_params_str.append(str(idx_param) + " mm/day")

        elif (idx_name in [cfg.idx_wg_days_above, cfg.idx_wx_days_above]) and (i == 1):
            idx_params_str.append(str(idx_param) + " " + cfg.unit_m_s)

        elif not ((idx_name in [cfg.idx_wg_days_above, cfg.idx_wx_days_above]) and (i == 4)):
            idx_params_str.append(str(idx_param))

    # Exit loop if the file already exists (reference file only).
    if not ((rcp == cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite)):

        # Will hold data arrays and units.
        da_idx_l    = []
        idx_units_l = []
        idx_name_l  = []

        # Temperature --------------------------------------------------------------------------------------------------

        if idx_name in [cfg.idx_tx_days_above, cfg.idx_tx90p]:

            # Collect required datasets and parameters.
            da_tasmax = ds_varidx_l[0][cfg.var_cordex_tasmax]
            thresh = idx_params_str[0]
            start_date = end_date = ""
            if len(idx_params_str) > 1:
                start_date = str(idx_params_str[1]).replace("nan", "")
                end_date = str(idx_params_str[2]).replace("nan", "")

            # Calculate index.
            da_idx = tx_days_above(da_tasmax, thresh, start_date, end_date)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        if idx_name == cfg.idx_tn_days_below:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]
            thresh = idx_params_str[0]
            start_date = end_date = ""
            if len(idx_params_str) > 1:
                start_date = str(idx_params_str[1]).replace("nan", "")
                end_date = str(idx_params_str[2]).replace("nan", "")

            # Calculate index.
            da_idx = tn_days_below(da_tasmin, thresh, start_date, end_date)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_tng_months_below:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]
            param_tasmin = float(idx_params_str[0])
            if da_tasmin.attrs[cfg.attrs_units] != cfg.unit_C:
                param_tasmin += cfg.d_KC

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.tn_mean(da_tasmin, freq=cfg.freq_MS))
                da_idx = xr.DataArray(indices_gen.threshold_count(da_idx, "<", param_tasmin, cfg.freq_YS))
            da_idx = da_idx.astype(float)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name in [cfg.idx_hot_spell_frequency, cfg.idx_hot_spell_max_length, cfg.idx_wsdi]:

            # Collect required datasets and parameters.
            da_tasmax = ds_varidx_l[0][cfg.var_cordex_tasmax]
            param_tasmax = idx_params_str[0]
            param_ndays = int(float(idx_params_str[1]))

            # Calculate index.
            if idx_name == cfg.idx_hot_spell_frequency:
                da_idx = xr.DataArray(
                    indices.hot_spell_frequency(da_tasmax, param_tasmax, param_ndays).values)
            elif idx_name == cfg.idx_hot_spell_max_length:
                da_idx = xr.DataArray(
                    indices.hot_spell_max_length(da_tasmax, param_tasmax, param_ndays).values)
            else:
                da_idx = xr.DataArray(
                    indices.warm_spell_duration_index(da_tasmax, da_tx90p, param_ndays).values)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name in [cfg.idx_heat_wave_max_length, cfg.idx_heat_wave_total_length]:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]
            da_tasmax = ds_varidx_l[1][cfg.var_cordex_tasmax]
            param_tasmin = idx_params_str[0]
            param_tasmax = idx_params_str[1]
            window = int(float(idx_params_str[2]))

            # Calculate index.
            if idx_name == cfg.idx_heat_wave_max_length:
                da_idx = xr.DataArray(
                    heat_wave_max_length(da_tasmin, da_tasmax, param_tasmin, param_tasmax, window).values)
            else:
                da_idx = xr.DataArray(heat_wave_total_length(
                    da_tasmin, da_tasmax, param_tasmin, param_tasmax, window).values)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name in [cfg.idx_txg, cfg.idx_txx]:

            # Collect required datasets and parameters.
            da_tasmax = ds_varidx_l[0][cfg.var_cordex_tasmax]

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx_name == cfg.idx_txg:
                    da_idx = xr.DataArray(indices.tx_mean(da_tasmax))
                else:
                    da_idx = xr.DataArray(indices.tx_max(da_tasmax))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_C)

        elif idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropical_nights]:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]

            # Calculate index.
            if idx_name in [cfg.idx_tnx, cfg.idx_tng]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    if idx_name == cfg.idx_tnx:
                        da_idx = xr.DataArray(indices.tn_max(da_tasmin))
                        idx_units = cfg.unit_C
                    else:
                        da_idx = xr.DataArray(indices.tn_mean(da_tasmin))
                        idx_units = cfg.unit_C
            else:
                param_tasmin = idx_params_str[0]
                da_idx = xr.DataArray(indices.tropical_nights(da_tasmin, param_tasmin))
                idx_units = cfg.unit_1

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(idx_units)

        elif idx_name in [cfg.idx_tgg, cfg.idx_etr]:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]
            da_tasmax = ds_varidx_l[1][cfg.var_cordex_tasmax]

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx_name == cfg.idx_tgg:
                    da_idx = xr.DataArray(indices.tg_mean(indices.tas(da_tasmin, da_tasmax)))
                else:
                    da_idx = xr.DataArray(indices.tx_max(da_tasmax) - indices.tn_min(da_tasmin))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_C)

        elif idx_name == cfg.idx_drought_code:

            # Collect required datasets and parameters.
            da_tas = ds_varidx_l[0][cfg.var_cordex_tas]
            da_pr  = ds_varidx_l[1][cfg.var_cordex_pr]
            da_lon, da_lat = utils.get_coordinates(ds_varidx_l[0])

            # Calculate index
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.drought_code(da_tas, da_pr, da_lat)).resample(time=cfg.freq_YS).mean()

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        # Precipitation ------------------------------------------------------------------------------------------------

        elif idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_prcptot]:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]

            # Calculate index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx_name == cfg.idx_rx1day:
                    da_idx = xr.DataArray(indices.max_1day_precipitation_amount(da_pr, cfg.freq_YS))
                elif idx_name == cfg.idx_rx5day:
                    da_idx = xr.DataArray(indices.max_n_day_precipitation_amount(da_pr, 5, cfg.freq_YS))
                else:
                    start_date = end_date = ""
                    if len(idx_params_str) == 3:
                        start_date = str(idx_params_str[1]).replace("nan", "")
                        end_date = str(idx_params_str[2]).replace("nan", "")
                    da_idx = xr.DataArray(precip_accumulation(da_pr, start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(da_idx.attrs[cfg.attrs_units])

        elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                          cfg.idx_wet_days, cfg.idx_dry_days, cfg.idx_sdii]:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            param_pr = idx_params_str[0]

            # Calculate index.
            if idx_name in cfg.idx_cwd:
                da_idx = xr.DataArray(
                    indices.maximum_consecutive_wet_days(da_pr, param_pr, cfg.freq_YS))
            elif idx_name in cfg.idx_cdd:
                da_idx = xr.DataArray(
                    indices.maximum_consecutive_dry_days(da_pr, param_pr, cfg.freq_YS))
            elif idx_name in [cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wet_days]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    da_idx = xr.DataArray(indices.wetdays(da_pr, param_pr, cfg.freq_YS))
            elif idx_name == cfg.idx_dry_days:
                da_idx = xr.DataArray(indices.dry_days(da_pr, param_pr, cfg.freq_YS))
            else:
                da_idx = xr.DataArray(indices.daily_pr_intensity(da_pr, param_pr))
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rain_season_start:

            # Collect required datasets and parameters.
            da_pr      = ds_varidx_l[0][cfg.var_cordex_pr]
            thresh_wet = idx_params_str[0] + " mm"
            window_wet = int(idx_params_str[1])
            thresh_dry = idx_params_str[2] + " mm"
            dry_days   = int(idx_params_str[3])
            window_dry = int(idx_params_str[4])
            start_date = str(idx_params_str[5]).replace("nan", "")
            end_date   = str(idx_params_str[6]).replace("nan", "")

            # Calculate index.
            da_idx =\
                xr.DataArray(rain_season_start(da_pr, thresh_wet, window_wet, thresh_dry, dry_days, window_dry,
                                               start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rain_season_end:

            # Collect required datasets and parameters.
            da_pr  = ds_varidx_l[0][cfg.var_cordex_pr]
            da_etp = None
            if ds_varidx_l[1] is not None:
                if cfg.var_cordex_evspsblpot in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsblpot]
                elif cfg.var_cordex_evspsbl in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsbl]
            da_start = None
            if ds_varidx_l[2] is not None:
                da_start = ds_varidx_l[2][cfg.idx_rain_season_start]
            da_start_next = None
            if ds_varidx_l[3] is not None:
                da_start_next = ds_varidx_l[3][cfg.idx_rain_season_start]
            op         = idx_params_str[0]
            thresh     = idx_params_str[1] + " mm"
            window     = -1 if str(idx_params_str[2]) == "nan" else int(idx_params_str[2])
            etp_rate   = ("0" if str(idx_params_str[3]) == "nan" else idx_params_str[3]) + " mm"
            start_date = str(idx_params_str[4]).replace("nan", "")
            end_date   = str(idx_params_str[5]).replace("nan", "")

            # Calculate index.
            da_idx = xr.DataArray(rain_season_end(da_pr, da_etp, da_start, da_start_next, op, thresh, window, etp_rate,
                                                  start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rain_season_length:

            # Collect required datasets and parameters.
            da_start = ds_varidx_l[0][cfg.idx_rain_season_start]
            da_end   = ds_varidx_l[1][cfg.idx_rain_season_end]

            # Calculate index.
            da_idx = rain_season_length(da_start, da_end)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rain_season_prcptot:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            da_start = ds_varidx_l[1][cfg.idx_rain_season_start]
            da_end = ds_varidx_l[2][cfg.idx_rain_season_end]

            # Calculate index.
            da_idx = xr.DataArray(rain_season_prcptot(da_pr, da_start, da_end))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.get_unit(idx_name))

        elif idx_name == cfg.idx_rain_season:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            # Rain start:
            s_thresh_wet = idx_params_str[0] + " mm"
            s_window_wet = int(idx_params_str[1])
            s_thresh_dry = idx_params_str[2] + " mm"
            s_dry_days   = int(idx_params_str[3])
            s_window_dry = int(idx_params_str[4])
            s_start_date = str(idx_params_str[5]).replace("nan", "")
            s_end_date   = str(idx_params_str[6]).replace("nan", "")
            # Rain end:
            da_etp = None
            if ds_varidx_l[1] is not None:
                if cfg.var_cordex_evspsblpot in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsblpot]
                elif cfg.var_cordex_evspsbl in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsbl]
            da_start_next = None
            if ds_varidx_l[3] is not None:
                da_start_next = ds_varidx_l[3][cfg.idx_rain_season_start]
            e_op         = idx_params_str[7]
            e_thresh     = idx_params_str[8] + " mm"
            e_window     = -1 if str(idx_params_str[9]) == "nan" else int(idx_params_str[9])
            e_etp_rate   = ("0" if str(idx_params_str[10]) == "nan" else idx_params_str[10]) + " mm"
            e_start_date = str(idx_params_str[11]).replace("nan", "")
            e_end_date   = str(idx_params_str[12]).replace("nan", "")

            # Calculate indices.
            da_start, da_end, da_length, da_prcptot =\
                rain_season(da_pr, da_etp, da_start_next, s_thresh_wet, s_window_wet, s_thresh_dry, s_dry_days,
                            s_window_dry, s_start_date, s_end_date, e_op, e_thresh, e_window, e_etp_rate,
                            e_start_date, e_end_date)

            # Add to list.
            da_idx_l = [da_start, da_end, da_length, da_prcptot]
            idx_units_l = [cfg.unit_1, cfg.unit_1, cfg.unit_1, cfg.get_unit(cfg.idx_rain_season_prcptot)]
            idx_name_l = [cfg.idx_rain_season_start, cfg.idx_rain_season_end, cfg.idx_rain_season_length,
                          cfg.idx_rain_season_prcptot]

        elif idx_name == cfg.idx_dry_spell_total_length:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            thresh = idx_params_str[0] + " mm"
            window = int(idx_params_str[1])
            op = idx_params_str[2]
            start_date = end_date = ""
            if len(idx_params_str) == 5:
                start_date = str(idx_params_str[3])
                end_date = str(idx_params_str[4])

            # Calculate index.
            da_idx = xr.DataArray(dry_spell_total_length(da_pr, thresh, window, op, start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        # Wind ---------------------------------------------------------------------------------------------------------

        elif idx_name in [cfg.idx_wg_days_above, cfg.idx_wx_days_above]:

            # Collect required datasets and parameters.
            param_vv     = float(idx_params_str[0])
            param_vv_neg = idx_params_str[1]
            param_dd     = float(idx_params_str[2])
            param_dd_tol = float(idx_params_str[3])
            start_date = end_date = ""
            if len(idx_params_str) == 6:
                start_date = str(idx_params_str[4]).replace("nan", "")
                end_date = str(idx_params_str[5]).replace("nan", "")
            if idx_name == cfg.idx_wg_days_above:
                da_uas = ds_varidx_l[0][cfg.var_cordex_uas]
                da_vas = ds_varidx_l[1][cfg.var_cordex_vas]
                da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas, param_vv_neg)
            else:
                da_vv = ds_varidx_l[0][cfg.var_cordex_sfcwindmax]
                da_dd = None

            # Calculate index.
            da_idx =\
                xr.DataArray(w_days_above(da_vv, da_dd, param_vv, param_dd, param_dd_tol, start_date, end_date))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        # ==============================================================================================================
        # TODO.CUSTOMIZATION.INDEX.END
        # ==============================================================================================================

        if len(idx_name_l) == 0:
            idx_name_l = [idx_name]

        # Loop through data arrays.
        ds_idx = None
        for i in range(len(da_idx_l)):
            da_idx = da_idx_l[i]
            idx_units = idx_units_l[i]

            # Assign units.
            da_idx.attrs[cfg.attrs_units] = idx_units

            # Convert to float. This is required to ensure that 'nan' values are not transformed into integers.
            da_idx = da_idx.astype(float)

            # Rename dimensions.
            da_idx = utils.rename_dimensions(da_idx)

            # Interpolate (to remove nan values).
            if np.isnan(da_idx).astype(int).max() > 0:
                da_idx = utils.interpolate_na_fix(da_idx)

            # Reorder dimensions to fit input data.
            utils.reorder_dims(da_idx, ds_varidx_l[0])

            # Apply mask.
            if da_mask is not None:
                da_idx = utils.apply_mask(da_idx, da_mask)

            # Create dataset.
            if i == 0:
                ds_idx = da_idx.to_dataset(name=idx_name_l[i])
                ds_idx.attrs[cfg.attrs_units] = idx_units
                ds_idx.attrs[cfg.attrs_sname] = idx_name
                ds_idx.attrs[cfg.attrs_lname] = idx_name
                ds_idx = utils.copy_coordinates(ds_varidx_l[0], ds_idx)

            # Add data array.
            ds_idx[idx_name_l[i]] = utils.copy_coordinates(ds_varidx_l[0][cfg.extract_idx(varidx_name_l[0])], da_idx)

        # Adjust calendar.
        ds_idx = ds_idx.squeeze()
        year_1 = cfg.per_fut[0]
        year_n = cfg.per_fut[1]
        if rcp == cfg.rcp_ref:
            year_1 = max(cfg.per_ref[0], int(str(ds_varidx_l[0][cfg.dim_time][0].values)[0:4]))
            year_n = min(cfg.per_ref[1], int(str(ds_varidx_l[0][cfg.dim_time]
                                                 [len(ds_varidx_l[0][cfg.dim_time]) - 1].values)[0:4]))
        ds_idx[cfg.dim_time] = utils.reset_calendar(ds_idx, year_1, year_n, cfg.freq_YS)

        # Save result to NetCDF file.
        desc = cfg.sep + idx_name + cfg.sep + os.path.basename(p_idx)
        utils.save_netcdf(ds_idx, p_idx, desc=desc)

    # Convert percentile threshold values for climate indices. This is sometimes required in time series.
    if (rcp == cfg.rcp_ref) and (idx_name == cfg.idx_prcptot):
        ds_idx = utils.open_netcdf(p_idx)
        da_idx = ds_idx.mean(dim=[cfg.dim_longitude, cfg.dim_latitude])[idx_name]
        param_pr = idx_params_str[0]
        if "p" in str(param_pr):
            param_pr = float(param_pr.replace("p", "")) / 100.0
            cfg.idx_params[cfg.idx_codes.index(idx_code)][0] = \
                float(round(da_idx.quantile(param_pr).values.ravel()[0], 2))

    if cfg.n_proc > 1:
        utils.log("Work done!", True)


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
    da_pr : xr.DataArray
        Precipitation data.
    start_date: str
        First day of year where season can start ("mm-dd").
    end_date: str
        Last day of year where season can start ("mm-dd")
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert start and end dates to doy.
    start_doy = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday

    # Subset based on day of year.
    da_pr = utils.subset_doy(da_pr, start_doy, end_doy)

    # Calculate index.
    da_idx = xr.DataArray(indices.precip_accumulation(da_pr, freq=cfg.freq_YS))

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
    da_tasmax : xr.DataArray
        Maximum temperature data.
    thresh : str
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
    da_tasmin : xr.DataArray
        Minimum temperature data.
    thresh : str
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
    freq: str = cfg.freq_YS
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Same as equivalent non-working function in xclim.indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    param_tasmin = convert_units_to(param_tasmin, tasmin)
    param_tasmax = convert_units_to(param_tasmax, tasmax)

    # Adjust calendars.
    if tasmin.time.dtype != tasmax.time.dtype:
        tasmin[cfg.dim_time] = tasmin[cfg.dim_time].astype("datetime64[ns]")
        tasmax[cfg.dim_time] = tasmax[cfg.dim_time].astype("datetime64[ns]")

    # Call the xclim function if time dimension is the same.
    n_tasmin = len(tasmin[cfg.dim_time])
    n_tasmax = len(tasmax[cfg.dim_time])

    if n_tasmin == n_tasmax:
        return indices.heat_wave_max_length(tasmin, tasmax, param_tasmin, param_tasmax, window, freq)

    # Calculate manually.
    else:

        cond = tasmin
        if n_tasmax > n_tasmin:
            cond = tasmax
        for t in cond.time:
            if (t.values in tasmin[cfg.dim_time]) and (t.values in tasmax[cfg.dim_time]):
                cond[cond[cfg.dim_time] == t] =\
                    (tasmin[tasmin[cfg.dim_time] == t] > param_tasmin) &\
                    (tasmax[tasmax[cfg.dim_time] == t] > param_tasmax)
            else:
                cond[cond[cfg.dim_time] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)
        max_l = group.map(rl.longest_run, dim=cfg.dim_time)

        return max_l.where(max_l >= window, 0)


def heat_wave_total_length(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    param_tasmin: str = "22.0 degC",
    param_tasmax: str = "30 degC",
    window: int = 3,
    freq: str = cfg.freq_YS
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Same as equivalent non-working function in xclim.indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    param_tasmin = convert_units_to(param_tasmin, tasmin)
    param_tasmax = convert_units_to(param_tasmax, tasmax)

    # Adjust calendars.
    if tasmin.time.dtype != tasmax.time.dtype:
        tasmin[cfg.dim_time] = tasmin[cfg.dim_time].astype("datetime64[ns]")
        tasmax[cfg.dim_time] = tasmax[cfg.dim_time].astype("datetime64[ns]")

    # Call the xclim function if time dimension is the same.
    n_tasmin = len(tasmin[cfg.dim_time])
    n_tasmax = len(tasmax[cfg.dim_time])

    if n_tasmin == n_tasmax:
        return indices.heat_wave_total_length(tasmin, tasmax, param_tasmin, param_tasmax, window, freq)

    # Calculate manually.
    else:

        cond = tasmin
        if n_tasmax > n_tasmin:
            cond = tasmax
        for t in cond.time:
            if (t.values in tasmin[cfg.dim_time]) and (t.values in tasmax[cfg.dim_time]):
                cond[cond[cfg.dim_time] == t] =\
                    (tasmin[tasmin[cfg.dim_time] == t] > param_tasmin) &\
                    (tasmax[tasmax[cfg.dim_time] == t] > param_tasmax)
            else:
                cond[cond[cfg.dim_time] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)

        return group.map(rl.windowed_run_count, args=(window,), dim=cfg.dim_time)


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
    pr : xr.DataArray:
        Daily precipitation.
    thresh : str
        Accumulated precipitation value under which a period is considered dry.
    window : int
        Number of days where the maximum or accumulated precipitation is under threshold.
    op : {"max", "sum"}
        Reduce operation..
    freq : str
      Resampling frequency.
    start_date : str
        First day of year to consider ("mm-dd").
    end_date : str
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


@declare_units(
    pr="[precipitation]",
    etp="[evapotranspiration]",
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
    pr : xr.DataArray
        Daily precipitation.
    etp : xr.DataArray
        Daily evapotranspiration.
    start_next : xr.DataArray
        First day of the next rain season.
    s_thresh_wet : str
        Accumulated precipitation threshold associated with {s_window_wet}.
    s_window_wet: int
        Number of days where accumulated precipitation is above {s_thresh_wet}.
    s_thresh_dry: str
        Daily precipitation threshold associated with {s_window_dry].
    s_dry_days: int
        Maximum number of dry days in {s_window_tot}.
    s_window_dry: int
        Number of days, after {s_window_wet}, during which daily precipitation is not greater than or equal to
        {s_thresh_dry} for {s_dry_days} consecutive days.
    s_start_date: str
        First day of year where season can start ("mm-dd").
    s_end_date: str
        Last day of year where season can start ("mm-dd").
    e_op : str
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
    e_thresh : str
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
    pr : xr.DataArray
        Precipitation data.
    thresh_wet : str
        Accumulated precipitation threshold associated with {window_wet}.
    window_wet: int
        Number of days where accumulated precipitation is above {thresh_wet}.
    thresh_dry: str
        Daily precipitation threshold associated with {window_dry}.
    dry_days: int
        Maximum number of dry days in {window_dry}.
    window_dry: int
        Number of days, after {window_wet}, during which daily precipitation is not greater than or equal to
        {thresh_dry} for {dry_days} consecutive days.
    start_date: str
        First day of year where season can start ("mm-dd").
    end_date: str
        Last day of year where season can start ("mm-dd").

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
    This correspond to definition no. 2, which is a simplification of an index mentioned in:
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

    # Obtain the first day of each year where conditions apply.
    start = (wet & no_dry_seq & doy).resample(time="YS").\
        map(rl.first_run, window=1, dim="time", coord="dayofyear")
    start = xr.where((start < 1) | (start > 365), np.nan, start)
    start.attrs["units"] = "1"

    return start


@declare_units(
    pr="[precipitation]",
    etp="[evapotranspiration]",
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
    pr : xr.DataArray
        Daily precipitation.
    etp : xr.DataArray
        Daily evapotranspiration.
    start : xr.DataArray
        First day of the current rain season.
    start_next : xr.DataArray
        First day of the next rain season.
    op : str
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
    thresh : str
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

            # Obtain the first day of each year where conditions apply.
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

        # Obtain the first day of each year where conditions apply.
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
        n_days = 0
        for t in range(len(end_loc.time)):
            n_days_t = int(xr.DataArray(pram_loc.time.dt.year == end_loc[t].time.dt.year).astype(int).sum())
            if not np.isnan(end_loc[t]):
                pos_end = int(end_loc[t]) + n_days - 1
                if op in ["max", "sum"]:
                    pos_win_1 = pos_end + 1
                    pos_win_2 = min(pos_win_1 + window, n_days_t + n_days)
                else:
                    pos_win_2 = pos_end + 1
                    pos_win_1 = max(0, pos_end - window)
                pos_range = [min(pos_win_1, pos_win_2), max(pos_win_1, pos_win_2)]
                if not ((pram_loc.isel(time=slice(pos_range[0], pos_range[1])).sum(dim="time") > 0) or
                        (pram_loc.isel(time=pos_end) > 0)):
                    end_loc[t] = np.nan
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
        sel = (np.isnan(start).astype(int) == 0) &\
              (np.isnan(end).astype(int) == 0) &\
              (end <= start)
        end = xr.where(sel, np.nan, end)

    # If the season ends after or on start day of the next season, the end day of the current season becomes the day
    # before the next season.
    if start_next is not None:
        sel = (np.isnan(start_next).astype(int) == 0) &\
              (np.isnan(end).astype(int) == 0) &\
              (end >= start_next)
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
    pr : xr.DataArray
        Daily precipitation.
    start : xr.DataArray
        Rain season start (first day of year).
    end: xr.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xr.DataArray
        Rain season accumulated precipitation (mm/year).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Unit conversion.
    pram = rate2amount(pr, out_units="mm")

    # Initialize the array that will contain results.
    prcptot = xr.zeros_like(start) * np.nan

    # Calculate the sum between two dates for a given year.
    def calc_sum(year: int, start_doy: int, end_doy: int):
        sel = (pram.time.dt.year == year) & \
              (pram.time.dt.dayofyear >= start_doy) &\
              (pram.time.dt.dayofyear <= end_doy)
        return xr.where(sel, pram, 0).sum()

    # Calculate the index.
    def calc_idx(loc: str = ""):
        if loc == "":
            prcptot_loc = prcptot
            start_loc = start
            end_loc = end
        else:
            prcptot_loc = prcptot[prcptot.location == loc].squeeze()
            start_loc = start[start.location == loc].squeeze()
            end_loc = end[end.location == loc].squeeze()

        end_shift = None
        for t in range(len(start.time.dt.year)):
            year = int(start.time.dt.year[t])

            # Start and end dates in the same calendar year.
            if start_loc.mean() <= end_loc.mean():
                if (np.isnan(start_loc[t]).astype(bool) == 0) and (np.isnan(end_loc[t]).astype(bool) == 0):
                    prcptot_loc[t] = calc_sum(year, int(start_loc[t]), int(end_loc[t]))

            # Start and end dates not in the same year (left shift required).
            else:
                end_shift = end_loc.shift(time=-1, fill_value=np.nan) if t == 0 else end_shift
                if (np.isnan(start_loc[t]).astype(bool) == 0) and (np.isnan(end_shift[t]).astype(bool) == 0):
                    prcptot_loc[t] = calc_sum(year, int(start_loc[t]), 365) + calc_sum(year, 1, int(end_shift[t]))

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
    da_vv : xr.DataArray
        Wind speed (m s-1).
    da_dd : xr.DataArray
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
        da_w_days_above = da_conds.resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)

    return da_w_days_above


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    not_req = " (not required)"

    # Explode the list of index codes.
    idx_codes_exploded = cfg.explode_idx_l(cfg.idx_codes)

    # Indices ----------------------------------------------------------------------------------------------------------

    # Calculate indices.
    utils.log("=")
    msg = "Step #6   Calculating indices"
    if cfg.opt_idx:
        utils.log(msg)
        for i in range(0, len(cfg.idx_codes)):
            generate(cfg.idx_codes[i])
    else:
        utils.log(msg + not_req)

    # Statistics -------------------------------------------------------------------------------------------------------

    utils.log("=")
    msg = "Step #7   Exporting results (indices)"
    if cfg.opt_stat[1] or cfg.opt_save_csv[1]:
        utils.log(msg)
    else:
        utils.log(msg + not_req)

    utils.log("-")
    msg = "Step #7a  Calculating statistics (indices)"
    if cfg.opt_stat[1]:
        utils.log(msg)
        statistics.calc_stats(cfg.cat_idx)
    else:
        utils.log(msg + not_req)

    utils.log("-")
    msg = "Step #7b  Converting NetCDF to CSV files (indices)"
    if cfg.opt_save_csv[1] and not cfg.opt_ra:
        utils.log(msg)
        utils.log("-")
        statistics.conv_nc_csv(cfg.cat_idx)
    else:
        utils.log(msg + not_req)

    # Plots ------------------------------------------------------------------------------------------------------------

    utils.log("=")
    msg = "Step #8   Generating plots and maps (indices)"
    if cfg.opt_plot[1] or cfg.opt_map[1]:
        utils.log(msg)
    else:
        utils.log(msg + not_req)

    # Generate plots.
    utils.log("-")
    msg = "Step #8b  Generating time series (indices)"
    if cfg.opt_ts[1]:
        utils.log(msg)
        statistics.calc_ts(cfg.cat_idx)
    else:
        utils.log(msg + not_req)

    # Generate maps.
    # Heat maps are not generated from data at stations:
    # - the result is not good with a limited number of stations;
    # - calculation is very slow (something is wrong).
    utils.log("-")
    msg = "Step #8c  Generating heat maps (indices)"
    if cfg.opt_ra and cfg.opt_map[1]:
        utils.log(msg)

        # Loop through indices.
        for i in range(len(idx_codes_exploded)):

            # Generate maps.
            statistics.calc_heatmap(idx_codes_exploded[i])

    else:
        utils.log(msg + " (not required)")


if __name__ == "__main__":
    run()
