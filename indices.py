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
from typing import Tuple
from xclim.indices import run_length as rl
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to
from xclim.core.utils import DayOfYearStr


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
        return

    # Load datasets (one per variable or index).
    ds_varidx_l = []
    for i_varidx in range(0, len(varidx_name_l)):
        varidx_name = varidx_name_l[i_varidx]

        try:
            # Open dataset.
            p_sim_j =\
                cfg.get_equivalent_idx_path(p_sim[i_sim], varidx_name_l[0], cfg.get_idx_group(varidx_name), stn, rcp)
            # TODO: Testing with dask.
            ds = utils.open_netcdf(p_sim_j)
            # ds = utils.open_netcdf(p_sim_j, chunks={cfg.dim_time: 10}).load()

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
            da_pr       = ds_varidx_l[0][cfg.var_cordex_pr]
            thresh_wet  = float(idx_params_str[0])
            window_wet  = int(idx_params_str[1])
            thresh_dry  = float(idx_params_str[2])
            window_dry  = int(idx_params_str[3])
            window_tot  = int(idx_params_str[4])
            start_date  = str(idx_params_str[5]).replace("nan", "")
            end_date    = str(idx_params_str[6]).replace("nan", "")

            # Calculate index.
            da_idx =\
                xr.DataArray(rain_season_start(da_pr, thresh_wet, window_wet, thresh_dry, window_dry, window_tot,
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
            method     = idx_params_str[0]
            thresh     = float(idx_params_str[1])
            etp        = -1.0 if str(idx_params_str[2]) == "nan" else float(idx_params_str[2])
            window     = -1 if str(idx_params_str[3]) == "nan" else int(idx_params_str[3])
            start_date = str(idx_params_str[4]).replace("nan", "")
            end_date   = str(idx_params_str[5]).replace("nan", "")

            # Calculate index.
            da_idx = xr.DataArray(rain_season_end(da_pr, da_etp, da_start, da_start_next, method, thresh, etp, window,
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
            s_thresh_wet = float(idx_params_str[0])
            s_window_wet = int(idx_params_str[1])
            s_thresh_dry = float(idx_params_str[2])
            s_window_dry = int(idx_params_str[3])
            s_window_tot = int(idx_params_str[4])
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
            e_method     = idx_params_str[7]
            e_thresh     = float(idx_params_str[8])
            e_etp        = -1.0 if str(idx_params_str[9]) == "" else float(idx_params_str[9])
            e_window     = -1 if str(idx_params_str[10]) == "" else int(idx_params_str[10])
            e_start_date = str(idx_params_str[11]).replace("nan", "")
            e_end_date   = str(idx_params_str[12]).replace("nan", "")

            # Calculate indices.
            da_start, da_end, da_length, da_prcptot =\
                rain_season(da_pr, da_etp, da_start_next, s_thresh_wet, s_window_wet, s_thresh_dry, s_window_dry,
                            s_window_tot, s_start_date, s_end_date, e_method, e_thresh, e_etp, e_window,
                            e_start_date, e_end_date)

            # Add to list.
            da_idx_l = [da_start, da_end, da_length, da_prcptot]
            idx_units_l = [cfg.unit_1, cfg.unit_1, cfg.unit_1, cfg.get_unit(cfg.idx_rain_season_prcptot)]
            idx_name_l = [cfg.idx_rain_season_start, cfg.idx_rain_season_end, cfg.idx_rain_season_length,
                          cfg.idx_rain_season_prcptot]

        elif idx_name == cfg.idx_dry_spell_total_length:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            method = idx_params_str[0]
            thresh = float(idx_params_str[1])
            window = int(idx_params_str[2])
            dry_fill = bool(idx_params_str[3])
            start_date = end_date = ""
            if len(idx_params_str) == 6:
                start_date = str(idx_params_str[4])
                end_date = str(idx_params_str[5])

            # Calculate index.
            da_idx = xr.DataArray(dry_spell_total_length(da_pr, method, thresh, window, dry_fill, start_date, end_date))

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
            dims = list(ds_varidx_l[0][list(ds_varidx_l[0].data_vars.variables.mapping)[0]].dims)
            for j in range(len(dims)):
                dims[j] = dims[j].replace(cfg.dim_rlat, cfg.dim_latitude).replace(cfg.dim_rlon, cfg.dim_longitude)
                if dims[j] == cfg.dim_lat:
                    dims[j] = cfg.dim_latitude
                if dims[j] == cfg.dim_lon:
                    dims[j] = cfg.dim_longitude
            da_idx = da_idx.transpose(dims[0], dims[1], dims[2])

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


def dry_spell_total_length(
    pr: xr.DataArray,
    method: str,
    thresh: float,
    window: int,
    dry_fill: bool = True,
    start_date: str = "",
    end_date: str = "",

) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate the total length of dry periods.

    A dry period occurs if:
    - the daily precipitation amount is less than {thresh} during a period of {window} consecutive days
      (with option {method}="1d";
    - the total precipitation amount over a period of {window} consecutive days is less than {thresh}
      (with option {method}="tot").

    Parameters
    ----------
    pr : xr.DataArray:
        Daily precipitation.
    method : str
        Method linked to the period over which to combine data {"1d" = one day, "cumul" = cumulative}.
    thresh : float
        If {method} == "1d": daily precipitation amount under which precipitation is considered negligible.
        If {method} == "tot": sum of daily precipitation amounts under which the period is considered dry.
    window : int
        Minimum number of days in a dry period.
    dry_fill : bool, optional
        If True, missing values near the end of dataset are assumed to be dry (default value).
        If False, missing values near the end of dataset are assumed to be wet.
        The file value is used to compensate for the fact that we are missing the last days of the year before dataset
        and the first days of the year after dataset.
    start_date : str, optional
        First day of year to consider ("mm-dd").
    end_date : str, optional
        Last day of year to consider ("mm-dd").

    Returns
    -------
    xarray.DataArray
        Dry spell total length (days/year).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Unit conversion.
    thresh = convert_units_to(str(thresh) + " mm/day", pr)

    # Eliminate negative values.
    pr = xr.where(pr < (thresh if method == "1d" else 0), 0, pr)

    # Identify dry days.
    if method == "1d":
        da_dry_last = pr.rolling(time=window).max() < thresh
        da_dry = da_dry_last.copy()
        for i in range(1, window):
            da_i = da_dry_last.shift(time=-i, fill_value=(dry_fill is False))
            da_dry = da_dry | da_i
    else:
        da_wet_last = pr.rolling(time=window).sum()
        da_wet = da_wet_last.copy()
        for i in range(1, window):
            da_i = da_wet_last.shift(time=-i, fill_value=(thresh if dry_fill is False else 0))
            da_wet = xr.ufuncs.maximum(da_wet, da_i)
        da_dry = (da_wet < thresh) | (pr == 0)

    # Identify days that are between 'start_date' and 'start_date'.
    doy_start = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    doy_end = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if doy_end >= doy_start:
        da_doy = (pr.time.dt.dayofyear >= doy_start) & (pr.time.dt.dayofyear <= doy_end)
    else:
        da_doy = (pr.time.dt.dayofyear <= doy_end) | (pr.time.dt.dayofyear >= doy_start)

    # Combine conditions.
    da_conds = da_dry & da_doy

    # Calculate the number of dry days per year.
    da_idx = da_conds.astype(float).resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)

    return da_idx


def rain_season(
    da_pr: xr.DataArray,
    da_etp: xr.DataArray,
    da_start_next: xr.DataArray,
    s_thresh_wet: float,
    s_window_wet: int,
    s_thresh_dry: float,
    s_window_dry: int,
    s_window_tot: int,
    s_start_date: str,
    s_end_date: str,
    e_method: str,
    e_thresh: float,
    e_etp: float,
    e_window: int,
    e_start_date: str,
    e_end_date: str
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate rain start, rain end, drain duration and rain quantity.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_etp : xr.DataArray
        Evapotranspiration data.
    da_start_next : xr.DataArray
        First day of the next rain season.
    s_thresh_wet : float
        Daily precipitation amount required in first {s_window_wet} days.
    s_window_wet: int
        Number of days with precipitation at season start (related to {s_thresh_wet}).
    s_thresh_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    s_window_dry: int
        Maximum number of days in a dry period embedded into the period of {s_window_tot} days.
    s_window_tot: int
        Number of days (after the first {s_window_wet} days) after which a dry season is searched for.
    s_start_date: str
        First day of year where season can start ("mm-dd").
    s_end_date: str
        Last day of year where season can start ("mm-dd").
    e_method : str
        Calculation method = {"depletion", "event", "cumul"]
        If "depletion": based on the period required for an amount of water (mm) to evaporate, considering
            that any amount of precipitation received during that period must evaporate. If {da_etp} is not available,
            the evapotranspiration rate is assumed to be {etp} (mm/day).
        If "event": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season stops when no daily precipitation greater than {thresh} have occurred over a period of
            {window} days.
        If "cumul": based on a total amount of precipitation received during the last days of the rain season.
            The rain season stops when the total amount of precipitation is less than {thresh} over a period of {window}
            days.
    e_thresh : float
        If {e_method} == "Depletion": precipitation amount that must evaporate (mm).
        If {e_method} == "Event": last non-negligible precipitation event of the rain season (mm).
    e_etp: float
        If {e_method} == "Depletion": Evapotranspiration rate (mm/day).
    e_window: int
        If {e_method} == "Event": period (number of days) during which there must not be a day with {e_thresh}
        precipitation.
    e_start_date: str
        First day of year at or after which the season ends ("mm-dd").
    e_end_date: str
        Last day of year at or before which the season ends ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    # Rename dimensions to have latitude and longitude dimensions.
    def rename_dimensions(da: xr.DataArray, lat_name: str = cfg.dim_latitude, lon_name: str = cfg.dim_longitude) \
            -> xr.DataArray:
        if (lat_name not in da.dims) or (lon_name not in da.dims):
            if "dim_0" in list(da.dims):
                da = da.rename({"dim_0": cfg.dim_time})
                da = da.rename({"dim_1": lat_name, "dim_2": lon_name})
            elif (cfg.dim_lat in list(da.dims)) or (cfg.dim_lon in list(da.dims)):
                da = da.rename({cfg.dim_lat: lat_name, cfg.dim_lon: lon_name})
            elif (cfg.dim_rlat in list(da.dims)) or (cfg.dim_rlon in list(da.dims)):
                da = da.rename({cfg.dim_rlat: lat_name, cfg.dim_rlon: lon_name})
            elif (lat_name not in list(da.dims)) and (lon_name not in list(da.dims)):
                if lat_name == cfg.dim_latitude:
                    da = da.expand_dims(latitude=1)
                if lon_name == cfg.dim_longitude:
                    da = da.expand_dims(longitude=1)
        return da
    da_pr = rename_dimensions(da_pr)
    da_etp = rename_dimensions(da_etp)

    # Calculate rain season start.
    da_start = xr.DataArray(rain_season_start(da_pr, s_thresh_wet, s_window_wet, s_thresh_dry, s_window_dry,
                                              s_window_tot, s_start_date, s_end_date))

    # Calculate rain season end.
    da_end = xr.DataArray(rain_season_end(da_pr, da_etp, da_start, da_start_next, e_method, e_thresh, e_etp, e_window,
                                          e_start_date, e_end_date))

    # Calculate rain season length.
    da_length = xr.where(da_end >= da_start, da_end - da_start + 1, da_end + 365 - da_start + 1)
    da_length = xr.where(da_length < 0, 0, da_length)

    # Calculate rain quantity.
    da_prcptot = indices_gen.aggregate_between_dates(da_pr, da_start, da_end, "sum") * 86400

    return da_start, da_end, da_length, da_prcptot


def rain_season_start(
    da_pr: xr.DataArray,
    thresh_wet: float,
    window_wet: int,
    thresh_dry: float,
    window_dry: int,
    window_tot: int,
    start_date: str,
    end_date: str
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the first day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    thresh_wet : float
        Daily precipitation amount required in first {window_wet} days.
    window_wet: int
        Number of days with precipitation at season start (related to {thresh_wet}).
    thresh_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    window_dry: int
        Maximum number of days in a dry period embedded into the period of {window_tot} days.
    window_tot: int
        Number of days (after the first {window_wet} days) after which a dry season is searched for.
    start_date: str
        First day of year where season can start ("mm-dd").
    end_date: str
        Last day of year where season can start ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    # Unit conversion.
    thresh_wet = convert_units_to(str(thresh_wet) + " mm/day", da_pr)
    thresh_dry = convert_units_to(str(thresh_dry) + " mm/day", da_pr)

    # Eliminate negative values.
    da_pr = xr.where(da_pr < 0, 0, da_pr)

    # Assign search boundaries.
    start_doy = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if (start_date == "") and (end_date != ""):
        start_doy = 1 if end_doy == 365 else end_doy + 1
    elif (start_date != "") and (end_date == ""):
        end_doy = 365 if start_doy == 1 else start_doy - 1

    # Flag the first day of each sequence of {window_wet} days with a total of {thresh_wet} in precipitation
    # (assign True).
    da_wet = xr.DataArray(da_pr.rolling(time=window_wet).sum() >= thresh_wet)\
        .shift(time=-(window_wet - 1), fill_value=False)

    # Identify dry days (assign 1).
    da_dry_day = xr.where(da_pr < thresh_dry, 1, 0)

    # Identify each day that is not followed by a sequence of {window_dry} days within a period of {window_tot} days,
    # starting after {window_wet} days (assign True).
    da_dry_seq = None
    for i in range(window_tot - window_dry - 1):
        da_dry_day_i = da_dry_day.shift(time=-(i + window_wet), fill_value=False)
        da_dry_seq_i = xr.DataArray(da_dry_day_i.rolling(time=window_dry).sum() >= window_dry)\
            .shift(time=-(window_dry - 1), fill_value=False)
        if i == 0:
            da_dry_seq = da_dry_seq_i.copy()
        else:
            da_dry_seq = da_dry_seq | da_dry_seq_i
    da_no_dry_seq = (da_dry_seq == False)

    # Flag days between {start_date} and {end_date} (or the opposite).
    if end_doy >= start_doy:
        da_doy = (da_pr.time.dt.dayofyear >= start_doy) & (da_pr.time.dt.dayofyear <= end_doy)
    else:
        da_doy = (da_pr.time.dt.dayofyear <= end_doy) | (da_pr.time.dt.dayofyear >= start_doy)

    # Combine conditions.
    da_conds = da_wet & da_no_dry_seq & da_doy

    # Obtain the first day of each year where conditions apply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_start = da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_start = xr.where((da_start < 1) | (da_start > 365), np.nan, da_start)

    return da_start


def rain_season_end(
    da_pr: xr.DataArray,
    da_etp: xr.DataArray,
    da_start: xr.DataArray,
    da_start_next: xr.DataArray,
    method: str,
    thresh: float,
    etp: float,
    window: int,
    start_date: str,
    end_date: str
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_etp : xr.DataArray
        Evapotranspiration data.
    da_start : xr.DataArray
        First day of the current rain season.
    da_start_next : xr.DataArray
        First day of the next rain season.
    method : str
        Calculation method = {"depletion", "event", "cumul"]
        If "depletion": based on the period required for an amount of water (mm) to evaporate, considering
            that any amount of precipitation received during that period must evaporate. If {da_etp} is not available,
            the evapotranspiration rate is assumed to be {etp} (mm/day).
        If "event": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season stops when no daily precipitation greater than {thresh} have occurred over a period of
            {window} days.
        If "cumul": based on a total amount of precipitation received during the last days of the rain season.
            The rain season stops when the total amount of precipitation is less than {thresh} over a period of {window}
            days.
    thresh : float
        If method == "depletion": precipitation amount that must evaporate (mm).
        If method == "event": threshold daily precipitation amount during a period (mm/day).
        If method == "cumul": threshold total precipitation amount over a period (mm).
    etp: float
        If method == "depletion": evapotranspiration rate (mm/day).
        Otherwise: not used.
    window: int
        If method in ["event", "cumul"]: length of period (number of days) used to verify if the rain season is ending.
        Otherwise: not used.
    start_date: str
        First day of year at or after which the season can end ("mm-dd").
    end_date: str
        Last day of year at or before which the season can end ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    # Unit conversion.
    thresh = convert_units_to(str(thresh) + " mm/day", da_pr)
    etp    = convert_units_to(str(etp) + " mm/day", da_pr)

    # Eliminate negative values.
    da_pr = xr.where(da_pr < 0, 0, da_pr)
    if da_etp is not None:
        da_etp = xr.where(da_etp < 0, 0, da_etp)

    # Assign search boundaries.
    start_doy = 1 if start_date == "" else datetime.datetime.strptime(start_date, "%m-%d").timetuple().tm_yday
    end_doy = 365 if end_date == "" else datetime.datetime.strptime(end_date, "%m-%d").timetuple().tm_yday
    if (start_date == "") and (end_date != ""):
        start_doy = 1 if end_doy == 365 else end_doy + 1
    elif (start_date != "") and (end_date == ""):
        end_doy = 365 if start_doy == 1 else start_doy - 1

    # Flag days between {start_date} and {end_date} (or the opposite).
    if end_doy >= start_doy:
        da_doy = (da_pr.time.dt.dayofyear >= start_doy) & (da_pr.time.dt.dayofyear <= end_doy)
    else:
        da_doy = (da_pr.time.dt.dayofyear <= end_doy) | (da_pr.time.dt.dayofyear >= start_doy)

    # Depletion method -------------------------------------------------------------------------------------------------

    da_end = None

    if method == "depletion":

        # Calculate the minimum length of the period.
        window_min = math.ceil(thresh / etp) if (etp > 0) else 0
        window_max =\
            (end_doy - start_doy + 1 if (end_doy >= start_doy) else (365 - start_doy + 1 + end_doy)) - window_min
        if etp == 0:
            window_min = window_max

        for window_i in list(range(window_min, window_max + 1)):

            # Flag the day before each sequence of {dt} days that results in evaporating a water column, considering
            # precipitation falling during this period (assign 1).
            if da_etp is None:
                da_dry_seq = xr.DataArray((da_pr.rolling(time=window_i).sum() + thresh) <= (window_i * etp))
            else:
                da_dry_seq =\
                    xr.DataArray((da_pr.rolling(time=window_i).sum() + thresh) <= da_etp.rolling(time=window_i).sum())

            # Combine conditions.
            da_conds = da_dry_seq & da_doy

            # Obtain the first day of each year where conditions apply.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_end_i = \
                    da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")

            # Update the cells that were not assigned yet.
            if da_end is None:
                da_end = da_end_i.copy()
            else:
                sel = xr.ufuncs.isnan(da_end) & ((xr.ufuncs.isnan(da_end_i).astype(int) == 0) | (da_end_i < da_end))
                da_end = xr.where(sel, da_end_i, da_end)

            # Exit loop if all cells were assigned a value.
            if xr.ufuncs.isnan(da_end).astype(int).sum() == 0:
                break

        # The season can't end at the beginning of the dataset; there was no rain yet, so no season.
        if da_end[0] <= window_min:
            da_end[0] = np.nan

    # Event method -----------------------------------------------------------------------------------------------------

    elif method in ["event", "cumul"]:

        # Determine if it rains heavily (assign 1) or not (assign 0).
        da_heavy_rain = xr.where(da_pr < thresh, 0, 1)

        # Flag the day (assign 1) before each sequence of:
        # {window} days with no amount reaching {thresh}:
        if method == "event":
            da_dry_seq = xr.DataArray(da_heavy_rain.rolling(time=window).sum() == 0)
        # {window} days with a total amount reaching {thresh}:
        else:
            da_dry_seq = xr.DataArray(da_pr.rolling(time=window).sum() < thresh)
        da_dry_seq = da_dry_seq.shift(time=-(window + 1), fill_value=False)

        # Combine conditions.
        da_conds = da_dry_seq & da_doy

        # Obtain the first day of each year where conditions apply.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_end =\
                da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")

    # Adjust or discard rain end values that are not compatible with the current or next season start values.
    # If the season ends before or on start day, discard rain end.
    if da_start is not None:
        sel = (xr.ufuncs.isnan(da_start).astype(int) == 0) &\
              (xr.ufuncs.isnan(da_end).astype(int) == 0) &\
              (da_end <= da_start)
        da_end = xr.where(sel, np.nan, da_end)

    # The season can't end on {start_date} if the start days of rain seasons are unknown.
    else:
        da_end = xr.where(da_end == start_doy, np.nan, da_end)

    # If the season ends after or on start day of the next season, the end day of the current season becomes the day
    # before the next season.
    if da_start_next is not None:
        sel = (xr.ufuncs.isnan(da_start_next).astype(int) == 0) &\
              (xr.ufuncs.isnan(da_end).astype(int) == 0) &\
              (da_end >= da_start_next)
        da_end = xr.where(sel, da_start_next - 1, da_end)
        da_end = xr.where(da_end < 1, 365, da_end)

    return da_end


def rain_season_length(
    da_start: xr.DataArray,
    da_end: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the length of the rain season.

    Parameters
    ----------
    da_start : xr.DataArray
        Rain season start (first day of year).
    da_end: xr.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xr.DataArray
        Rain season length (days/year).
    --------------------------------------------------------------------------------------------------------------------
    """

    da_length = xr.where(da_end >= da_start, da_end - da_start + 1, da_end + 365 - da_start + 1)
    da_length = xr.where(da_length < 0, 0, da_length)

    return da_length


def rain_season_prcptot(
    da_pr: xr.DataArray,
    da_start: xr.DataArray,
    da_end: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine precipitation amount during rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_start : xr.DataArray
        Rain season start (first day of year).
    da_end: xr.DataArray
        Rain season end (last day of year).

    Returns
    -------
    xr.DataArray
        Rain season precipitation amount.
    --------------------------------------------------------------------------------------------------------------------
    """

    da_prcptot = indices_gen.aggregate_between_dates(da_pr, da_start, da_end, "sum") * cfg.spd

    return da_prcptot


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
    if cfg.opt_plot[1]:
        utils.log("=")
        utils.log("Step #8b  Generating time series (indices)")
        statistics.calc_ts(cfg.cat_idx)

    # Generate maps.
    # Heat maps are not generated from data at stations:
    # - the result is not good with a limited number of stations;
    # - calculation is very slow (something is wrong).
    utils.log("-")
    msg = "Step #8c  Generating heat maps (indices)"
    if cfg.opt_ra and (cfg.opt_map[1] or cfg.opt_save_csv[1]):
        utils.log(msg)

        # Loop through indices.
        for i in range(len(idx_codes_exploded)):

            # Generate maps.
            statistics.calc_heatmap(idx_codes_exploded[i])

    else:
        utils.log(msg + " (not required)")


if __name__ == "__main__":
    run()
