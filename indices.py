# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions related to climate indices.
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


def generate(idx_code: str):

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
    if idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights, cfg.idx_tngmonthsbelow, cfg.idx_heatwavemaxlen,
                    cfg.idx_heatwavetotlen, cfg.idx_tgg, cfg.idx_etr, cfg.idx_tndaysbelow]:
        varidx_name_l.append(cfg.var_cordex_tasmin)

    if idx_name in [cfg.idx_tx90p, cfg.idx_txdaysabove, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_txg,
                    cfg.idx_txx, cfg.idx_wsdi, cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen, cfg.idx_tgg,
                    cfg.idx_etr]:
        varidx_name_l.append(cfg.var_cordex_tasmax)

    # Precipitation.
    if idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_cwd, cfg.idx_cdd, cfg.idx_sdii, cfg.idx_prcptot,
                    cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_rainstart,
                    cfg.idx_rainend, cfg.idx_rainqty, cfg.idx_drydurtot, cfg.idx_rainseason]:
        varidx_name_l.append(cfg.var_cordex_pr)

        if idx_name in [cfg.idx_rainend, cfg.idx_rainseason]:
            if cfg.var_cordex_evspsblpot in cfg.variables_cordex:
                varidx_name_l.append(cfg.var_cordex_evspsblpot)
            elif cfg.var_cordex_evspsbl in cfg.variables_cordex:
                varidx_name_l.append(cfg.var_cordex_evspsbl)
            else:
                varidx_name_l.append("nan")

        if idx_name in [cfg.idx_rainend, cfg.idx_raindur, cfg.idx_rainqty]:
            varidx_name_l.append(idx_code.replace(idx_name, cfg.idx_rainstart))

        if idx_name == cfg.idx_rainseason:
            varidx_name_l.append("nan")

        if idx_name in [cfg.idx_rainend, cfg.idx_rainseason]:
            varidx_name_l.append(str(idx_params[len(idx_params) - 1]))

        if idx_name in [cfg.idx_raindur, cfg.idx_rainqty]:
            varidx_name_l.append(idx_code.replace(idx_name, cfg.idx_rainend))

    # Temperature-precipitation.
    if idx_name == cfg.idx_dc:
        varidx_name_l.append(cfg.var_cordex_tas)
        varidx_name_l.append(cfg.var_cordex_pr)

    # Wind.
    if idx_name == cfg.idx_wgdaysabove:
        varidx_name_l.append(cfg.var_cordex_uas)
        varidx_name_l.append(cfg.var_cordex_vas)

    elif idx_name == cfg.idx_wxdaysabove:
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
            da_mask = utils.create_mask(stn)

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


def generate_single(idx_code: str, idx_params, varidx_name_l: [str], p_sim: [str], stn: str, rcp: str,
                    da_mask: xr.DataArray, i_sim: int):

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
        if (idx_name in [cfg.idx_txdaysabove, cfg.idx_tngmonthsbelow, cfg.idx_tx90p, cfg.idx_prcptot,
                         cfg.idx_tropicalnights]) or\
           ((idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]) and (i == 0)) or \
           ((idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]) and (i <= 1)) or \
           ((idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 0)):

            if "p" in str(idx_param):
                idx_param = float(idx_param.replace("p", "")) / 100.0
                if rcp == cfg.rcp_ref:
                    # Calculate percentile.
                    if (idx_name in [cfg.idx_tx90p, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]) or\
                       (i == 1):
                        idx_param = ds_varidx_l[i][cfg.var_cordex_tasmax].quantile(idx_param).values.ravel()[0]
                    elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
                        idx_param = ds_varidx_l[i][cfg.var_cordex_tasmin].quantile(idx_param).values.ravel()[0]
                    elif idx_name == cfg.idx_prcptot:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            da_i = ds_varidx_l[i][cfg.var_cordex_pr].resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)
                        dims = utils.get_coord_names(ds_varidx_l[i])
                        idx_param = da_i.mean(dim=dims).quantile(idx_param).values.ravel()[0] * cfg.spd
                    elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
                        if idx_name == cfg.idx_wgdaysabove:
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

        if (idx_name in [cfg.idx_tx90p, cfg.idx_tropicalnights]) or\
           ((idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi, cfg.idx_tndaysbelow,
                          cfg.idx_txdaysabove]) and (i == 0)) or \
           ((idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]) and (i <= 1)):
            idx_ref = str(idx_param) + " " + cfg.unit_C
            idx_fut = str(idx_param + cfg.d_KC) + " " + cfg.unit_K
            idx_params_str.append(idx_ref if (rcp == cfg.rcp_ref) else idx_fut)

        elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                          cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:
            idx_params_str.append(str(idx_param) + " mm/day")

        elif (idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 1):
            idx_params_str.append(str(idx_param) + " " + cfg.unit_m_s)

        elif not ((idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 4)):
            idx_params_str.append(str(idx_param))

    # Exit loop if the file already exists (reference file only).
    if not ((rcp == cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite)):

        # Will hold data arrays and units.
        da_idx_l    = []
        idx_units_l = []
        idx_name_l  = []

        # Temperature --------------------------------------------------------------------------------------------------

        if idx_name in [cfg.idx_txdaysabove, cfg.idx_tx90p]:

            # Collect required datasets and parameters.
            da_tasmax = ds_varidx_l[0][cfg.var_cordex_tasmax]
            param_tasmax = idx_params_str[0]
            doy_a = doy_b = -1
            if len(idx_params_str) > 1:
                doy_a = -1.0 if str(idx_params_str[1]) == "nan" else int(idx_params_str[1])
                doy_b = -1.0 if str(idx_params_str[2]) == "nan" else int(idx_params_str[2])

            # Calculate index.
            da_idx = tx_days_above(da_tasmax, param_tasmax, doy_a, doy_b)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        if idx_name == cfg.idx_tndaysbelow:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]
            param_tasmin = idx_params_str[0]
            doy_a = doy_b = -1
            if len(idx_params_str) > 1:
                doy_a = -1.0 if str(idx_params_str[1]) == "nan" else int(idx_params_str[1])
                doy_b = -1.0 if str(idx_params_str[2]) == "nan" else int(idx_params_str[2])

            # Calculate index.
            da_idx = tn_days_below(da_tasmin, param_tasmin, doy_a, doy_b)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_tngmonthsbelow:

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

        elif idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]:

            # Collect required datasets and parameters.
            da_tasmax = ds_varidx_l[0][cfg.var_cordex_tasmax]
            param_tasmax = idx_params_str[0]
            param_ndays = int(float(idx_params_str[1]))

            # Calculate index.
            if idx_name == cfg.idx_hotspellfreq:
                da_idx = xr.DataArray(
                    indices.hot_spell_frequency(da_tasmax, param_tasmax, param_ndays).values)
            elif idx_name == cfg.idx_hotspellmaxlen:
                da_idx = xr.DataArray(
                    indices.hot_spell_max_length(da_tasmax, param_tasmax, param_ndays).values)
            else:
                da_idx = xr.DataArray(
                    indices.warm_spell_duration_index(da_tasmax, da_tx90p, param_ndays).values)
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:

            # Collect required datasets and parameters.
            da_tasmin = ds_varidx_l[0][cfg.var_cordex_tasmin]
            da_tasmax = ds_varidx_l[1][cfg.var_cordex_tasmax]
            param_tasmin = idx_params_str[0]
            param_tasmax = idx_params_str[1]
            window = int(float(idx_params_str[2]))

            # Calculate index.
            if idx_name == cfg.idx_heatwavemaxlen:
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

        elif idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights]:

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

        elif idx_name == cfg.idx_dc:

            # Collect required datasets and parameters.
            da_tas = ds_varidx_l[0][cfg.var_cordex_tas]
            da_pr  = ds_varidx_l[1][cfg.var_cordex_pr]
            da_lon, da_lat = utils.get_coordinates(ds_varidx_l[0])

            # Calculate index
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.drought_code(
                    da_tas, da_pr, da_lat, shut_down_mode="temperature")).resample(time=cfg.freq_YS).mean()

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
                    doy_a = doy_b = -1
                    if len(idx_params_str) == 3:
                        doy_a = -1.0 if str(idx_params_str[1]) == "nan" else int(idx_params_str[1])
                        doy_b = -1.0 if str(idx_params_str[2]) == "nan" else int(idx_params_str[2])
                    da_idx = xr.DataArray(precip_accumulation(da_pr, doy_a, doy_b))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(da_idx.attrs[cfg.attrs_units])

        elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                          cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:

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
            elif idx_name in [cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    da_idx = xr.DataArray(indices.wetdays(da_pr, param_pr, cfg.freq_YS))
            elif idx_name == cfg.idx_drydays:
                da_idx = xr.DataArray(indices.dry_days(da_pr, param_pr, cfg.freq_YS))
            else:
                da_idx = xr.DataArray(indices.daily_pr_intensity(da_pr, param_pr))
            da_idx = da_idx.astype(int)

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rainstart:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            pr_wet = float(idx_params_str[0])
            dt_wet = int(idx_params_str[1])
            doy_a = -1.0 if str(idx_params_str[2]) == "nan" else int(idx_params_str[2])
            doy_b = -1.0 if str(idx_params_str[3]) == "nan" else int(idx_params_str[3])
            pr_dry = float(idx_params_str[4])
            dt_dry = int(idx_params_str[5])
            dt_tot = int(idx_params_str[6])

            # Calculate index.
            da_idx = xr.DataArray(rain_start(da_pr, pr_wet, dt_wet, doy_a, doy_b, pr_dry, dt_dry, dt_tot))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rainend:

            # Collect required datasets and parameters.
            da_pr  = ds_varidx_l[0][cfg.var_cordex_pr]
            da_etp = None
            if ds_varidx_l[1] is not None:
                if cfg.var_cordex_evspsblpot in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsblpot]
                elif cfg.var_cordex_evspsbl in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsbl]
            da_rainstart = None
            if ds_varidx_l[2] is not None:
                da_rainstart = ds_varidx_l[2][cfg.idx_rainstart]
            da_rainstart_next = None
            if ds_varidx_l[3] is not None:
                da_rainstart_next = ds_varidx_l[3][cfg.idx_rainstart]
            meth  = idx_params_str[0]
            pr    = float(idx_params_str[1])
            etp   = -1.0 if str(idx_params_str[2]) == "nan" else float(idx_params_str[2])
            dt    = -1.0 if str(idx_params_str[3]) == "nan" else float(idx_params_str[3])
            doy_a = -1.0 if str(idx_params_str[4]) == "nan" else int(idx_params_str[4])
            doy_b = -1.0 if str(idx_params_str[5]) == "nan" else int(idx_params_str[5])

            # Calculate index.
            da_idx =\
                xr.DataArray(rain_end(da_pr, da_etp, da_rainstart, da_rainstart_next, meth, pr, etp, dt, doy_a, doy_b))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_raindur:

            # Collect required datasets and parameters.
            da_rainstart = ds_varidx_l[0][cfg.idx_rainstart]
            da_rainend   = ds_varidx_l[1][cfg.idx_rainend]

            # Calculate index.
            da_idx = da_rainend - da_rainstart + 1
            da_idx.values[da_idx.values < 0] = 0

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        elif idx_name == cfg.idx_rainqty:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            da_rainstart = ds_varidx_l[1][cfg.idx_rainstart]
            da_rainend = ds_varidx_l[2][cfg.idx_rainend]

            # Calculate index.
            da_idx = xr.DataArray(rain_qty(da_pr, da_rainstart, da_rainend))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.get_unit(idx_name))

        elif idx_name == cfg.idx_rainseason:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            # Rain start:
            rs_pr_wet = float(idx_params_str[0])
            rs_dt_wet = int(idx_params_str[1])
            rs_doy_a = -1.0 if str(idx_params_str[2]) == "nan" else int(idx_params_str[2])
            rs_doy_b = -1.0 if str(idx_params_str[3]) == "nan" else int(idx_params_str[3])
            rs_pr_dry = float(idx_params_str[4])
            rs_dt_dry = int(idx_params_str[5])
            rs_dt_tot = int(idx_params_str[6])
            # Rain end:
            da_etp = None
            if ds_varidx_l[1] is not None:
                if cfg.var_cordex_evspsblpot in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsblpot]
                elif cfg.var_cordex_evspsbl in cfg.variables_cordex:
                    da_etp = ds_varidx_l[1][cfg.var_cordex_evspsbl]
            da_rainstart_next = None
            if ds_varidx_l[3] is not None:
                da_rainstart_next = ds_varidx_l[3][cfg.idx_rainstart]
            re_meth  = idx_params_str[7]
            re_pr    = float(idx_params_str[8])
            re_etp   = -1.0 if str(idx_params_str[9]) == "nan" else float(idx_params_str[9])
            re_dt    = -1.0 if str(idx_params_str[10]) == "nan" else float(idx_params_str[10])
            re_doy_a = -1.0 if str(idx_params_str[11]) == "nan" else int(idx_params_str[11])
            re_doy_b = -1.0 if str(idx_params_str[12]) == "nan" else int(idx_params_str[12])

            # Calculate indices.
            da_rainstart, da_rainend, da_raindur, da_rainqty =\
                rain_season(da_pr, da_etp, da_rainstart_next, rs_pr_wet, rs_dt_wet, rs_doy_a, rs_doy_b, rs_pr_dry,
                            rs_dt_dry, rs_dt_tot, re_meth, re_pr, re_etp, re_dt, re_doy_a, re_doy_b)

            # Add to list.
            da_idx_l = [da_rainstart, da_rainend, da_raindur, da_rainqty]
            idx_units_l = [cfg.unit_1, cfg.unit_1, cfg.unit_1, cfg.get_unit(cfg.idx_rainqty)]
            idx_name_l = [cfg.idx_rainstart, cfg.idx_rainend, cfg.idx_raindur, cfg.idx_rainqty]

        elif idx_name == cfg.idx_drydurtot:

            # Collect required datasets and parameters.
            da_pr = ds_varidx_l[0][cfg.var_cordex_pr]
            per = idx_params_str[0]
            p_tot = idx_params_str[1]
            p_tot = 1.0 if (str(p_tot) == "nan") else float(p_tot)
            d_dry = int(idx_params_str[2])
            p_dry = idx_params_str[3]
            p_dry = 1.0 if (str(p_dry) == "nan") else float(p_dry)
            doy_a = doy_b = -1
            if len(idx_params_str) == 6:
                doy_a = -1.0 if str(idx_params_str[4]) == "nan" else int(idx_params_str[4])
                doy_b = -1.0 if str(idx_params_str[5]) == "nan" else int(idx_params_str[5])

            # Calculate index.
            da_idx = xr.DataArray(tot_duration_dry_periods(da_pr, per, p_tot, d_dry, p_dry, doy_a, doy_b))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        # Wind ---------------------------------------------------------------------------------------------

        elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:

            # Collect required datasets and parameters.
            param_vv     = float(idx_params_str[0])
            param_vv_neg = idx_params_str[1]
            param_dd     = float(idx_params_str[2])
            param_dd_tol = float(idx_params_str[3])
            doy_a = doy_b = -1
            if len(idx_params_str) == 6:
                doy_a = -1.0 if str(idx_params_str[4]) == "nan" else int(idx_params_str[4])
                doy_b = -1.0 if str(idx_params_str[5]) == "nan" else int(idx_params_str[5])
            if idx_name == cfg.idx_wgdaysabove:
                da_uas = ds_varidx_l[0][cfg.var_cordex_uas]
                da_vas = ds_varidx_l[1][cfg.var_cordex_vas]
                da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas, param_vv_neg)
            else:
                da_vv = ds_varidx_l[0][cfg.var_cordex_sfcwindmax]
                da_dd = None

            # Calculate index.
            da_idx = xr.DataArray(wind_days_above(da_vv, da_dd, param_vv, param_dd, param_dd_tol, doy_a, doy_b))

            # Add to list.
            da_idx_l.append(da_idx)
            idx_units_l.append(cfg.unit_1)

        # ======================================================================================================
        # TODO.CUSTOMIZATION.INDEX.END
        # ======================================================================================================

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
            ds_idx[idx_name_l[i]] =\
                utils.copy_coordinates(ds_varidx_l[0][cfg.extract_idx(varidx_name_l[0])], da_idx)

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
        desc = "/" + idx_name + "/" + os.path.basename(p_idx)
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


def precip_accumulation(da_pr: xr.DataArray, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent function in xclim.indices.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    doy_a: int
        First day of year to consider.
    doy_b: int
        Last day of year to consider.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Subset based on 'day of year'.
    da_pr = utils.subset_doy(da_pr, doy_a, doy_b)

    # Calculate index.
    da_idx = xr.DataArray(indices.precip_accumulation(da_pr, freq=cfg.freq_YS))

    return da_idx


def tx_days_above(da_tasmax: xr.DataArray, param_tasmax: str, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent function in xclim.indices.

    Parameters
    ----------
    da_tasmax : xr.DataArray
        Maximum temperature data.
    param_tasmax : str
        Maximum temperature threshold value.
    doy_a: int
        First day of year to consider.
    doy_b: int
        Last day of year to consider.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Subset based on 'day of year'.
    da_tasmax = utils.subset_doy(da_tasmax, doy_a, doy_b)

    # Calculate index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_idx = xr.DataArray(indices.tx_days_above(da_tasmax, param_tasmax).values)

    return da_idx


def tn_days_below(da_tasmin: xr.DataArray, param_tasmin: str, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper of equivalent function in xclim.indices.

    Parameters
    ----------
    da_tasmin : xr.DataArray
        Minimum temperature data.
    param_tasmin : str
        Minimum temperature threshold value.
    doy_a: int
        First day of year to consider.
    doy_b: int
        Last day of year to consider.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Subset based on 'day of year'.
    da_tasmin = utils.subset_doy(da_tasmin, doy_a, doy_b)

    # Calculate index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_idx = xr.DataArray(indices.tn_days_below(da_tasmin, param_tasmin).values)

    return da_idx


def heat_wave_max_length(tasmin: xr.DataArray, tasmax: xr.DataArray, param_tasmin: str = "22.0 degC",
                         param_tasmax: str = "30 degC", window: int = 3, freq: str = cfg.freq_YS) -> xr.DataArray:

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


def heat_wave_total_length(tasmin: xr.DataArray, tasmax: xr.DataArray, param_tasmin: str = "22.0 degC",
                           param_tasmax: str = "30 degC", window: int = 3, freq: str = cfg.freq_YS) -> xr.DataArray:

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


def tot_duration_dry_periods(da_pr: xr.DataArray,  per: str, pr_tot: float, dt_dry: int, pr_dry: float, doy_a: int,
                             doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the total duration of dry periods. A dry period occurs if precipitation amount is less than 'pr_dry'
    during 'dt_dry' consecutive days.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    per: str
        Period over which to combine data {"1d" = one day, "tot" = total}.
    pr_tot: float
        Sum of daily precipitation amounts under which the period is considered dry (only if per="tot).
    dt_dry: int
        Number of days to have a dry period.
    pr_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    doy_a: int
        First day of year to consider.
    doy_b: int
        Last day of year to consider.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    pr_dry = convert_units_to(str(pr_dry) + " mm/day", da_pr)
    pr_tot = convert_units_to(str(pr_tot) + " mm/day", da_pr)

    # Condition #1: Days that belong to a dry period.
    da_cond1 = da_pr.copy()
    da_cond1 = da_cond1.astype(bool)
    da_cond1[:, :, :] = False
    n_t = len(da_pr[cfg.dim_time])
    for t in range(n_t - dt_dry):
        if per == "1d":
            da_t = xr.DataArray(da_pr[t:(t + dt_dry), :, :] < pr_dry).sum(dim=cfg.dim_time) == dt_dry
        else:
            da_t = da_pr[t:(t + dt_dry), :, :]
            if pr_dry >= 0:
                da_t = xr.DataArray(da_t >= pr_dry).astype(float) * da_t
            da_t = xr.DataArray(da_t.sum(dim=cfg.dim_time)) < pr_tot
        da_cond1[t:(t + dt_dry), :, :] = da_cond1[t:(t + dt_dry), :, :] | da_t

    # Condition #2 : Days that are between 'doy_a' and 'doy_b'.
    da_cond2 = da_pr.copy()
    da_cond2 = da_cond2.astype(bool)
    da_cond2[:, :, :] = True
    if (doy_a > -1) or (doy_b > -1):
        if doy_a == -1:
            doy_a = 1
        if doy_b == -1:
            doy_b = 365
        if doy_b >= doy_a:
            cond2 = (da_pr.time.dt.dayofyear >= doy_a) & (da_pr.time.dt.dayofyear <= doy_b)
        else:
            cond2 = (da_pr.time.dt.dayofyear <= doy_b) | (da_pr.time.dt.dayofyear >= doy_a)
        for t in range(n_t):
            da_cond2[t, :, :] = cond2[t]

    # Combine conditions.
    da_conds = da_cond1 & da_cond2

    # Calculate the number of dry days per year.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_idx = da_conds.astype(float).resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)

    return da_idx


def rain_season(da_pr: xr.DataArray, da_etp: xr.DataArray, da_rainstart_next: xr.DataArray, rs_pr_wet: float,
                rs_dt_wet: int, rs_doy_a: int, rs_doy_b: int, rs_pr_dry: float, rs_dt_dry: int, rs_dt_tot: int,
                re_method: str, re_pr: float, re_etp: float, re_dt: float, re_doy_a: int, re_doy_b: int)\
        -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate rain start, rain end, drain duration and rain quantity.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_etp : xr.DataArray
        Evapotranspiration data.
    da_rainstart_next : xr.DataArray
        First day of the next rain season.
    rs_pr_wet : float
        Daily precipitation amount required in first 'rs_dt_wet' days.
    rs_dt_wet: int
        Number of days with precipitation at season start (related to 'rs_pr_wet').
    rs_doy_a: int
        First day of year where season can start.
    rs_doy_b: int
        Last day of year where season can start.
    rs_pr_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    rs_dt_dry: int
        Maximum number of days in a dry period embedded into the period of 'rs_dt_tot' days.
    rs_dt_tot: int
        Number of days (after the first 'rs_dt_wet' days) after which a dry season is searched for.
    re_method : str
        Calculation method = {"depletion", "event"]
        The 'depletion' method is based on the period required for an amount of water (mm) 're_pr' to evaporate at a
        rate of 're_etp' (mm/day), considering that any amount of precipitation received during that period must
        evaporate as well.
        The 'event' method is based on the occurrence (or not) of an event preventing the end of the rain season. The
        rain season stops when no daily precipitation greater than 're_pr' have occurred over a period of 're_dt'
        days.
    re_pr : float
        If re_method == "Depletion": precipitation amount that must evaporate (mm).
        If re_method == "Event": last non-negligible precipitation event of the rain season (mm).
    re_etp: float
        If re_method == "Depletion": Evapotranspiration rate (mm/day).
    re_dt: float
        If re_method == "Event": period (number of days) during which there must not be a day with 're_pr'
        precipitation.
    re_doy_a: int
        First day of year at or after which the season ends.
    re_doy_b: int
        Last day of year at or before which the season ends.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Rename dimensions to have latitude and longitude dimensions.
    da_pr = utils.rename_dimensions(da_pr)
    da_etp = utils.rename_dimensions(da_etp)

    # Calculate rain start.
    # time1 = utils.get_current_time()
    da_rainstart =\
        xr.DataArray(rain_start(da_pr, rs_pr_wet, rs_dt_wet, rs_doy_a, rs_doy_b, rs_pr_dry, rs_dt_dry, rs_dt_tot))
    # da_rainstart = utils.rename_dimensions(da_rainstart)

    # Calculate rain end.
    # time2 = utils.get_current_time()
    da_rainend = xr.DataArray(rain_end(da_pr, da_etp, da_rainstart, da_rainstart_next, re_method, re_pr, re_etp, re_dt,
                                       re_doy_a, re_doy_b))
    da_rainend = utils.rename_dimensions(da_rainend)

    # Calculate rain duration.
    # time3 = utils.get_current_time()
    da_raindur = da_rainend - da_rainstart + 1
    da_raindur = utils.rename_dimensions(da_raindur)
    da_raindur.values[da_raindur.values < 0] = 0

    # Calculate rain quantity.
    # time4 = utils.get_current_time()
    da_rainqty = xr.DataArray(rain_qty(da_pr, da_rainstart, da_rainend))
    da_rainqty = utils.rename_dimensions(da_rainqty)

    # time5 = utils.get_current_time()

    return da_rainstart, da_rainend, da_raindur, da_rainqty


def rain_start_old(da_pr: xr.DataArray, pr_wet: float, dt_wet: int, doy_a: int, doy_b: int, pr_dry: float, dt_dry: int,
                   dt_tot: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the first day of the rain season.
    This algorithm is fast.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    pr_wet : float
        Daily precipitation amount required in first 'dt_wet' days.
    dt_wet: int
        Number of days with precipitation at season start (related to 'pr_wet').
    doy_a: int
        First day of year where season can start.
    doy_b: int
        Last day of year where season can start.
    pr_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    dt_dry: int
        Maximum number of days in a dry period embedded into the period of 'dt_tot' days.
    dt_tot: int
        Number of days (after the first 'dt_wet' days) after which a dry season is searched for.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    pr_wet = convert_units_to(str(pr_wet) + " mm/day", da_pr)
    pr_dry = convert_units_to(str(pr_dry) + " mm/day", da_pr)

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])

    # Condition #1: Flag the first day of each series of 'dt_wet' days with a total of 'pr_wet' in precipitation.
    da_cond1 = xr.DataArray(da_pr.rolling(time=dt_wet).sum() >= pr_wet)
    da_cond1[0:(n_t-dt_wet), :, :] = da_cond1[dt_wet:n_t, :, :].values
    da_cond1[(n_t-dt_wet):n_t] = False

    # Condition #2: Flag days that are not followed by a sequence of 'dt_dry' consecutive days over the next 'dt_wet' +
    # 'dt_tot' days. These days must also consider 'doy_a' and 'doy_b'.
    da_cond2 = da_cond1.copy()
    da_cond2[:, :, :] = True
    for t in range(n_t):
        if ((doy_a < 0) | ((doy_a >= 0) and (da_pr[t].time.dt.dayofyear >= doy_a))) and \
           ((doy_b < 0) | ((doy_b >= 0) and (da_pr[t].time.dt.dayofyear <= doy_b))) and \
           (t < n_t - dt_dry):
            t1 = t + dt_wet
            t2 = t1 + dt_tot
            da_t = (da_pr[t1:t2, :, :].max(dim=cfg.dim_time) >= pr_dry)
            da_cond2[t] = da_cond2[t] & da_t
        else:
            da_cond2[t] = False

    # Combine conditions.
    da_conds = da_cond1 & da_cond2

    # Obtain the first day of each year where conditions apply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_start = da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_start.values[(da_start.values < 0) | (da_start.values > 365)] = np.nan

    return da_start


def rain_start(da_pr: xr.DataArray, pr_wet: float, dt_wet: int, doy_a: int, doy_b: int, pr_dry: float, dt_dry: int,
               dt_tot: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the first day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    pr_wet : float
        Daily precipitation amount required in first 'dt_wet' days.
    dt_wet: int
        Number of days with precipitation at season start (related to 'pr_wet').
    doy_a: int
        First day of year where season can start.
    doy_b: int
        Last day of year where season can start.
    pr_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    dt_dry: int
        Maximum number of days in a dry period embedded into the period of 'dt_tot' days.
    dt_tot: int
        Number of days (after the first 'dt_wet' days) after which a dry season is searched for.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    pr_wet = convert_units_to(str(pr_wet) + " mm/day", da_pr)
    pr_dry = convert_units_to(str(pr_dry) + " mm/day", da_pr)

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])

    # Assign search boundaries.
    if (doy_a == -1) and (doy_b == -1):
        doy_a = 1
        doy_b = 365
    elif doy_a == -1:
        doy_a = 1 if doy_b == 365 else doy_b + 1
    elif doy_b == -1:
        doy_b = 365 if doy_a == 1 else doy_a - 1

    # Flag the first day of each sequence of 'dt_wet' days with a total of 'pr_wet' in precipitation
    # (assign True).
    da_onset = xr.DataArray(da_pr.rolling(time=dt_wet).sum() >= pr_wet)
    da_onset[0:(n_t-dt_wet)] = da_onset[dt_wet:n_t].values
    da_onset[(n_t-dt_wet):n_t] = False

    # Determine if it rains (assign 1) or not (assign 0).
    da_rainy_day = da_pr.copy()
    da_rainy_day.values[da_rainy_day.values < pr_dry] = 0
    da_rainy_day.values[da_rainy_day.values > 0] = 1

    # Flag the first day of each sequence of 'dt_dry' consecutive dry days (assign 1)
    da_dry_seq = xr.DataArray(da_rainy_day.rolling(time=dt_dry).sum() == 0)
    da_dry_seq[0:(n_t-dt_dry)] = da_dry_seq[dt_dry:n_t].values
    da_dry_seq[(n_t-dt_dry):n_t] = False
    da_dry_seq = da_dry_seq.astype(int)

    # Flag the first day of each sequence of fewer than 'dt_dry' consecutive dry days (assign True).
    da_no_dry_per = xr.DataArray(da_dry_seq.rolling(time=dt_tot).sum() < dt_dry)
    da_no_dry_per[0:(n_t-dt_dry)] = da_no_dry_per[dt_dry:n_t].values
    da_no_dry_per[(n_t-dt_dry):n_t] = False

    # Flag days between 'doy_a' and 'doy_b' (or the opposite).
    da_doy = xr.ones_like(da_pr).astype(bool)
    if doy_b >= doy_a:
        da_doy.values[(da_pr.time.dt.dayofyear < doy_a) | (da_pr.time.dt.dayofyear > doy_b)] = False
    else:
        da_doy.values[(da_pr.time.dt.dayofyear > doy_b) & (da_pr.time.dt.dayofyear < doy_a)] = False

    # Combine conditions.
    da_conds = da_onset & da_no_dry_per & da_doy

    # Obtain the first day of each year where conditions apply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_start = da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_start.values[(da_start.values < 0) | (da_start.values > 365)] = np.nan

    return da_start


def rain_end_old(da_pr: xr.DataArray, da_rainstart1: xr.DataArray, da_rainstart2: xr.DataArray, method: str, pr: float,
                 etp: float, dt: float, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_rainstart1 : xr.DataArray
        First day of the current rain season.
    da_rainstart2 : xr.DataArray
        First day of the next rain season.
    method : str
        Calculation method = {"depletion", "event"]
        If method == "depletion": based on the period required for an amount of water (mm) to evaporate, considering
        that any amount of precipitation received during that period must evaporate. The evapotranspiration rate is
        assumed to be 'etp' (mm/day).
        If method == "event": based on the occurrence (or not) of an event during the last days of a rain season.
        The rain season stops when no daily precipitation greater than 'pr' have occurred over a period of 'dt' days.
    pr : float
        If method == "depletion": precipitation amount that must evaporate (mm).
        If method == "event": threshold daily precipitation amount during a period (mm/day).
    etp: float
        If method == "depletion": evapotranspiration rate (mm/day).
        Otherwise: not used.
    dt: float
        If method == "event": length of period (number of days) used to verify if the rain season is ending.
        Otherwise: not used.
    doy_a: int
        First day of year at or after which the season can end.
    doy_b: int
        Last day of year at or before which the season can end.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Rename coordinates.
    if (cfg.dim_rlat in list(da_pr.dims)) or (cfg.dim_rlon in list(da_pr.dims)):
        da_pr = da_pr.rename({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})

    # Unit conversion.
    pr  = convert_units_to(str(pr) + " mm/day", da_pr)
    etp = convert_units_to(str(etp) + " mm/day", da_pr)

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])

    # Provide default values.
    if doy_a == -1:
        doy_a = 1
    if doy_b == -1:
        doy_b = 365

    # Depletion method -------------------------------------------------------------------------------------------------

    if method == "depletion":

        # DataArray that will hold results (one value per year).
        # Only the resulting structure is needed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_end = da_pr.resample(time=cfg.freq_YS).min(dim=cfg.dim_time)
            da_end[:, :, :] = -1

        # Calculate the minimum number of days that is required for evapotranspiration (assuming no rain).
        n_et = int(pr / etp)

        # Loop through combinations of intervals.
        da_end_y = None
        t1_prev_y = -1
        t_first_doy = 0
        for t1 in range(n_t - n_et):

            # Day of year and year of 't1'.
            t1_doy = int(da_pr[t1].time.dt.dayofyear.values)
            t1_y = int(da_pr[t1].time.dt.year.values)

            # Initialize the array that will hold annual results.
            if (da_end_y is None) or (t1_y != t1_prev_y):
                da_end_y = da_end[da_end.time.dt.year == t1_y].squeeze().copy()
                t_first_doy = t1
            t1_prev_y = t1_y

            # Determine the range of cells to evaluate.
            t2_min = max(t_first_doy + doy_a - 1, t1 + n_et)
            if doy_a <= doy_b:
                t2_max = min(t_first_doy + doy_b - 1, t2_min + 365 - t1_doy - n_et + 1)
            else:
                t2_max = min(t2_min + (365 - doy_a - 1) + doy_b - 1, n_t)

            # Loop through ranges.
            for t2 in range(t2_min, t2_max):

                # Day of year and year of 't1'.
                t2_doy = int(da_pr[t2].time.dt.dayofyear.values)
                t2_y = int(da_pr[t2].time.dt.year.values)

                # Examine the current 't1'-'t2' combination.
                if (doy_a == doy_b) or\
                   ((doy_a < doy_b) and (t1_doy >= doy_a) and (t2_doy <= doy_b)) or\
                   ((doy_a > doy_b) and (t1_y == t2_y) and (t2_doy <= doy_a)) or\
                   ((doy_a > doy_b) and (t1_y < t2_y) and (t2_doy <= doy_b)):
                    da_t1t2 = (da_pr[t1:t2, :, :].sum(dim=cfg.dim_time) - (t2 - t1 + 1) * etp)
                    da_better     = (da_t1t2 < -pr) & ((da_end_y == -1) | (t2_doy < da_end_y))
                    da_not_better = (da_t1t2 >= -pr) | ((da_end_y == -1) | (t2_doy >= da_end_y))
                    da_end_y = (da_better * t2_doy) + (da_not_better * da_end_y)

            da_end[da_end.time.dt.year == t1_y] = da_end_y

    # Event method -----------------------------------------------------------------------------------------------------

    else:

        # Combined conditions.
        da_conds = da_pr.copy().astype(float)
        da_conds[:, :, :] = 0

        # Extract year and day of year.
        year_l = utils.extract_date_field(da_pr, "year")
        doy_l = utils.extract_date_field(da_pr, "doy")

        # Calculate conditions at each time step.
        for t in range(n_t):

            year     = year_l[t]
            doy      = doy_l[t]

            # Condition #1: Rain season ends within imposed boundaries (doy_a and doy_b).
            case_1 = (doy_a <= doy_b) & (doy >= doy_a) & (doy <= doy_b)
            case_2 = (doy_a > doy_b) & (doy >= doy_a)
            case_3 = (doy_a > doy_b) & (doy <= doy_b)
            cond1 = case_1 | case_2 | case_3

            # Condition #2a: Rain season can't stop before it begins.
            da_rainstart1_t = da_rainstart1[da_rainstart1[cfg.dim_time].dt.year == year].squeeze()
            da_cond2a = ((case_1 | case_2) & (doy >= da_rainstart1_t)) | (case_3 & (doy <= da_rainstart1_t))
            # Condition #2b: Rain season can't stop after the beginning of the following rain season.
            da_cond2b = True
            if da_rainstart2 is not None:
                da_rainstart2_t = da_rainstart2[da_rainstart2[cfg.dim_time].dt.year == year].squeeze()
                da_cond2b = ((case_1 | case_3) & (doy <= da_rainstart2_t)) | (case_2 & (doy >= da_rainstart2_t))

            # Condition #3: Current precipitation exceeds threshold.
            da_cond3 = da_pr[t] >= pr

            # Condition #4: No precipitation exceeding threshold in the next 'dt' days.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_cond4 = (da_pr[(t+1):(t+int(dt)+1)].max(dim=cfg.dim_time) < pr) if (t < n_t - int(dt) - 1) else False

            # Combine conditions.
            da_conds[t] = xr.DataArray(cond1 & da_cond2a & da_cond2b & da_cond3 & da_cond4)
            da_conds[t].values[da_conds[t].values == 0] = np.nan
            da_conds[t] = da_conds[t].astype(float) * doy

        # Summarize by year.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_end = da_conds.resample(time=cfg.freq_YS).min(dim=cfg.dim_time)

        # Impose an end day if there is a start day. It can be associated with the start of the following rain season or
        # 'doy_b'.
        for t in range(len(da_end[cfg.dim_time])):
            if da_rainstart2 is not None:
                sel = (np.isnan(da_end[t].values)) & (np.isnan(da_rainstart1[t].values).astype(int) == 0)
                da_end[t].values[sel] = da_rainstart2[t].values[sel] - 1
            da_end[t].values[da_end[t].values >= doy_b] = doy_b - 1

    return da_end


def rain_end(da_pr: xr.DataArray, da_etp: xr.DataArray, da_rainstart: xr.DataArray, da_rainstart_next: xr.DataArray,
             method: str, pr: float, etp: float, dt: float, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_etp : xr.DataArray
        Evapotranspiration data.
    da_rainstart : xr.DataArray
        First day of the current rain season.
    da_rainstart_next : xr.DataArray
        First day of the next rain season.
    method : str
        Calculation method = {"depletion", "event", "total"]
        If method == "depletion": based on the period required for an amount of water (mm) to evaporate, considering
        that any amount of precipitation received during that period must evaporate. If 'da_etp' is not available, the
        evapotranspiration rate is assumed to be 'etp' (mm/day).
        If method == "event": based on the occurrence (or not) of an event during the last days of a rain season.
        The rain season stops when no daily precipitation greater than 'pr' have occurred over a period of 'dt' days.
        If method == "total": based on a total amount of precipitation received during the last days of the rain season.
        The rain season stops when the total amount of precipitation is less than 'pr' over a period of 'dt' days.
    pr : float
        If method == "depletion": precipitation amount that must evaporate (mm).
        If method == "event": threshold daily precipitation amount during a period (mm/day).
        If method == "total": threshold total precipitation amount over a period (mm).
    etp: float
        If method == "depletion": evapotranspiration rate (mm/day).
        Otherwise: not used.
    dt: float
        If method in ["event", "total"]: length of period (number of days) used to verify if the rain season is ending.
        Otherwise: not used.
    doy_a: int
        First day of year at or after which the season can end.
    doy_b: int
        Last day of year at or before which the season can end.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0
    if da_etp is not None:
        da_etp.values[da_etp.values < 0] = 0

    # Rename coordinates.
    if (cfg.dim_rlat in list(da_pr.dims)) or (cfg.dim_rlon in list(da_pr.dims)):
        da_pr = da_pr.rename({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})
        if da_etp is not None:
            da_etp = da_etp.rename({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})

    # Unit conversion.
    pr  = convert_units_to(str(pr) + " mm/day", da_pr)
    etp = convert_units_to(str(etp) + " mm/day", da_pr)

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])

    # Assign search boundaries.
    if (doy_a == -1) and (doy_b == -1):
        doy_a = 1
        doy_b = 365
    elif doy_a == -1:
        doy_a = 1 if doy_b == 365 else doy_b + 1
    elif doy_b == -1:
        doy_b = 365 if doy_a == 1 else doy_a - 1

    # Flag days between 'doy_a' and 'doy_b' (or the opposite).
    da_doy = xr.ones_like(da_pr).astype(bool)
    if doy_b >= doy_a:
        da_doy.values[(da_pr.time.dt.dayofyear < doy_a) |
                      (da_pr.time.dt.dayofyear > doy_b)] = False
    else:
        da_doy.values[(da_pr.time.dt.dayofyear > doy_b) &
                      (da_pr.time.dt.dayofyear < doy_a)] = False

    # Depletion method -------------------------------------------------------------------------------------------------

    if method == "depletion":

        da_rainend = None

        # Calculate the minimum length of the period.
        dt_min = math.ceil(pr / etp)
        dt_max = (doy_b - doy_a + 1 if doy_b >= doy_a else (365 - doy_a + 1 + doy_b)) - dt_min

        for dt in list(range(dt_min, dt_max + 1)):

            # Flag the day before each sequence of 'dt' days that results in evaporating a water column, considering
            # precipitation falling during this period (assign 1).
            if da_etp is None:
                da_dry_seq = xr.DataArray((da_pr.rolling(time=int(dt)).sum() + pr) < (dt * etp))
            else:
                da_dry_seq = xr.DataArray((da_pr.rolling(time=int(dt)).sum() + pr) < da_etp.rolling(time=int(dt)).sum())
            da_dry_seq[0:(n_t - int(dt) - 1)] = da_dry_seq[(int(dt) + 1):n_t].values
            da_dry_seq[(n_t - int(dt) - 1):n_t] = False

            # Combine conditions.
            da_conds = da_dry_seq & da_doy

            # Obtain the first day of each year where conditions apply.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_rainend_i = \
                    da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")

            # Update the cells that were not assigned yet.
            if da_rainend is None:
                da_rainend = da_rainend_i.copy()
            else:
                sel = (np.isnan(da_rainend.values)) &\
                      ((np.isnan(da_rainend_i.values).astype(int) == 0) | (da_rainend_i.values < da_rainend.values))
                da_rainend.values[sel] = da_rainend_i.values[sel]

            # Exit loop if all cells were assigned a value.
            if np.isnan(da_rainend).astype(int).sum() == 0:
                break

    # Event method -----------------------------------------------------------------------------------------------------

    elif method in ["event", "total"]:

        # Determine if it rains heavily (assign 1) or not (assign 0).
        da_heavy_rain = da_pr.copy()
        da_heavy_rain.values[da_heavy_rain.values < pr] = 0
        da_heavy_rain.values[da_heavy_rain.values > 0] = 1

        # Flag the day (assign 1) before each sequence of:
        # 'dt' days with no amount reaching 'pr':
        if method == "event":
            da_dry_seq = xr.DataArray(da_heavy_rain.rolling(time=int(dt)).sum() == 0)
        # 'dt' days with a total amount reaching 'pr':
        else:
            da_dry_seq = xr.DataArray(da_pr.rolling(time=int(dt)).sum() < pr)

        da_dry_seq[0:(n_t - int(dt) - 1)] = da_dry_seq[(int(dt) + 1):n_t].values
        da_dry_seq[(n_t - int(dt) - 1):n_t] = False

        # Combine conditions.
        da_conds = da_dry_seq & da_doy

        # Obtain the first day of each year where conditions apply.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_rainend =\
                da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")

    # Adjust or discard rain end values that are not compatible with the current or next season start values.
    # If the season ends before or on start day, discard rain end.
    if da_rainstart is not None:
        sel = (np.isnan(da_rainstart.values).astype(int) == 0) &\
              (np.isnan(da_rainend.values).astype(int) == 0) & \
              (da_rainend.values <= da_rainstart.values)
        da_rainend.values[sel] = np.nan
    # If the season ends after or on start day of the next season, the end day of the current season becomes the day
    # before the next season.
    if da_rainstart_next is not None:
        sel = (np.isnan(da_rainstart_next.values).astype(int) == 0) &\
              (np.isnan(da_rainend.values).astype(int) == 0) & \
              (da_rainend.values >= da_rainstart_next.values)
        da_rainend.values[sel] = da_rainstart_next.values[sel] - 1
        da_rainend.values[da_rainend.values < 1] = 365

    return da_rainend


def rain_qty_old(da_pr: xr.DataArray, da_rainstart: xr.DataArray, da_rainend: xr.DataArray) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_rainstart : xr.DataArray
        Rain start (first day of year).
    da_rainend: xr.DataArray
        Rain end (last day of year).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Convert units.
    if da_pr.attrs[cfg.attrs_units] == cfg.unit_kg_m2s1:
        da_pr.values *= cfg.spd

    # Rename coordinates.
    if (cfg.dim_rlat in list(da_pr.dims)) or (cfg.dim_rlon in list(da_pr.dims)):
        da_pr = da_pr.rename({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})

    # Extract years.
    years_idx = utils.extract_date_field(da_rainstart.time, "year")

    # Discard precipitation amounts that are not happening during rain season.
    n_t = len(da_pr[cfg.dim_time])
    doy_prev = 365
    for t in range(n_t):

        # Extract year and day of year.
        y = int(da_pr[cfg.dim_time][t].dt.year)
        doy = int(da_pr[cfg.dim_time][t].dt.dayofyear)

        # Extract start and end days of rain season.
        if doy < doy_prev:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                da_start = da_rainstart[np.array(years_idx) == str(y)].squeeze()
                da_end = da_rainend[np.array(years_idx) == str(y)].squeeze()

        # Condition.
        da_cond = (da_end > da_start) &\
                  (((da_start <= da_end) & (doy >= da_start) & (doy <= da_end)) |
                   ((da_start > da_end) & ((doy <= da_start) | (doy >= da_end))))

        # Discard values.
        da_pr[t] = da_pr[t] * da_cond.astype(float)

        doy_prev = doy

    # Sum by year.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_qty = da_pr.resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)

    return da_qty


def rain_qty(da_pr: xr.DataArray, da_rainstart: xr.DataArray, da_rainend: xr.DataArray) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray
        Precipitation data.
    da_rainstart : xr.DataArray
        Rain start (first day of year).
    da_rainend: xr.DataArray
        Rain end (last day of year).
    --------------------------------------------------------------------------------------------------------------------
    """

    da_qty = indices_gen.aggregate_between_dates(da_pr, da_rainstart, da_rainend, "sum") * cfg.spd

    return da_qty


def wind_days_above(da_vv: xr.DataArray, da_dd: xr.DataArray, param_vv: float, param_dd: float,
                    param_dd_tol: float, doy_a: int, doy_b: int) -> xr.DataArray:

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
        Parameter related to 'da_windfromdir' (degrees).
    param_dd_tol: float
        Parameter tolerance related to 'da_windfromdir' (degrees).
    doy_a: int
        First day of year at or after which the season ends.
    doy_b: int
        Last day of year at or before which the season ends.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Condition #1: Subset based on 'day of year'.
    da_vv = utils.subset_doy(da_vv, doy_a, doy_b)
    da_dd = utils.subset_doy(da_dd, doy_a, doy_b)

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
        da_wgdaysabove = da_conds.resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)

    return da_wgdaysabove


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
