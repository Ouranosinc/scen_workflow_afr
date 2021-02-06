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
import multiprocessing
import numpy as np
import os.path
import statistics
import utils
import xarray as xr
import xclim.indices as indices
import xclim.indices.generic as indices_gen
import warnings
from typing import Union, List
from xclim.indices import run_length as rl
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to


def create_mask(stn: str) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a mask, based on climate scenarios for the temperature or precipitation variable.
    All values with a value are attributed a value of 1. Other values are assigned 'nan'.

    Parameters
    ----------
    stn : str
        Station name.
    --------------------------------------------------------------------------------------------------------------------
    """

    da_mask = None

    f_list = glob.glob(cfg.get_d_scen(stn, cfg.cat_obs) + "*/*" + cfg.f_ext_nc)
    for i in range(len(f_list)):

        # Open NetCDF file.
        ds = utils.open_netcdf(f_list[i])
        var = list(ds.data_vars)[0]
        if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:

            # Flag 'nan' values.
            # if var == cfg.var_cordex_pr:
            #     p_dry_error = convert_units_to(str(0.0000008) + " mm/day", ds[var])
            #     ds[var].values[(ds[var].values > 0) & (ds[var].values <= p_dry_error)] = np.nan

            # Create mask.
            da_mask = ds[var][0] * 0 + 1

            break

    return da_mask


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
    var_or_idx_list = []

    # Temperature.
    if idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights, cfg.idx_tngmonthsbelow, cfg.idx_heatwavemaxlen,
                    cfg.idx_heatwavetotlen, cfg.idx_tgg, cfg.idx_etr]:
        var_or_idx_list.append(cfg.var_cordex_tasmin)

    if idx_name in [cfg.idx_tx90p, cfg.idx_txdaysabove, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_txg,
                    cfg.idx_txx, cfg.idx_wsdi, cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen, cfg.idx_tgg,
                    cfg.idx_etr]:
        var_or_idx_list.append(cfg.var_cordex_tasmax)

    # Precipitation.
    if idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_cwd, cfg.idx_cdd, cfg.idx_sdii, cfg.idx_prcptot,
                    cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_rainstart,
                    cfg.idx_rainend, cfg.idx_rainqty, cfg.idx_drydurtot]:
        var_or_idx_list.append(cfg.var_cordex_pr)

    if idx_name in [cfg.idx_rainend, cfg.idx_raindur, cfg.idx_rainqty]:
        var_or_idx_list.append(idx_code.replace(idx_name, cfg.idx_rainstart))

        if (idx_name == cfg.idx_rainend) and (str(idx_params[6]) != "nan"):
            var_or_idx_list.append(str(idx_params[6]))

        if idx_name in [cfg.idx_raindur, cfg.idx_rainqty]:
            var_or_idx_list.append(idx_code.replace(idx_name, cfg.idx_rainend))

    # Temperature-precipitation.
    if idx_name == cfg.idx_dc:
        var_or_idx_list.append(cfg.var_cordex_tas)
        var_or_idx_list.append(cfg.var_cordex_pr)

    # Wind.
    if idx_name == cfg.idx_wgdaysabove:
        var_or_idx_list.append(cfg.var_cordex_uas)
        var_or_idx_list.append(cfg.var_cordex_vas)
    elif idx_name == cfg.idx_wxdaysabove:
        var_or_idx_list.append(cfg.var_cordex_sfcwindmax)

    # ==================================================================================================================
    # TODO.CUSTOMIZATION.INDEX.END
    # ==================================================================================================================

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Verify if this variable or index is available for the current station.
        utils.log("Verifying data availability (based on directories).", True)
        var_or_idx_list_avail = True
        for var_or_idx in var_or_idx_list:
            if ((cfg.extract_idx(var_or_idx) in cfg.variables_cordex) and
                not os.path.isdir(cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx))) or\
               ((cfg.extract_idx(var_or_idx) not in cfg.variables_cordex) and
                not os.path.isdir(cfg.get_d_idx(stn, var_or_idx))):
                var_or_idx_list_avail = False
                break
        if not var_or_idx_list_avail:
            continue

        # Create mask.
        da_mask = None
        if stn == cfg.obs_src_era5_land:
            da_mask = create_mask(stn)

        # Loop through emissions scenarios.
        for rcp in rcps:

            utils.log("Processing: '" + idx_code + "', '" + stn + "', '" + cfg.get_rcp_desc(rcp) + "'", True)

            # List simulation files for the first variable. As soon as there is no file for one variable, the analysis
            # for the current RCP needs to abort.
            utils.log("Collecting simulation files.", True)
            if rcp == cfg.rcp_ref:
                if cfg.extract_idx(var_or_idx_list[0]) in cfg.variables_cordex:
                    p_sim = cfg.get_d_scen(stn, cfg.cat_obs, var_or_idx_list[0]) +\
                            cfg.extract_idx(var_or_idx_list[0]) + "_" + stn + cfg.f_ext_nc
                else:
                    p_sim = cfg.get_d_idx(cfg.obs_src, var_or_idx_list[0]) +\
                            cfg.extract_idx(var_or_idx_list[0]) + "_ref" + cfg.f_ext_nc
                if os.path.exists(p_sim) and (type(p_sim) is str):
                    p_sim = [p_sim]
            else:
                if cfg.extract_idx(var_or_idx_list[0]) in cfg.variables_cordex:
                    d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx_list[0])
                else:
                    d = cfg.get_d_idx(stn, var_or_idx_list[0])
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
            if len(var_or_idx_list) > 1:
                p_sim_fix = []
                for p_sim_i in p_sim:
                    missing = False
                    for var_or_idx_j in var_or_idx_list[1:]:
                        p_sim_j = cfg.get_equivalent_idx_path(p_sim_i, var_or_idx_list[0], var_or_idx_j, stn, rcp)
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
                    generate_single(idx_code, idx_params, var_or_idx_list, p_sim, stn, rcp, da_mask, i_sim)

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
                            generate_single(idx_code, idx_params, var_or_idx_list, p_sim, stn, rcp, da_mask, i_sim)

                    # Parallel processing mode.
                    else:

                        try:
                            utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                            pool = multiprocessing.Pool(processes=min(cfg.n_proc, n_sim))
                            func = functools.partial(generate_single, idx_code, idx_params, var_or_idx_list, p_sim,
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


def generate_single(idx_code: str, idx_params, var_or_idx_list: [str], p_sim: [str], stn: str, rcp: str,
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
    var_or_idx_list : [str]
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
                os.path.basename(p_sim[i_sim]).replace(cfg.extract_idx(var_or_idx_list[0]), idx_name)

    # Exit loop if the file already exists (simulations files only; not reference file).
    if (rcp != cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite):
        return

    # Load datasets (one per variable or index).
    ds_var_or_idx = []
    for i_var_or_idx in range(0, len(var_or_idx_list)):
        var_or_idx_i = var_or_idx_list[i_var_or_idx]

        # Open dataset.
        p_sim_j = cfg.get_equivalent_idx_path(p_sim[i_sim], var_or_idx_list[0], var_or_idx_i, stn, rcp)
        ds = utils.open_netcdf(p_sim_j)

        # Adjust temperature units.
        if cfg.extract_idx(var_or_idx_i) in\
                [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
            if ds[var_or_idx_i].attrs[cfg.attrs_units] == cfg.unit_K:
                ds[var_or_idx_i] = ds[var_or_idx_i] - cfg.d_KC
            elif rcp == cfg.rcp_ref:
                ds[var_or_idx_i][cfg.attrs_units] = cfg.unit_C
            ds[var_or_idx_i].attrs[cfg.attrs_units] = cfg.unit_C

        # Apply mask.
        if cfg.extract_idx(var_or_idx_i) not in cfg.variables_cordex:
            ds[var_or_idx_i] = utils.apply_mask(ds[cfg.extract_idx(var_or_idx_i)], da_mask)

        ds_var_or_idx.append(ds)

    # ======================================================================================================
    # TODO.CUSTOMIZATION.INDEX.BEGIN
    # Calculate indices.
    # ======================================================================================================

    idx_units = None

    # Calculate the 90th percentile of tasmax for the reference period.
    if (idx_name == cfg.idx_wsdi) and (rcp == cfg.rcp_ref):
        da_tx90p = percentile_doy(ds_var_or_idx[0][cfg.var_cordex_tasmax], per=0.9)

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
                        idx_param = ds_var_or_idx[i][cfg.var_cordex_tasmax].quantile(idx_param).values.ravel()[0]
                    elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
                        idx_param = ds_var_or_idx[i][cfg.var_cordex_tasmin].quantile(idx_param).values.ravel()[0]
                    elif idx_name == cfg.idx_prcptot:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            da_i = ds_var_or_idx[i][cfg.var_cordex_pr].resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)
                        dim = utils.get_coord_names(ds_var_or_idx[i])
                        idx_param =\
                            da_i.sum(dim=dim).quantile(idx_param).values.ravel()[0] / float(da_mask.sum()) * cfg.spd
                    elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
                        if idx_name == cfg.idx_wgdaysabove:
                            da_uas = ds_var_or_idx[0][cfg.var_cordex_uas]
                            da_vas = ds_var_or_idx[1][cfg.var_cordex_vas]
                            da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas)
                        else:
                            da_vv = ds_var_or_idx[0][cfg.var_cordex_sfcwindmax]
                        idx_param = da_vv.quantile(idx_param).values.ravel()[0]
                    # Round value and save it.
                    idx_param = float(round(idx_param, 2))
                    cfg.idx_params[cfg.idx_names.index(idx_name)][i] = idx_param
                else:
                    idx_param = cfg.idx_params[cfg.idx_names.index(idx_name)][i]

        # Combine threshold and unit -----------------------------------------------------------------------

        if (idx_name in [cfg.idx_txdaysabove, cfg.idx_tx90p, cfg.idx_tropicalnights]) or\
           ((idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]) and (i == 0)) or \
           ((idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]) and (i <= 1)):
            idx_ref = str(idx_param) + " " + cfg.unit_C
            idx_fut = str(idx_param + cfg.d_KC) + " " + cfg.unit_K
            idx_params_str.append(idx_ref if (rcp == cfg.rcp_ref) else idx_fut)

        elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                          cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:
            idx_params_str.append(str(idx_param) + " mm/day")

        elif (idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 1):
            idx_params_str.append(str(idx_param) + " " + cfg.unit_m_s1)

        elif not ((idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 4)):
            idx_params_str.append(str(idx_param))

        # Split lists --------------------------------------------------------------------------------------

        if (idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 4):
            if str(idx_params[i]) == "nan":
                idx_params[i] = str(list(range(1, 13))).replace("'", "")
            items = idx_params[i].replace("[", "").replace(", ", ";").replace("]", "").split(";")
            idx_params_str.append([int(i) for i in items])

    # Exit loop if the file already exists (reference file only).
    if not ((rcp == cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite)):

        # Temperature --------------------------------------------------------------------------------------

        da_idx = None
        if idx_name in [cfg.idx_txdaysabove, cfg.idx_tx90p]:
            da_tasmax = ds_var_or_idx[0][cfg.var_cordex_tasmax]
            param_tasmax = idx_params_str[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.tx_days_above(da_tasmax, param_tasmax).values)
            da_idx = da_idx.astype(int)
            idx_units = cfg.unit_1

        elif idx_name == cfg.idx_tngmonthsbelow:
            da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
            param_tasmin = float(idx_params_str[0])
            if da_tasmin.attrs[cfg.attrs_units] != cfg.unit_C:
                param_tasmin += cfg.d_KC
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.tn_mean(da_tasmin, freq=cfg.freq_MS))
                da_idx = xr.DataArray(indices_gen.threshold_count(da_idx, "<", param_tasmin, cfg.freq_YS))
            da_idx = da_idx.astype(float)
            idx_units = cfg.unit_1

        elif idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]:
            da_tasmax = ds_var_or_idx[0][cfg.var_cordex_tasmax]
            param_tasmax = idx_params_str[0]
            param_ndays = int(float(idx_params_str[1]))
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
            idx_units = cfg.unit_1

        elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
            da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
            da_tasmax = ds_var_or_idx[1][cfg.var_cordex_tasmax]
            param_tasmin = idx_params_str[0]
            param_tasmax = idx_params_str[1]
            window = int(float(idx_params_str[2]))
            if idx_name == cfg.idx_heatwavemaxlen:
                da_idx = xr.DataArray(
                    heat_wave_max_length(da_tasmin, da_tasmax, param_tasmin, param_tasmax, window).values)
            else:
                da_idx = xr.DataArray(heat_wave_total_length(
                    da_tasmin, da_tasmax, param_tasmin, param_tasmax, window).values)
            da_idx = da_idx.astype(int)
            idx_units = cfg.unit_1

        elif idx_name in [cfg.idx_txg, cfg.idx_txx]:
            da_tasmax = ds_var_or_idx[0][cfg.var_cordex_tasmax]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx_name == cfg.idx_txg:
                    da_idx = xr.DataArray(indices.tx_mean(da_tasmax))
                else:
                    da_idx = xr.DataArray(indices.tx_max(da_tasmax))
            idx_units = cfg.unit_C

        elif idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights]:
            da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
            if idx_name in [cfg.idx_tnx, cfg.idx_tng]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    if idx_name == cfg.idx_tnx:
                        da_idx = xr.DataArray(indices.tn_max(da_tasmin))
                        idx_units = cfg.unit_C
                    elif idx_name == cfg.idx_tng:
                        da_idx = xr.DataArray(indices.tn_mean(da_tasmin))
                        idx_units = cfg.unit_C
            else:
                param_tasmin = idx_params_str[0]
                da_idx = xr.DataArray(indices.tropical_nights(da_tasmin, param_tasmin))
                idx_units = cfg.unit_1

        elif idx_name in [cfg.idx_tgg, cfg.idx_etr]:
            da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
            da_tasmax = ds_var_or_idx[1][cfg.var_cordex_tasmax]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx_name == cfg.idx_tgg:
                    da_idx = xr.DataArray(indices.tg_mean(indices.tas(da_tasmin, da_tasmax)))
                else:
                    da_idx = xr.DataArray(indices.tx_max(da_tasmax) - indices.tn_min(da_tasmin))
            idx_units = cfg.unit_C

        elif idx_name == cfg.idx_dc:
            da_tas = ds_var_or_idx[0][cfg.var_cordex_tas]
            da_pr  = ds_var_or_idx[1][cfg.var_cordex_pr]
            da_lon, da_lat = utils.get_coordinates(ds_var_or_idx[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_idx = xr.DataArray(indices.drought_code(
                    da_tas, da_pr, da_lat, shut_down_mode="temperature")).resample(time=cfg.freq_YS).mean()
            idx_units = cfg.unit_1

        # Precipitation ------------------------------------------------------------------------------------

        elif idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_prcptot]:
            da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                if idx_name == cfg.idx_rx1day:
                    da_idx = xr.DataArray(indices.max_1day_precipitation_amount(da_pr, cfg.freq_YS))
                elif idx_name == cfg.idx_rx5day:
                    da_idx = xr.DataArray(indices.max_n_day_precipitation_amount(da_pr, 5, cfg.freq_YS))
                else:
                    da_idx = xr.DataArray(indices.precip_accumulation(da_pr, freq=cfg.freq_YS))
            idx_units = da_idx.attrs[cfg.attrs_units]

        elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                          cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:
            da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
            param_pr = idx_params_str[0]
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
            elif idx_name == cfg.idx_sdii:
                da_idx = xr.DataArray(indices.daily_pr_intensity(da_pr, param_pr))
            da_idx = da_idx.astype(int)
            idx_units = cfg.unit_1

        elif idx_name == cfg.idx_rainstart:
            da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
            pr_wet = float(idx_params_str[0])
            dt_wet = int(idx_params_str[1])
            doy   = int(idx_params_str[2])
            pr_dry = float(idx_params_str[3])
            dt_dry = int(idx_params_str[4])
            dt_tot = int(idx_params_str[5])
            da_idx = xr.DataArray(rain_start(da_pr, pr_wet, dt_wet, doy, pr_dry, dt_dry, dt_tot))
            idx_units = cfg.unit_1

        elif idx_name == cfg.idx_rainend:
            da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
            da_rainstart1 = ds_var_or_idx[1][cfg.idx_rainstart]
            da_rainstart2 = None
            if len(ds_var_or_idx) > 2:
                da_rainstart2 = ds_var_or_idx[2][cfg.idx_rainstart]
            meth  = idx_params_str[0]
            pr    = float(idx_params_str[1])
            etp   = -1.0 if str(idx_params_str[2]) == "nan" else float(idx_params_str[2])
            dt    = -1.0 if str(idx_params_str[3]) == "nan" else float(idx_params_str[3])
            doy_a = int(idx_params_str[4])
            doy_b = int(idx_params_str[5])
            da_idx = xr.DataArray(rain_end(da_pr, da_rainstart1, da_rainstart2, meth, pr, etp, dt, doy_a, doy_b))
            idx_units = cfg.unit_1

        elif idx_name == cfg.idx_raindur:
            da_rainstart = ds_var_or_idx[0][cfg.idx_rainstart]
            da_rainend   = ds_var_or_idx[1][cfg.idx_rainend]
            da_idx = da_rainend - da_rainstart
            da_idx.values[da_idx.values < 0] = 0
            idx_units = cfg.unit_1

        elif idx_name == cfg.idx_rainqty:
            da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
            da_rainstart = ds_var_or_idx[1][cfg.idx_rainstart]
            da_rainend = ds_var_or_idx[2][cfg.idx_rainend]
            da_idx = xr.DataArray(rain_qty(da_pr, da_rainstart, da_rainend))
            idx_units = cfg.get_unit(idx_name)

        elif idx_name == cfg.idx_drydurtot:
            da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
            p_dry = float(idx_params_str[0])
            d_dry = int(idx_params_str[1])
            per   = idx_params_str[2]
            doy_a = idx_params_str[3]
            doy_a = None if (str(doy_a) == "nan") else int(doy_a)
            doy_b = idx_params_str[4]
            doy_b = None if (str(doy_b) == "nan") else int(doy_b)
            da_idx = xr.DataArray(tot_duration_dry_periods(da_pr,  p_dry, d_dry, per, doy_a, doy_b))
            idx_units = cfg.unit_1

        # Wind ---------------------------------------------------------------------------------------------

        elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
            param_vv     = float(idx_params_str[0])
            param_vv_neg = idx_params_str[1]
            param_dd     = float(idx_params_str[2])
            param_dd_tol = float(idx_params_str[3])
            param_months = idx_params_str[4]
            if idx_name == cfg.idx_wgdaysabove:
                da_uas = ds_var_or_idx[0][cfg.var_cordex_uas]
                da_vas = ds_var_or_idx[1][cfg.var_cordex_vas]
                da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas, param_vv_neg)
            else:
                da_vv = ds_var_or_idx[0][cfg.var_cordex_sfcwindmax]
                da_dd = None
            da_idx = xr.DataArray(
                wind_days_above(da_vv, da_dd, param_vv, param_dd, param_dd_tol, param_months))
            idx_units = cfg.unit_1

        da_idx.attrs[cfg.attrs_units] = idx_units

        # ======================================================================================================
        # TODO.CUSTOMIZATION.INDEX.END
        # ======================================================================================================

        # Convert to float. This is required to ensure that 'nan' values are not transformed into integers.
        da_idx = da_idx.astype(float)

        # Rename dimensions.
        if "dim_0" in list(da_idx.dims):
            da_idx = da_idx.rename({"dim_0": cfg.dim_time})
            da_idx = da_idx.rename({"dim_1": cfg.dim_latitude, "dim_2": cfg.dim_longitude})
        elif (cfg.dim_lat in list(da_idx.dims)) or (cfg.dim_lon in list(da_idx.dims)):
            da_idx = da_idx.rename({cfg.dim_lat: cfg.dim_latitude, cfg.dim_lon: cfg.dim_longitude})
        elif (cfg.dim_rlat in list(da_idx.dims)) or (cfg.dim_rlon in list(da_idx.dims)):
            da_idx = da_idx.rename({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})
        elif (cfg.dim_latitude not in list(da_idx.dims)) and (cfg.dim_longitude not in list(da_idx.dims)):
            da_idx = da_idx.expand_dims(longitude=1)
            da_idx = da_idx.expand_dims(latitude=1)

        # Interpolate (to remove nan values).
        if np.isnan(da_idx).astype(int).max() > 0:
            da_idx = utils.interpolate_na_fix(da_idx)

        # Apply mask.
        if da_mask is not None:
            da_idx = utils.apply_mask(da_idx, da_mask)

        # Create dataset.
        da_idx.name = idx_name
        ds_idx = da_idx.to_dataset()
        ds_idx.attrs[cfg.attrs_units] = idx_units
        ds_idx.attrs[cfg.attrs_sname] = idx_name
        ds_idx.attrs[cfg.attrs_lname] = idx_name
        ds_idx = utils.copy_coordinates(ds_var_or_idx[0], ds_idx)
        ds_idx[idx_name] =\
            utils.copy_coordinates(ds_var_or_idx[0][cfg.extract_idx(var_or_idx_list[0])], ds_idx[idx_name])
        ds_idx = ds_idx.squeeze()

        # Adjust calendar.
        year_1 = cfg.per_fut[0]
        year_n = cfg.per_fut[1]
        if rcp == cfg.rcp_ref:
            year_1 = max(cfg.per_ref[0], int(str(ds_var_or_idx[0][cfg.dim_time][0].values)[0:4]))
            year_n = min(cfg.per_ref[1], int(str(ds_var_or_idx[0][cfg.dim_time]
                                                 [len(ds_var_or_idx[0][cfg.dim_time]) - 1].values)[0:4]))
        ds_idx[cfg.dim_time] = utils.reset_calendar(ds_idx, year_1, year_n, cfg.freq_YS)

        # Save result to NetCDF file.
        desc = "/" + idx_name + "/" + os.path.basename(p_idx)
        utils.save_netcdf(ds_idx, p_idx, desc=desc)

    # Convert percentile threshold values for climate indices. This is sometimes required in time series.
    if (rcp == cfg.rcp_ref) and (idx_name == cfg.idx_prcptot):
        ds_idx = utils.open_netcdf(p_idx)
        da_idx = ds_idx.mean(dim=[cfg.dim_longitude, cfg.dim_latitude])[idx_name]
        param_pr = cfg.idx_params[cfg.idx_codes.index(idx_code)][0]
        if "p" in str(param_pr):
            param_pr = float(param_pr.replace("p", "")) / 100.0
            cfg.idx_params[cfg.idx_codes.index(idx_code)][0] = \
                float(round(da_idx.quantile(param_pr).values.ravel()[0], 2))


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
                    (tasmin[tasmin[cfg.dim_time] == t] > param_tasmin) & (tasmax[tasmax[cfg.dim_time] == t] > param_tasmax)
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


def tot_duration_dry_periods(da_pr: xr.DataArray,  pr_dry: float, dt_dry: int, per: str, doy_a: int, doy_b: int)\
        -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the total duration of dry periods. A dry period occurs if precipitation amount is less than 'pr_dry'
    during 'dt_dry' consecutive days.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    pr_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    dt_dry: int
        Number of days to have a dry period.
    per: str
        Period over which to combine data {"1d" = one day, "tot" = total}.
    doy_a: int
        First day of year to consider.
    doy_b: int
        Last day of year to consider.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion. The exact value seems to be 0.0000008 mm/day (instead of 0.000001 mm/day).
    pr_dry = convert_units_to(str(pr_dry) + " mm/day", da_pr)

    # Condition #1: Days that belong to a dry period.
    da_cond1 = da_pr.copy()
    da_cond1 = da_cond1.astype(bool)
    da_cond1[:, :, :] = False
    n_t = len(da_pr[cfg.dim_time])
    for t in range(n_t - dt_dry):
        if per == "1d":
            da_t = xr.DataArray(da_pr[t:(t + dt_dry), :, :] < pr_dry).sum(dim=cfg.dim_time) == dt_dry
        else:
            da_t = xr.DataArray(da_pr[t:(t + dt_dry), :, :].sum(dim=cfg.dim_time)) < pr_dry
        da_cond1[t:(t + dt_dry), :, :] = da_cond1[t:(t + dt_dry), :, :] | da_t

    # Condition #2 : Days that are between 'doy_a' and 'doy_b'.
    da_cond2 = da_pr.copy()
    da_cond2 = da_cond2.astype(bool)
    da_cond2[:, :, :] = True
    if (doy_a is not None) and (doy_b is not None):
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


def rain_start(da_pr: xr.DataArray, pr_wet: float, dt_wet: int, doy: int, pr_dry: float, dt_dry: int, dt_tot: int)\
        -> xr.DataArray:

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
    doy: int
        Day of year after which the season starts.
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
    # 'dt_tot' days. These days must also consider 'doy'.
    da_cond2 = da_cond1.copy()
    da_cond2[:, :, :] = True
    for t in range(n_t):
        if (da_pr[t].time.dt.dayofyear >= doy) and (t < n_t - dt_dry):
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


def rain_end(da_pr: xr.DataArray, da_rainstart1: xr.DataArray, da_rainstart2: xr.DataArray, method: str, pr: float,
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
        The 'depletion' method is based on the period required for an amount of water (mm) to evaporate at a rate of
        'et_rate' (mm/day), considering that any amount of precipitation received during that period must evaporate as
        well.
        The 'event' method is based on the occurrence (or not) of an event preventing the end of the rain season. The
        rain season stops when no daily precipitation greater than 'p_event' have occurred over a period of 'd_event'
        days.
    pr : float
        'Depletion' method: precipitation amount that must evaporate (mm).
        'Event' method: last non-negligible precipitation event of the rain season (mm).
    etp: float
        'Depletion' method: Evapotranspiration rate (mm/day).
    dt: float
        'Event' method: period (number of days) during which there must not be a day with 'pr' precipitation.
    doy_a: int
        First day of year at or after which the season ends.
    doy_b: int
        Last day of year at or before which the season ends.
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

    # TODO: Ideally, the algorithm should not depend on user-specified boundaries (doy_a and doy_b). This could be
    #       problematic if the climate changes substantially.

    else:

        # Combined conditions.
        da_conds = da_pr.copy().astype(float)
        da_conds[:, :, :] = 0

        # Extract year and day of year.
        year_list = utils.extract_date_field(da_pr, "year")
        doy_list = utils.extract_date_field(da_pr, "doy")

        # Calculate conditions at each time step.
        for t in range(n_t):

            year     = year_list[t]
            doy      = doy_list[t]

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

    return da_end


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


def wind_days_above(da_vv: xr.DataArray, da_dd: xr.DataArray, param_vv: float, param_dd: float = None,
                    param_dd_tol: float = 45, months: List[int] = None) -> xr.DataArray:

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
    months: [int]
        List of months.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Condition #1: Wind speed.
    da_cond1 = da_vv > param_vv

    # Condition #2: Wind direction.
    if da_dd is None:
        da_cond2 = True
    else:
        da_cond2 = (da_dd - param_dd <= param_dd_tol) if param_dd is not None else True

    # Condition #3: Month.
    da_cond3 = True
    if months is not None:
        for i in range(len(months)):
            da_cond3_i = da_vv.time.dt.month == months[i]
            da_cond3 = da_cond3_i if i == 0 else da_cond3 | da_cond3_i

    # Combine conditions.
    da_conds = da_cond1 & da_cond2 & da_cond3

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
    if cfg.opt_stat[1] or (cfg.opt_ra and (cfg.opt_map[1] or cfg.opt_save_csv[1])):
        utils.log(msg)
        statistics.calc_stats(cfg.cat_idx)
    else:
        utils.log(msg + not_req)

    utils.log("-")
    msg = "Step #7b  Exporting results to CSV files (indices)"
    if cfg.opt_save_csv[1]:
        utils.log(msg)
        utils.log("-")
        utils.log("Step #7b1 Generating times series (indices)")
        statistics.calc_ts(cfg.cat_idx)
        if not cfg.opt_ra:
            utils.log("-")
            utils.log("Step #7b2 Converting NetCDF to CSV files (indices)")
            statistics.conv_nc_csv(cfg.cat_idx)
    else:
        utils.log(msg + not_req)

    # Plots ------------------------------------------------------------------------------------------------------------

    utils.log("=")
    msg = "Step #8   Generating diagrams and maps (indices)"
    if cfg.opt_plot[1] or cfg.opt_map[1]:
        utils.log(msg)
    else:
        utils.log(msg + not_req)

    # Generate plots.
    if cfg.opt_plot[1]:

        if not cfg.opt_save_csv[1]:
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
        for i in range(len(cfg.idx_codes)):
            statistics.calc_heatmap(cfg.idx_codes[i])

    else:
        utils.log(msg + " (not required)")


if __name__ == "__main__":
    run()
