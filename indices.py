# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions related to climate indices.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import numpy as np
import os.path
import statistics
import utils
import xarray as xr
import xclim.indices as indices
import xclim.indices.generic as indices_gen
import warnings
from typing import List
from xclim.indices import run_length as rl
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to


def create_mask(stn: str, var: str = None) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a mask, based on climate scenarios for the temperature or precipitation variable.
    All values with a value are attributed a value of 1. Other values are assigned 'nan'.

    Parameters
    ----------
    stn : str
        Station name.
    var : str, optional
        Imposed variable name.
    --------------------------------------------------------------------------------------------------------------------
    """

    da_mask = None

    f_list = glob.glob(cfg.get_d_scen(stn, cfg.cat_obs, var if var is not None else "*") + "/*" + cfg.f_ext_nc)
    for i in range(len(f_list)):

        # Open NetCDF file.
        ds = utils.open_netcdf(f_list[i])
        var = list(ds.data_vars)[0]
        if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax, cfg.var_cordex_pr]:

            # Flag 'nan' values.
            if var == cfg.var_cordex_pr:
                p_dry_error = convert_units_to(str(0.000001) + " mm/day", ds[var])
                ds[var].values[(ds[var].values > 0) & (ds[var].values <= p_dry_error)] = np.nan

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
    idx_name   = cfg.get_idx_name(idx_code)
    idx_params = cfg.idx_params[cfg.idx_codes.index(idx_code)]

    # Emission scenarios.
    rcps = [cfg.rcp_ref] + cfg.rcps

    # ==================================================================================================================
    # TODO.CUSTOMIZATION.INDEX.BEGIN
    # Specify the required variable(s) by copying the following
    # code block.
    # ==================================================================================================================

    # Select variables.
    var_or_idx_list = []

    # Temperature.
    if idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights, cfg.idx_tngmonthsbelow]:
        var_or_idx_list.append(cfg.var_cordex_tasmin)

    elif idx_name in [cfg.idx_tx90p, cfg.idx_txdaysabove, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_txg,
                      cfg.idx_txx, cfg.idx_wsdi]:
        var_or_idx_list.append(cfg.var_cordex_tasmax)

    elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen, cfg.idx_tgg, cfg.idx_etr]:
        var_or_idx_list.append(cfg.var_cordex_tasmin)
        var_or_idx_list.append(cfg.var_cordex_tasmax)

    # Precipitation.
    elif idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_cwd, cfg.idx_cdd, cfg.idx_sdii, cfg.idx_prcptot,
                      cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_rainstart,
                      cfg.idx_rainend, cfg.idx_drydurtot]:
        var_or_idx_list.append(cfg.var_cordex_pr)

    elif idx_name == cfg.idx_raindur:
        var_or_idx_list.append(cfg.idx_rainstart)
        var_or_idx_list.append(cfg.idx_rainend)

    # Temperature-precipitation.
    elif idx_name == cfg.idx_dc:
        var_or_idx_list.append(cfg.var_cordex_tas)
        var_or_idx_list.append(cfg.var_cordex_pr)

    # Wind.
    elif idx_name == cfg.idx_wgdaysabove:
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
        var_or_idx_list_avail = True
        for var_or_idx in var_or_idx_list:
            if ((var_or_idx in cfg.variables_cordex) and
                not os.path.isdir(cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx))) or\
               ((var_or_idx not in cfg.variables_cordex) and
                not os.path.isdir(cfg.get_d_idx(stn, var_or_idx))):
                var_or_idx_list_avail = False
                break
        if not var_or_idx_list_avail:
            continue

        # Variable that holds the 90th percentile of tasmax for the reference period.
        da_tx90p = None

        # Create mask.
        da_mask = None
        if stn == cfg.obs_src_era5_land:
            # TODO: Remove the second argument after regenerating scenarios for temperature variables.
            da_mask = create_mask(stn, cfg.var_cordex_pr)

        # Loop through emissions scenarios.
        for rcp in rcps:

            utils.log("Processing: '" + idx_code + "', '" + stn + "', '" + cfg.get_rcp_desc(rcp) + "'", True)

            # Analysis of simulation files -----------------------------------------------------------------------------

            utils.log("Collecting simulation files.", True)

            # List simulation files for the first variable. As soon as there is no file for one variable, the analysis
            # for the current RCP needs to abort.
            if rcp == cfg.rcp_ref:
                if var_or_idx_list[0] in cfg.variables_cordex:
                    p_sim = cfg.get_d_scen(stn, cfg.cat_obs, var_or_idx_list[0]) + var_or_idx_list[0] +\
                            "_" + stn + cfg.f_ext_nc
                else:
                    p_sim = cfg.get_d_idx(cfg.obs_src, var_or_idx_list[0]) + var_or_idx_list[0] + "_ref" + cfg.f_ext_nc
                if os.path.exists(p_sim) and (type(p_sim) is str):
                    p_sim = [p_sim]
            else:
                if var_or_idx_list[0] in cfg.variables_cordex:
                    d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx_list[0])
                else:
                    d = cfg.get_d_idx(stn, var_or_idx_list[0])
                p_sim = glob.glob(d + "*_" + rcp + cfg.f_ext_nc)
            if not p_sim:
                continue

            utils.log("Calculating climate indices", True)

            # Ensure that simulations are available for all other variables (than the first one).
            if len(var_or_idx_list) > 1:
                p_sim_fix = []
                for p_sim_i in p_sim:
                    missing = False
                    for var_j in var_or_idx_list[1:]:
                        p_sim_j = p_sim_i.replace(var_or_idx_list[0], var_j)
                        if not os.path.exists(p_sim_j):
                            missing = True
                            break
                    if not missing:
                        p_sim_fix.append(p_sim_i)
                p_sim = p_sim_fix

            # Loop through simulations.
            for i_sim in range(len(p_sim)):

                # Scenarios --------------------------------------------------------------------------------------------

                # Determine file name.
                if rcp == cfg.rcp_ref:
                    p_idx = cfg.get_d_idx(stn, idx_code) + idx_name + "_ref" + cfg.f_ext_nc
                else:
                    p_idx = cfg.get_d_scen(stn, cfg.cat_idx, idx_code) +\
                            os.path.basename(p_sim[i_sim]).replace(var_or_idx_list[0], idx_name)

                # Exit loop if the file already exists (simulations files only; not reference file).
                if (rcp != cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite):
                    continue

                # Load datasets (one per variable).
                ds_var_or_idx = []
                for i_var in range(0, len(var_or_idx_list)):
                    var = var_or_idx_list[i_var]
                    p_sim_i = p_sim[i_sim] if i_var == 0 else p_sim[i_sim].replace(var_or_idx_list[0], var)
                    ds = utils.open_netcdf(p_sim_i)

                    # Adjust temperature units.
                    if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                        if ds[var].attrs[cfg.attrs_units] == cfg.unit_K:
                            ds[var] = ds[var] - cfg.d_KC
                        elif rcp == cfg.rcp_ref:
                            ds[var][cfg.attrs_units] = cfg.unit_C
                        ds[var].attrs[cfg.attrs_units] = cfg.unit_C

                    ds_var_or_idx.append(ds)

                # Indices ----------------------------------------------------------------------------------------------

                # ======================================================================================================
                # TODO.CUSTOMIZATION.INDEX.BEGIN
                # Calculate the index by copying the following code block.
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
                    if (idx_name in [cfg.idx_txdaysabove, cfg.idx_tngmonthsbelow, cfg.idx_tx90p,
                                     cfg.idx_tropicalnights]) or\
                       ((idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]) and (i == 0)) or \
                       ((idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]) and (i <= 1)) or \
                       ((idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 0)):

                        if "p" in str(idx_param):
                            idx_param = float(idx_param.replace("p", "")) / 100.0
                            if rcp == cfg.rcp_ref:
                                if (idx_name in [cfg.idx_tx90p, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen,
                                                 cfg.idx_wsdi]) or (i == 1):
                                    idx_param =\
                                        ds_var_or_idx[i][cfg.var_cordex_tasmax].quantile(idx_param).values.ravel()[0]
                                elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
                                    idx_param =\
                                        ds_var_or_idx[i][cfg.var_cordex_tasmin].quantile(idx_param).values.ravel()[0]
                                elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
                                    if idx_name == cfg.idx_wgdaysabove:
                                        da_uas = ds_var_or_idx[0][cfg.var_cordex_uas]
                                        da_vas = ds_var_or_idx[1][cfg.var_cordex_vas]
                                        da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas)
                                    else:
                                        da_vv = ds_var_or_idx[0][cfg.var_cordex_sfcwindmax]
                                    idx_param = da_vv.quantile(idx_param).values.ravel()[0]
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
                        idx_params_str.append(str(idx_param) + " " + cfg.unit_ms1)

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
                        da_idx = xr.DataArray(indices.tx_days_above(da_tasmax, param_tasmax).values)
                        da_idx = da_idx.astype(int)
                        idx_units = cfg.unit_1

                    elif idx_name == cfg.idx_tngmonthsbelow:
                        da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
                        param_tasmin = float(idx_params_str[0])
                        if da_tasmin.attrs[cfg.attrs_units] != cfg.unit_C:
                            param_tasmin += cfg.d_KC
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
                        if idx_name == cfg.idx_txg:
                            da_idx = xr.DataArray(indices.tx_mean(da_tasmax))
                        else:
                            da_idx = xr.DataArray(indices.tx_max(da_tasmax))
                        idx_units = cfg.unit_C

                    elif idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights]:
                        da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
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
                        if idx_name == cfg.idx_tgg:
                            da_idx = xr.DataArray(indices.tg_mean(indices.tas(da_tasmin, da_tasmax)))
                        else:
                            # TODO: xclim function does not seem to be working (values are much too low).
                            # da_idx = xr.DataArray(indices.extreme_temperature_range(da_tasmin, da_tasmax))
                            da_idx = xr.DataArray(indices.tx_max(da_tasmax) - indices.tn_min(da_tasmin))
                        idx_units = cfg.unit_C

                    elif idx_name == cfg.idx_dc:
                        da_tas = ds_var_or_idx[0][cfg.var_cordex_tas]
                        da_pr  = ds_var_or_idx[1][cfg.var_cordex_pr]
                        da_lon, da_lat = utils.get_coordinates(ds_var_or_idx[0])
                        da_idx = xr.DataArray(indices.drought_code(
                            da_tas, da_pr, da_lat, shut_down_mode="temperature")).resample(time=cfg.freq_YS).mean()
                        idx_units = cfg.unit_1

                    # Precipitation ------------------------------------------------------------------------------------

                    elif idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_prcptot]:
                        da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
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
                            da_idx = xr.DataArray(indices.wetdays(da_pr, param_pr, cfg.freq_YS))
                        elif idx_name == cfg.idx_drydays:
                            da_idx = xr.DataArray(indices.dry_days(da_pr, param_pr, cfg.freq_YS))
                        elif idx_name == cfg.idx_sdii:
                            da_idx = xr.DataArray(indices.daily_pr_intensity(da_pr, param_pr))
                        da_idx = da_idx.astype(int)
                        idx_units = cfg.unit_1

                    elif idx_name == cfg.idx_rainstart:
                        da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
                        p_wet = float(idx_params_str[0])
                        d_wet = int(idx_params_str[1])
                        doy = int(idx_params_str[2])
                        p_dry = float(idx_params_str[3])
                        d_dry = int(idx_params_str[4])
                        d_tot = int(idx_params_str[5])
                        da_idx = xr.DataArray(rain_start_2(da_pr, p_wet, d_wet, doy, p_dry, d_dry, d_tot))
                        idx_units = cfg.unit_1

                    elif idx_name == cfg.idx_rainend:
                        da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
                        p_stock = float(idx_params_str[0])
                        et_rate = float(idx_params_str[1])
                        doy_a = int(idx_params_str[2])
                        doy_b = int(idx_params_str[3])
                        da_idx = xr.DataArray(rain_end_2(da_pr, p_stock, et_rate, doy_a, doy_b))
                        idx_units = cfg.unit_1

                    elif idx_name == cfg.idx_raindur:
                        da_rainstart = ds_var_or_idx[0][cfg.idx_rainstart]
                        da_rainend = ds_var_or_idx[1][cfg.idx_rainend]
                        da_idx = da_rainend - da_rainstart
                        idx_units = cfg.unit_1

                    elif idx_name == cfg.idx_drydurtot:
                        da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
                        p_dry = float(idx_params_str[0])
                        d_dry = int(idx_params_str[1])
                        per = idx_params_str[2]
                        doy_a = idx_params_str[3]
                        doy_a = None if (str(doy_a) == "nan") else int(doy_a)
                        doy_b = idx_params_str[4]
                        doy_b = None if (str(doy_b) == "nan") else int(doy_b)
                        da_idx = xr.DataArray(tot_duration_dry_periods(da_pr,  p_dry, d_dry, per, doy_a, doy_b))
                        idx_units = cfg.unit_1

                    # Wind ---------------------------------------------------------------------------------------------

                    elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
                        param_vv = float(idx_params_str[0])
                        param_vv_neg = idx_params_str[1]
                        param_dd = float(idx_params_str[2])
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

                    # Apply mask.
                    if da_mask is not None:
                        for t in range(len(da_idx.time)):
                            da_idx[t] = da_idx[t] * da_mask.values

                    # Create dataset.
                    da_idx.name = idx_name
                    ds_idx = da_idx.to_dataset()
                    ds_idx.attrs[cfg.attrs_units] = idx_units
                    if "dim_0" in list(ds_idx.dims):
                        ds_idx = ds_idx.rename_dims({"dim_0": cfg.dim_time})
                    if "dim_1" in list(ds_idx.dims):
                        ds_idx = ds_idx.rename_dims({"dim_1": cfg.dim_lat, "dim_2": cfg.dim_lon})
                    else:
                        ds_idx = ds_idx.expand_dims(lon=1)
                        ds_idx = ds_idx.expand_dims(lat=1)
                    ds_idx.attrs[cfg.attrs_sname] = idx_name
                    ds_idx.attrs[cfg.attrs_lname] = idx_name
                    ds_idx = utils.copy_coordinates(ds_var_or_idx[0], ds_idx)
                    ds_idx[idx_name] = utils.copy_coordinates(ds_var_or_idx[0][var_or_idx_list[0]], ds_idx[idx_name])
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
        tasmin["time"] = tasmin["time"].astype("datetime64[ns]")
        tasmax["time"] = tasmax["time"].astype("datetime64[ns]")

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
            if (t.values in tasmin["time"]) and (t.values in tasmax["time"]):
                cond[cond["time"] == t] =\
                    (tasmin[tasmin["time"] == t] > param_tasmin) & (tasmax[tasmax["time"] == t] > param_tasmax)
            else:
                cond[cond["time"] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)
        max_l = group.map(rl.longest_run, dim="time")

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
        tasmin["time"] = tasmin["time"].astype("datetime64[ns]")
        tasmax["time"] = tasmax["time"].astype("datetime64[ns]")

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
            if (t.values in tasmin["time"]) and (t.values in tasmax["time"]):
                cond[cond["time"] == t] =\
                    (tasmin[tasmin["time"] == t] > param_tasmin) & (tasmax[tasmax["time"] == t] > param_tasmax)
            else:
                cond[cond["time"] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)

        return group.map(rl.windowed_run_count, args=(window,), dim="time")


def tot_duration_dry_periods(da_pr: xr.DataArray,  p_dry: float, d_dry: int, per: str, doy_a: int, doy_b: int)\
        -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the total duration of dry periods. A dry period occurs if precipitation amount is less than 'p_dry' during
    'd_dry' consecutive days.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    p_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    d_dry: int
        Maximum number of days in a dry period embedded into the period of 'd_tot' days.
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
    p_dry = convert_units_to(str(p_dry) + " mm/day", da_pr)

    # Condition #1: Days that belong to a dry period.
    da_cond1 = da_pr.copy()
    da_cond1 = da_cond1.astype(bool)
    da_cond1[:, :, :] = False
    n_t = len(da_pr[cfg.dim_time])
    for t in range(n_t - d_dry):
        if per == "1d":
            da_t = (da_pr[t:(t + d_dry), :, :] < p_dry).sum(dim=cfg.dim_time) == d_dry
        else:
            da_t = da_pr[t:(t + d_dry), :, :].sum(dim=cfg.dim_time) < p_dry
        da_cond1[t:(t + d_dry), :, :] = da_cond1[t:(t + d_dry), :, :] | da_t

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


def rain_start_1(da_pr: xr.DataArray, p_wet: float, d_wet: int, doy: int, p_dry: float, d_dry: int, d_tot: int)\
               -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the first day of the rain season.
    This algorithm is very slow.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    p_wet : float
        Daily precipitation amount required in first 'd_wet' days.
    d_wet: int
        Number of days with precipitation at season start (related to 'p_wet').
    doy: int
        Day of year after which the season starts.
    p_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    d_dry: int
        Maximum number of days in a dry period embedded into the period of 'd_tot' days.
    d_tot: int
        Number of days (after the first 'd_wet' days) after which a dry season is searched for.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    p_wet = convert_units_to(str(p_wet) + " mm/day", da_pr)
    p_dry = convert_units_to(str(p_dry) + " mm/day", da_pr)

    # Determiner if 't' is the first day of the rain season.
    da_cond = da_pr.copy()
    da_cond[:, :, :] = False
    n_t = len(da_pr[cfg.dim_time])
    for t in range(n_t - d_wet - d_dry):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_cond1 = (da_pr[t:(t + d_wet), :, :].sum(dim=cfg.dim_time) >= p_wet)
        if (da_pr[t, :, :].time.dt.dayofyear >= doy) and (da_cond1[:, :].sum() > 0):
            for j in range(1, d_tot - d_dry + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    da_cond2 = ((da_pr >= p_dry)[(t + j):(t + j + d_dry)].
                                resample(time=cfg.freq_YS).sum(dim=cfg.dim_time) > 0)[0]
                    da_cond[t] = (da_cond[t] > 0) | (da_cond1 & da_cond2)

    # Obtain the first day of each year where conditions apply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_start = da_cond.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_start.values[(da_start.values < 0) | (da_start.values > 365)] = np.nan

    return da_start


def rain_start_2(da_pr: xr.DataArray, p_wet: float, d_wet: int, doy: int, p_dry: float, d_dry: int, d_tot: int)\
                 -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the first day of the rain season.
    This algorithm is fast.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    p_wet : float
        Daily precipitation amount required in first 'd_wet' days.
    d_wet: int
        Number of days with precipitation at season start (related to 'p_wet').
    doy: int
        Day of year after which the season starts.
    p_dry: float
        Daily precipitation amount under which precipitation is considered negligible.
    d_dry: int
        Maximum number of days in a dry period embedded into the period of 'd_tot' days.
    d_tot: int
        Number of days (after the first 'd_wet' days) after which a dry season is searched for.
    --------------------------------------------------------------------------------------------------------------------
    """
    # Eliminate negative values.

    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    p_wet = convert_units_to(str(p_wet) + " mm/day", da_pr)
    p_dry = convert_units_to(str(p_dry) + " mm/day", da_pr)

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])

    # Condition #1: Flag the first day of each series of 'd_wet' days with a total of 'p_wet' in precipitation.
    da_cond1 = xr.DataArray(da_pr.rolling(time=d_wet).sum() >= p_wet)
    da_cond1[0:(n_t-d_wet), :, :] = da_cond1[d_wet:n_t, :, :].values
    da_cond1[(n_t-d_wet):n_t] = False

    # Condition #2: Flag days that are not followed by a sequence of 'd_dry' consecutive days over the next 'd_wet' +
    # 'd_tot' days. These days must also consider 'doy'.
    da_cond2 = da_cond1.copy()
    da_cond2[:, :, :] = True
    for t in range(n_t):
        if (da_pr[t].time.dt.dayofyear >= doy) and (t < n_t - d_dry):
            t1 = t + d_wet
            t2 = t1 + d_dry
            da_t = (da_pr[t1:t2, :, :].max(dim=cfg.dim_time) >= p_dry)
            da_cond2[t] = da_cond2[t] & da_t

    # Combine conditions.
    da_conds = da_cond1 & da_cond2

    # Obtain the first day of each year where conditions apply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_start = da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_start.values[(da_start.values < 0) | (da_start.values > 365)] = np.nan

    return da_start


def rain_end_1(da_pr: xr.DataArray, p_stock: float, et_rate: float, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    p_stock : float
        Amount of precipitation to evaporate (mm).
    et_rate: int
        Evapotranspiration rate (mm/day).
    doy_a: int
        First day of year at or after which the season ends.
    doy_b: int
        Last day of year at or before which the season ends.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    p_stock = convert_units_to(str(p_stock) + " mm/day", da_pr)
    et_rate = convert_units_to(str(et_rate) + " mm/day", da_pr)

    # Calculate the minimum number of days that is required for evapotranspiration (assuming no rain).
    n_et = int(p_stock / et_rate)

    # DataArray that will hold results (one value per year).
    # Only the resulting structure is needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_end = da_pr.resample(time=cfg.freq_YS).min(dim=cfg.dim_time)

    # Extract years.
    years = utils.extract_years(da_end)

    for y in range(len(years)):

        # Extract values.
        years_str = [str(years[y]) + "-01-01", str(years[y]) + "-12-31"]
        da_y = da_pr.sel(time=slice(years_str[0], years_str[1]))

        # Look for the last day in a day sequence that results into water exhaustion.
        da_end_y = da_end[y, :, :].copy()
        da_end_y[:, :] = doy_b
        for i in range(doy_a - 1, doy_b - n_et):
            for j in range(i + n_et, doy_b):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    da_ij = da_y[i:j].resample(time=cfg.freq_YS).sum(dim=cfg.dim_time).squeeze() - (j - i + 1) * et_rate
                da_better = (da_ij < -p_stock) & (j < da_end_y)
                da_not_better = (da_ij >= -p_stock) | (j >= da_end_y)
                da_end_y = (da_better * (j + 1)) + (da_not_better * da_end_y)

        # Save result.
        da_end[y] = da_end_y

    return da_end


def rain_end_2(da_pr: xr.DataArray, p_stock: float, et_rate: float, doy_a: int, doy_b: int) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the last day of the rain season.

    Parameters
    ----------
    da_pr : xr.DataArray:
        Precipitation data.
    p_stock : float
        Amount of precipitation to evaporate (mm).
    et_rate: int
        Evapotranspiration rate (mm/day).
    doy_a: int
        First day of year at or after which the season ends.
    doy_b: int
        Last day of year at or before which the season ends.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Unit conversion.
    p_stock = convert_units_to(str(p_stock) + " mm/day", da_pr)
    et_rate = convert_units_to(str(et_rate) + " mm/day", da_pr)

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])

    # Calculate the minimum number of days that is required for evapotranspiration (assuming no rain).
    n_et = int(p_stock / et_rate)

    # DataArray that will hold results (one value per year).
    # Only the resulting structure is needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_end = da_pr.resample(time=cfg.freq_YS).min(dim=cfg.dim_time)
        da_end[:, :, :] = -1

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
                da_t1t2 = (da_pr[t1:t2, :, :].sum(dim=cfg.dim_time) - (t2 - t1 + 1) * et_rate)
                da_better     = (da_t1t2 < -p_stock) & ((da_end_y == -1) | (t2_doy < da_end_y))
                da_not_better = (da_t1t2 >= -p_stock) | ((da_end_y == -1) | (t2_doy >= da_end_y))
                da_end_y = (da_better * t2_doy) + (da_not_better * da_end_y)

        da_end[da_end.time.dt.year == t1_y] = da_end_y

    return da_end


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
