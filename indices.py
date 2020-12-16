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
import pandas as pd
import statistics
import utils
import xarray as xr
import xclim.indices as indices
import warnings
from typing import List
from xclim.indices import run_length as rl
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to


def generate(idx_name: str, idx_threshs: [float]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a time series.

    Parameters:
    idx_name : str
        Index name.
    idx_threshs : [float]
        Threshold values associated with each indicator.
    --------------------------------------------------------------------------------------------------------------------
    """

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
    if idx_name in [cfg.idx_tnx, cfg.idx_tng, cfg.idx_tropicalnights]:
        var_or_idx_list.append(cfg.var_cordex_tasmin)

    elif idx_name in [cfg.idx_tx90p, cfg.idx_txdaysabove, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_txx,
                      cfg.idx_wsdi]:
        var_or_idx_list.append(cfg.var_cordex_tasmax)

    elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen, cfg.idx_tgg, cfg.idx_etr]:
        var_or_idx_list.append(cfg.var_cordex_tasmin)
        var_or_idx_list.append(cfg.var_cordex_tasmax)

    # Precipitation.
    elif idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_cwd, cfg.idx_cdd, cfg.idx_sdii, cfg.idx_prcptot,
                      cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_rainstart,
                      cfg.idx_rainend]:
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

        # Loop through emissions scenarios.
        for rcp in rcps:

            utils.log("Processing: '" + idx_name + "', '" + stn + "', '" + cfg.get_rcp_desc(rcp) + "'", True)

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
                    p_idx = cfg.get_d_idx(stn, idx_name) + idx_name + "_ref" + cfg.f_ext_nc
                else:
                    p_idx = cfg.get_d_scen(stn, cfg.cat_idx, idx_name) +\
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
                idx_threshs_str = []
                for i in range(len(idx_threshs)):
                    idx_thresh = idx_threshs[i]

                    # Convert thresholds (percentile to absolute value) ------------------------------------------------

                    if (idx_name == cfg.idx_tx90p) or ((idx_name == cfg.idx_wsdi) and (i == 0)):
                        idx_thresh = "90p"
                    if (idx_name in [cfg.idx_txdaysabove, cfg.idx_tx90p, cfg.idx_tropicalnights]) or\
                       ((idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]) and (i == 0)) or \
                       ((idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]) and (i <= 1)) or \
                       ((idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 0)):

                        if "p" in str(idx_thresh):
                            idx_thresh = float(idx_thresh.replace("p", "")) / 100.0
                            if rcp == cfg.rcp_ref:
                                if (idx_name in [cfg.idx_tx90p, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen,
                                                 cfg.idx_wsdi]) or (i == 1):
                                    idx_thresh =\
                                        ds_var_or_idx[i][cfg.var_cordex_tasmax].quantile(idx_thresh).values.ravel()[0]
                                elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
                                    idx_thresh =\
                                        ds_var_or_idx[i][cfg.var_cordex_tasmin].quantile(idx_thresh).values.ravel()[0]
                                elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
                                    if idx_name == cfg.idx_wgdaysabove:
                                        da_uas = ds_var_or_idx[0][cfg.var_cordex_uas]
                                        da_vas = ds_var_or_idx[1][cfg.var_cordex_vas]
                                        da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas)
                                    else:
                                        da_vv = ds_var_or_idx[0][cfg.var_cordex_sfcwindmax]
                                    idx_thresh = da_vv.quantile(idx_thresh).values.ravel()[0]
                                idx_thresh = float(round(idx_thresh, 2))
                                cfg.idx_threshs[cfg.idx_names.index(idx_name)][i] = idx_thresh
                            else:
                                idx_thresh = cfg.idx_threshs[cfg.idx_names.index(idx_name)][i]

                    # Combine threshold and unit -----------------------------------------------------------------------

                    if (idx_name in [cfg.idx_txdaysabove, cfg.idx_tx90p, cfg.idx_tropicalnights]) or\
                       ((idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]) and (i == 0)) or \
                       ((idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]) and (i <= 1)):
                        idx_ref = str(idx_thresh) + " " + cfg.unit_C
                        idx_fut = str(idx_thresh + cfg.d_KC) + " " + cfg.unit_K
                        idx_threshs_str.append(idx_ref if (rcp == cfg.rcp_ref) else idx_fut)

                    elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                                      cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:
                        idx_threshs_str.append(str(idx_thresh) + " mm/day")

                    elif (idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 1):
                        idx_threshs_str.append(str(idx_thresh) + " " + cfg.unit_ms1)

                    elif not ((idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 4)):
                        idx_threshs_str.append(str(idx_thresh))

                    # Split lists --------------------------------------------------------------------------------------

                    if (idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]) and (i == 4):
                        if idx_threshs[i] == "nan":
                            idx_threshs_str.append(idx_threshs[i])
                        else:
                            items = idx_threshs[i].replace("[", "").replace("]", "").split(";")
                            idx_threshs_str.append([int(i) for i in items])

                # Exit loop if the file already exists (reference file only).
                if (rcp == cfg.rcp_ref) and os.path.exists(p_idx) and (not cfg.opt_force_overwrite):
                    continue

                # Temperature ------------------------------------------------------------------------------------------

                da_idx = None
                if idx_name in [cfg.idx_txdaysabove, cfg.idx_tx90p]:
                    da_tasmax = ds_var_or_idx[0][cfg.var_cordex_tasmax]
                    thresh_tasmax = idx_threshs_str[0]
                    da_idx = xr.DataArray(indices.tx_days_above(da_tasmax, thresh_tasmax).values)
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                elif idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen, cfg.idx_wsdi]:
                    da_tasmax = ds_var_or_idx[0][cfg.var_cordex_tasmax]
                    thresh_tasmax = idx_threshs_str[0]
                    thresh_ndays = int(float(idx_threshs_str[1]))
                    if idx_name == cfg.idx_hotspellfreq:
                        da_idx = xr.DataArray(
                            indices.hot_spell_frequency(da_tasmax, thresh_tasmax, thresh_ndays).values)
                    elif idx_name == cfg.idx_hotspellmaxlen:
                        da_idx = xr.DataArray(
                            indices.hot_spell_max_length(da_tasmax, thresh_tasmax, thresh_ndays).values)
                    else:
                        da_idx = xr.DataArray(
                            indices.warm_spell_duration_index(da_tasmax, da_tx90p, thresh_ndays).values)
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
                    da_tasmin = ds_var_or_idx[0][cfg.var_cordex_tasmin]
                    da_tasmax = ds_var_or_idx[1][cfg.var_cordex_tasmax]
                    thresh_tasmin = idx_threshs_str[0]
                    thresh_tasmax = idx_threshs_str[1]
                    window = int(float(idx_threshs_str[2]))
                    if idx_name == cfg.idx_heatwavemaxlen:
                        da_idx = xr.DataArray(
                            heat_wave_max_length(da_tasmin, da_tasmax, thresh_tasmin, thresh_tasmax, window).values)
                    else:
                        da_idx = xr.DataArray(
                            heat_wave_total_length(da_tasmin, da_tasmax, thresh_tasmin, thresh_tasmax, window).values)
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                elif idx_name == cfg.idx_txx:
                    da_tasmax = ds_var_or_idx[0][cfg.var_cordex_tasmax]
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
                        thresh_tasmin = idx_threshs_str[0]
                        da_idx = xr.DataArray(indices.tropical_nights(da_tasmin, thresh_tasmin))
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
                    da_idx = xr.DataArray(indices.drought_code(da_tas, da_pr, da_lat, shut_down_mode="temperature")).\
                        resample(time=cfg.freq_YS).mean()
                    idx_units = cfg.unit_1

                # Precipitation ----------------------------------------------------------------------------------------

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
                    thresh_pr = idx_threshs_str[0]
                    if idx_name in cfg.idx_cwd:
                        da_idx = xr.DataArray(
                            indices.maximum_consecutive_wet_days(da_pr, thresh_pr, cfg.freq_YS))
                    elif idx_name in cfg.idx_cdd:
                        da_idx = xr.DataArray(
                            indices.maximum_consecutive_dry_days(da_pr, thresh_pr, cfg.freq_YS))
                    elif idx_name in [cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays]:
                        da_idx = xr.DataArray(indices.wetdays(da_pr, thresh_pr, cfg.freq_YS))
                    elif idx_name == cfg.idx_drydays:
                        da_idx = xr.DataArray(indices.dry_days(da_pr, thresh_pr, cfg.freq_YS))
                    elif idx_name == cfg.idx_sdii:
                        da_idx = xr.DataArray(indices.daily_pr_intensity(da_pr, thresh_pr))
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                elif idx_name == cfg.idx_rainstart:
                    da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
                    p_wet = float(idx_threshs_str[0])
                    d_wet = int(idx_threshs_str[1])
                    doy = int(idx_threshs_str[2])
                    p_dry = float(idx_threshs_str[3])
                    d_dry = int(idx_threshs_str[4])
                    d_tot = int(idx_threshs_str[5])
                    da_idx = xr.DataArray(rain_season_start(da_pr, p_wet, d_wet, doy, p_dry, d_dry, d_tot))
                    idx_units = cfg.unit_1

                elif idx_name == cfg.idx_rainend:
                    da_pr = ds_var_or_idx[0][cfg.var_cordex_pr]
                    p_stock = float(idx_threshs_str[0])
                    et_rate = float(idx_threshs_str[1])
                    doy = int(idx_threshs_str[2])
                    da_idx = xr.DataArray(rain_season_end(da_pr, p_stock, et_rate, doy))
                    idx_units = cfg.unit_1

                elif idx_name == cfg.idx_raindur:
                    da_rainstart = ds_var_or_idx[0][cfg.idx_rainstart]
                    da_rainend = ds_var_or_idx[1][cfg.idx_rainend]
                    da_idx = da_rainend - da_rainstart
                    idx_units = cfg.unit_1

                # Wind -------------------------------------------------------------------------------------------------

                elif idx_name in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
                    thresh_vv = float(idx_threshs_str[0])
                    thresh_vv_neg = idx_threshs_str[1]
                    thresh_dd = float(idx_threshs_str[2])
                    thresh_dd_tol = float(idx_threshs_str[3])
                    thresh_months = None if (idx_threshs_str[4] == "nan") else idx_threshs_str[4]
                    if idx_name == cfg.idx_wgdaysabove:
                        da_uas = ds_var_or_idx[0][cfg.var_cordex_uas]
                        da_vas = ds_var_or_idx[1][cfg.var_cordex_vas]
                        da_vv, da_dd = indices.uas_vas_2_sfcwind(da_uas, da_vas, thresh_vv_neg)
                    else:
                        da_vv = ds_var_or_idx[0][cfg.var_cordex_sfcwindmax]
                        da_dd = None
                    da_idx =\
                        xr.DataArray(strong_wind(da_vv, da_dd, thresh_vv, thresh_dd, thresh_dd_tol, thresh_months))
                    idx_units = cfg.unit_1

                da_idx.attrs[cfg.attrs_units] = idx_units

                # ======================================================================================================
                # TODO.CUSTOMIZATION.INDEX.END
                # ======================================================================================================

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


def heat_wave_max_length(tasmin: xr.DataArray, tasmax: xr.DataArray, thresh_tasmin: str = "22.0 degC",
                         thresh_tasmax: str = "30 degC", window: int = 3, freq: str = cfg.freq_YS) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Same as equivalent non-working function in xclim.indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)

    # Adjust calendars.
    if tasmin.time.dtype != tasmax.time.dtype:
        tasmin["time"] = tasmin["time"].astype("datetime64[ns]")
        tasmax["time"] = tasmax["time"].astype("datetime64[ns]")

    # Call the xclim function if time dimension is the same.
    n_tasmin = len(tasmin[cfg.dim_time])
    n_tasmax = len(tasmax[cfg.dim_time])

    if n_tasmin == n_tasmax:
        return indices.heat_wave_max_length(tasmin, tasmax, thresh_tasmin, thresh_tasmax, window, freq)

    # Calculate manually.
    else:

        cond = tasmin
        if n_tasmax > n_tasmin:
            cond = tasmax
        for t in cond.time:
            if (t.values in tasmin["time"]) and (t.values in tasmax["time"]):
                cond[cond["time"] == t] =\
                    (tasmin[tasmin["time"] == t] > thresh_tasmin) & (tasmax[tasmax["time"] == t] > thresh_tasmax)
            else:
                cond[cond["time"] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)
        max_l = group.map(rl.longest_run, dim="time")

        return max_l.where(max_l >= window, 0)


def heat_wave_total_length(tasmin: xr.DataArray, tasmax: xr.DataArray, thresh_tasmin: str = "22.0 degC",
                           thresh_tasmax: str = "30 degC", window: int = 3, freq: str = cfg.freq_YS) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Same as equivalent non-working function in xclim.indices.
    --------------------------------------------------------------------------------------------------------------------
    """

    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)
    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)

    # Adjust calendars.
    if tasmin.time.dtype != tasmax.time.dtype:
        tasmin["time"] = tasmin["time"].astype("datetime64[ns]")
        tasmax["time"] = tasmax["time"].astype("datetime64[ns]")

    # Call the xclim function if time dimension is the same.
    n_tasmin = len(tasmin[cfg.dim_time])
    n_tasmax = len(tasmax[cfg.dim_time])

    if n_tasmin == n_tasmax:
        return indices.heat_wave_total_length(tasmin, tasmax, thresh_tasmin, thresh_tasmax, window, freq)

    # Calculate manually.
    else:

        cond = tasmin
        if n_tasmax > n_tasmin:
            cond = tasmax
        for t in cond.time:
            if (t.values in tasmin["time"]) and (t.values in tasmax["time"]):
                cond[cond["time"] == t] =\
                    (tasmin[tasmin["time"] == t] > thresh_tasmin) & (tasmax[tasmax["time"] == t] > thresh_tasmax)
            else:
                cond[cond["time"] == t] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            group = cond.resample(time=freq)

        return group.map(rl.windowed_run_count, args=(window,), dim="time")


def rain_season_start(da_pr: xr.DataArray, p_wet: float, d_wet: int, doy: int, p_dry: float, d_dry: int, d_tot: int)\
                      -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the first day of the rain season.

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

    # Length of dimensions.
    n_t = len(da_pr[cfg.dim_time])
    n_lat = len(da_pr[cfg.dim_latitude])
    n_lon = len(da_pr[cfg.dim_longitude])

    # Condition #1: Flag the start of all sequences of 'd_wet' consecutive days with a sum of at least 'p_wet' in
    # precipitation.
    p_wet = convert_units_to(str(p_wet) + " mm/day", da_pr)
    da_cond1 = xr.DataArray(da_pr.rolling(time=d_wet).sum() >= p_wet)

    # Condition #2: Flag the last day of all sequences of 'd_dry' consecutive days with less than 'p_dry' in
    # precipitation on each day. Then identify each day within a 'd_tot' sequence in which there is at least one dry
    # period.
    p_dry = convert_units_to(str(p_dry) + " mm/day", da_pr)
    da_cond2 = da_cond1
    da_cond2[:, :, :] = True
    for t in range(n_t):
        for j in range(1, d_tot - d_dry + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_cond2[t] = da_cond2[t] &\
                    ((da_pr >= p_dry)[t + j:(t + j + d_dry)].resample(time=cfg.freq_YS).sum(dim=cfg.dim_time) > 0)[0]

    # Condition #3: Flag days at or after 'doy'.
    da_cond3 = (da_pr.time.dt.dayofyear >= doy)

    # Combine the 3 conditions.
    da_conds = da_cond1 & da_cond2 & da_cond3

    # Obtain the first day of each year where conditions apply.
    # Then remove errors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_start = da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_start.values[(da_start.values < 0) | (da_start.values > 365)] = np.nan

    return da_start


def rain_season_end(da_pr: xr.DataArray, p_stock: float, et_rate: float, doy: int) -> xr.DataArray:

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
    doy: int
        Day of year after which the season ends.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Eliminate negative values.
    da_pr.values[da_pr.values < 0] = 0

    # Condition #1: Calculate the stock remaining, assuming that it will take 'p_stock' / 'et_rate' days for the water
    # to evaporate. This is a simplification of reality.
    p_stock = convert_units_to(str(p_stock) + " mm/day", da_pr)
    et_rate = convert_units_to(str(et_rate) + " mm/day", da_pr)
    da_cond1 = (p_stock - xr.DataArray(da_pr.rolling(time=int(p_stock/et_rate)).sum()) <= 0)

    # Condition #2: Identify days at or after 'doy'.
    da_cond2 = (da_pr.time.dt.dayofyear >= doy)

    # Combine the 2 conditions.
    da_conds = da_cond1 & da_cond2

    # Obtain the first day of each year where conditions apply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        da_end = da_conds.resample(time=cfg.freq_YS).map(rl.first_run, window=1, dim=cfg.dim_time, coord="dayofyear")
    da_end.values[(da_end.values < 0) | (da_end.values > 365)] = np.nan

    return da_end


def strong_wind(da_vv: xr.DataArray, da_dd: xr.DataArray, thresh_vv: float, thresh_dd: float = None,
                thresh_dd_tol: float = 45, months: List[int] = None) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate the number of days with a strong wind, potentially from a specified direction.

    Parameters
    ----------
    da_vv : xr.DataArray
        Wind speed (m s-1).
    da_dd : xr.DataArray
        Direction for which the wind is coming from (degrees).
    thresh_vv: float
        Threshold related to 'da_wind' (m s-1).
    thresh_dd: float
        Threshold related to 'da_windfromdir' (degrees).
    thresh_dd_tol: float
        Threshold tolerance related to 'da_windfromdir' (degrees).
    months: [int]
        List of months.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Condition #1: Wind speed.
    da_cond1 = da_vv > thresh_vv

    # Condition #2: Wind direction.
    da_cond2 = (da_dd - thresh_dd <= thresh_dd_tol) if thresh_dd is not None else True

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
        da_strong_wind = da_conds.resample(time=cfg.freq_YS).sum(dim=cfg.dim_time)

    return da_strong_wind


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
        for i in range(0, len(cfg.idx_names)):
            generate(cfg.idx_names[i], cfg.idx_threshs[i])
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

        for i in range(len(cfg.idx_names)):

            # Get the minimum and maximum values in the statistics file.
            idx = cfg.idx_names[i]
            p_stat = cfg.get_d_scen(cfg.obs_src, cfg.cat_stat, cfg.cat_idx + "/" + idx) +\
                idx + "_" + cfg.obs_src + cfg.f_ext_csv
            if not os.path.exists(p_stat):
                z_min = z_max = None
            else:
                df_stats = pd.read_csv(p_stat, sep=",")
                vals = df_stats[(df_stats["stat"] == cfg.stat_min) |
                                (df_stats["stat"] == cfg.stat_max) |
                                (df_stats["stat"] == "none")]["val"]
                z_min = min(vals)
                z_max = max(vals)

            # Reference period.
            statistics.calc_heatmap(cfg.idx_names[i], cfg.rcp_ref, [cfg.per_ref], z_min, z_max)

            # Future period.
            for rcp in cfg.rcps:
                statistics.calc_heatmap(cfg.idx_names[i], rcp, cfg.per_hors, z_min, z_max)

    else:
        utils.log(msg + " (not required)")


if __name__ == "__main__":
    run()
