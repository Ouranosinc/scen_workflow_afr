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
import os.path
import pandas as pd
import statistics
import utils
import xarray as xr
import xclim.indices as indices
from xclim.core.calendar import resample_doy
from xclim.indices import run_length as rl
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

    # ==========================================================
    # TODO.CUSTOMIZATION.INDEX.BEGIN
    # Specify the required variable(s) by copying the following
    # code block.
    # ==========================================================

    # Select variables.
    vars = []

    # Temperature.
    if idx_name == cfg.idx_txdaysabove:
        vars.append(cfg.var_cordex_tasmax)

    elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
        vars.append(cfg.var_cordex_tasmin)
        vars.append(cfg.var_cordex_tasmax)

    # Precipitation.
    if idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_cwd, cfg.idx_cdd, cfg.idx_sdii, cfg.idx_prcptot,
                    cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays, cfg.idx_drydays]:
        vars.append(cfg.var_cordex_pr)

    # ==========================================================
    # TODO.CUSTOMIZATION.INDEX.END
    # ==========================================================

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Verify if this variable is available for the current station.
        vars_avail = True
        for var in vars:
            if not(os.path.isdir(cfg.get_d_scen(stn, cfg.cat_qqmap, var))):
                vars_avail = False
                break
        if not vars_avail:
            continue

        # Loop through emissions scenarios.
        for rcp in rcps:

            utils.log("Processing: '" + idx_name + "', '" + stn + "', '" + cfg.get_rcp_desc(rcp) + "'", True)

            # Analysis of simulation files -----------------------------------------------------------------------------

            utils.log("Collecting simulation files.", True)

            # List simulation files. As soon as there is no file for one variable, the analysis for the current RCP
            # needs to abort.
            n_sim = 0
            p_sim = []
            for var in vars:
                if rcp == cfg.rcp_ref:
                    p_sim_i = cfg.get_d_scen(stn, cfg.cat_obs, "") + var + "/" + var + "_" + stn + ".nc"
                    if os.path.exists(p_sim_i) and (type(p_sim_i) is str):
                        p_sim_i = [p_sim_i]
                else:
                    p_format = cfg.get_d_scen(stn, cfg.cat_qqmap, "") + var + "/*_" + rcp + ".nc"
                    p_sim_i = glob.glob(p_format)
                if not p_sim_i:
                    p_sim = []
                else:
                    p_sim.append(p_sim_i)
                    n_sim = len(p_sim_i)
            if not p_sim:
                continue

            utils.log("Calculating climate indices", True)

            # Loop through simulations.
            for i_sim in range(0, n_sim):

                # Scenarios --------------------------------------------------------------------------------------------

                # Skip iteration if the file already exists.
                if rcp == cfg.rcp_ref:
                    p_idx = cfg.get_d_idx(stn, idx_name) + idx_name + "_ref.nc"
                else:
                    p_idx = cfg.get_d_scen(stn, cfg.cat_idx, idx_name) +\
                            os.path.basename(p_sim[0][i_sim]).replace(vars[0], idx_name)
                if os.path.exists(p_idx) and (not cfg.opt_force_overwrite):
                    continue

                # Load datasets (one per variable).
                ds_scen = []
                for i_var in range(0, len(vars)):
                    var = vars[i_var]
                    ds = utils.open_netcdf(p_sim[i_var][i_sim])

                    # Adjust temperature units.
                    if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                        if ds[var].attrs[cfg.attrs_units] == cfg.unit_K:
                            ds[var] = ds[var] - cfg.d_KC
                        elif rcp == cfg.rcp_ref:
                            ds[var][cfg.attrs_units] = cfg.unit_C
                        ds[var].attrs[cfg.attrs_units] = cfg.unit_C

                    ds_scen.append(ds)

                # Indices ----------------------------------------------------------------------------------------------

                idx_units = None

                # ==========================================================
                # TODO.CUSTOMIZATION.INDEX.BEGIN
                # Calculate the index by copying the following code block.
                # ==========================================================

                # Merge threshold value and unit, if required. Ex: "0.0 C" for temperature.
                idx_threshs_str = []
                for i in range(len(idx_threshs)):
                    idx_thresh = idx_threshs[i]

                    # Temperature.
                    if (idx_name == cfg.idx_txdaysabove) or\
                       ((i == 0) and (idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen])) or \
                       ((i <= 1) and (idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen])):
                        idx_ref = str(idx_thresh) + " " + cfg.unit_C
                        idx_fut = str(idx_thresh + cfg.d_KC) + " " + cfg.unit_K
                        idx_threshs_str.append(idx_ref if (rcp == cfg.rcp_ref) else idx_fut)

                    # Precipitation.
                    elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                                      cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:
                        idx_threshs_str.append(str(idx_thresh) + " " + cfg.unit_mm + "/day")

                    else:
                        idx_threshs_str.append(str(idx_thresh))

                # Temperature.
                da_idx = None
                if idx_name == cfg.idx_txdaysabove:
                    da_tasmax = ds_scen[0][cfg.var_cordex_tasmax]
                    thresh_tasmax = idx_threshs_str[0]
                    da_idx = xr.DataArray(indices.tx_days_above(da_tasmax, thresh_tasmax).values)
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                elif idx_name in [cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen]:
                    da_tasmax = ds_scen[0][cfg.var_cordex_tasmax]
                    thresh_tasmax = idx_threshs_str[0]
                    thresh_ndays = int(idx_threshs_str[1])
                    if idx_name == cfg.idx_hotspellfreq:
                        da_idx = xr.DataArray(
                            indices.hot_spell_frequency(da_tasmax, thresh_tasmax, thresh_ndays).values)
                    else:
                        da_idx = xr.DataArray(
                            indices.hot_spell_max_length(da_tasmax, thresh_tasmax, thresh_ndays).values)
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                elif idx_name in [cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen]:
                    da_tasmin = ds_scen[0][cfg.var_cordex_tasmin]
                    da_tasmax = ds_scen[1][cfg.var_cordex_tasmax]
                    thresh_tasmin = idx_threshs_str[0]
                    thresh_tasmax = idx_threshs_str[1]
                    thresh_ndays = int(float(idx_threshs_str[2]))
                    if idx_name == cfg.idx_heatwavemaxlen:
                        da_idx = xr.DataArray(heat_wave_max_length(da_tasmin, da_tasmax, thresh_tasmin,
                                                                   thresh_tasmax, thresh_ndays).values)
                    else:
                        da_idx = xr.DataArray(heat_wave_total_length(da_tasmin, da_tasmax, thresh_tasmin,
                                                                     thresh_tasmax, thresh_ndays).values)
                    da_idx = da_idx.astype(int)
                    idx_units = cfg.unit_1

                # Precipitation.
                elif idx_name in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_prcptot]:
                    da_pr = ds_scen[0][cfg.var_cordex_pr]
                    if idx_name == cfg.idx_rx1day:
                        da_idx = xr.DataArray(indices.max_1day_precipitation_amount(da_pr, cfg.freq_YS))
                    elif idx_name == cfg.idx_rx5day:
                        da_idx = xr.DataArray(indices.max_n_day_precipitation_amount(da_pr, 5, cfg.freq_YS))
                    else:
                        da_idx = xr.DataArray(indices.precip_accumulation(da_pr, freq=cfg.freq_YS))
                    idx_units = da_idx.attrs[cfg.attrs_units]

                elif idx_name in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm,
                                  cfg.idx_wetdays, cfg.idx_drydays, cfg.idx_sdii]:
                    da_pr = ds_scen[0][cfg.var_cordex_pr]
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

                da_idx.attrs[cfg.attrs_units] = idx_units

                # ==========================================================
                # TODO.CUSTOMIZATION.INDEX.END
                # ==========================================================

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
                ds_idx = utils.copy_coordinates(ds_scen[0], ds_idx)
                ds_idx[idx_name] = utils.copy_coordinates(ds_scen[0][vars[0]], ds_idx[idx_name])
                ds_idx = ds_idx.squeeze()

                # Adjust calendar.
                year_1 = cfg.per_fut[0]
                year_n = cfg.per_fut[1]
                if rcp == cfg.rcp_ref:
                    year_1 = max(cfg.per_ref[0], int(str(ds_scen[0][cfg.dim_time][0].values)[0:4]))
                    year_n = min(cfg.per_ref[1],
                                 int(str(ds_scen[0][cfg.dim_time][len(ds_scen[0][cfg.dim_time]) - 1].values)[0:4]))
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

    cond = tasmin
    for t in range(len(cond.time)):
        cond[t] = (tasmin[t] > thresh_tasmin) & (tasmax[t] > thresh_tasmax)

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

    thresh_tasmax = convert_units_to(thresh_tasmax, tasmax)
    thresh_tasmin = convert_units_to(thresh_tasmin, tasmin)

    cond = tasmin
    for t in range(len(cond.time)):
        cond[t] = (tasmin[t] > thresh_tasmin) & (tasmax[t] > thresh_tasmax)

    group = cond.resample(time=freq)
    return group.map(rl.windowed_run_count, args=(window,), dim="time")


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
    if cfg.opt_stat[1] or (cfg.opt_ra and (cfg.opt_plot_heat[1] or cfg.opt_save_csv[1])):
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
    if cfg.opt_plot[1] or cfg.opt_plot_heat[1]:
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
    if cfg.opt_ra and (cfg.opt_plot_heat[1] or cfg.opt_save_csv[1]):

        utils.log(msg)

        for i in range(len(cfg.idx_names)):

            # Get the minimum and maximum values in the statistics file.
            idx = cfg.idx_names[i]
            p_stat = cfg.get_d_scen(cfg.obs_src, cfg.cat_stat, cfg.cat_idx + "/" + idx) +\
                idx + "_" + cfg.obs_src + ".csv"
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
            statistics.calc_heatmap(cfg.idx_names[i], cfg.idx_threshs[i], cfg.rcp_ref, [cfg.per_ref], z_min, z_max)

            # Future period.
            for rcp in cfg.rcps:
                statistics.calc_heatmap(cfg.idx_names[i], cfg.idx_threshs[i], rcp, cfg.per_hors, z_min, z_max)

    else:
        utils.log(msg + " (not required)")


if __name__ == "__main__":
    run()
