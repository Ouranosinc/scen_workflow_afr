# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Aggregation of a dataset (hourly to daily).
# The algorithm is only compatible with cfg.obs_src_era5 and cfg.obs_src_era5_land datasets.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import os
import plot as plot
import utils
import xarray as xr
import xclim.subset as subset


def aggregate(p_hour, p_day, set_name, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a NetCDF file by aggregating hourly to daily frequency.

    Parameters
    ----------
    p_hour : str
        Path of a NetCDF file containing hourly data (read).
    p_day : str
        Path of a NetCDF file containing daily data (written).
    set_name : str
        Set name.
    var : str
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    # DEBUG: Select the date that will be used to visualize data.
    opt_debug = False
    dbg_date = ""
    if set_name == cfg.obs_src_era5:
        dbg_date = "1979-01-01"
    elif set_name == cfg.obs_src_era5_land:
        dbg_date = "1981-01-01"
    dbg_latitude  = 0
    dbg_longitude = 0

    # Hourly data.
    ds_hour = xr.open_dataset(p_hour)[var]

    # Daily data.
    dir_day = os.path.dirname(p_day) + "/"
    fn_day  = os.path.basename(p_day)

    # Loop through statistics.
    for stat in [cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_sum]:

        # Output file name.
        var_stat = var + stat
        if var == cfg.var_era5_t2m and ((stat == cfg.stat_min) or (stat == cfg.stat_max)):
            p_day_stat = dir_day + var_stat + "/" + fn_day.replace(var + "_", var_stat + "_")
        else:
            p_day_stat = dir_day + var + "/" + fn_day

        # Aggregate only if output file does not exist.
        if not(os.path.exists(p_day_stat)):

            # Aggregation.
            ds_day = None
            save = False
            if stat == cfg.stat_mean:
                if (var == cfg.var_era5_d2m) or (var == cfg.var_era5_t2m) or\
                   (var == cfg.var_era5_sh) or (var == cfg.var_era5_u10) or (var == cfg.var_era5_v10):
                    ds_day = ds_hour.resample(time=cfg.freq_D).mean()
                    save = True
            elif stat == cfg.stat_min:
                if var == cfg.var_era5_t2m:
                    ds_day = ds_hour.resample(time=cfg.freq_D).min()
                    save = True
            elif stat == cfg.stat_max:
                if var == cfg.var_era5_t2m:
                    ds_day = ds_hour.resample(time=cfg.freq_D).max()
                    save = True
            elif stat == cfg.stat_sum:
                if (var == cfg.var_era5_tp) or (var == cfg.var_era5_e) or (var == cfg.var_era5_pev) or\
                        (var == cfg.var_era5_ssrd):
                    ds_day = ds_hour.resample(time=cfg.freq_D).sum()
                    save = True

            # Save NetCDF file.
            if save:
                utils.save_dataset(ds_day, p_day_stat)

        # Numerical test and plot for a given day of year.
        if opt_debug and os.path.exists(p_day_stat):

            ds_day = xr.open_dataset(p_day_stat)[var]

            # Plot #1: Time-series.
            # Hourly data.
            ds_hour_sel = subset.subset_gridpoint(ds_hour, lat=dbg_latitude, lon=dbg_longitude, tolerance=10 * 1000)
            # Daily data.
            ds_day_sel = subset.subset_gridpoint(ds_day, lat=dbg_latitude, lon=dbg_longitude, tolerance=10 * 1000)
            plot.plot_year(ds_hour_sel, ds_day_sel, set_name, var)

            # Verify values.
            # Hourly data.
            ds_hour_sel = ds_hour.sel(time=dbg_date, latitude=dbg_latitude, longitude=dbg_longitude)
            val_hour = ds_hour_sel.data.mean()
            # Daily data.
            ds_day_sel = ds_day.sel(time=dbg_date, latitude=dbg_latitude, longitude=dbg_longitude)
            val_day    = ds_day_sel.data.mean()

            # Clip to country boundaries.
            if cfg.d_bounds != "":
                ds_day = subset.subset_shape(ds_day, cfg.d_bounds)

            # Plot #2: Map.
            plot.plot_dayofyear(ds_day, set_name, var, dbg_date)


def calc_vapour_pressure(temperature):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate vapour pressure or saturation vapour pressure (hPa).
    Source: http://www.reahvac.com/tools/humidity-formulas/

    Parameters
    ----------
    temperature : float
        Temperature or dew temperature (C).

    Returns
    -------
        Vapour pressure (hPa)
        If a temperature is passed, saturation vapour pressure is calculated.
        If a dew point temperature is passed, actual vapour pressure is calculated.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate vapour pressure  (hPa).
    vapour_pressure = 6.11 * 10.0 ** (7.5 * temperature / (237.7 + temperature))

    return vapour_pressure


def calc_spec_humidity(temperature, pressure):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate specific humidity.
    Source: https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html#eq:2

    Parameters
    ----------
    temperature : float
        Temperature or dew temperature (C).
    pressure : float
        Atmospheric pressure (hPa).

    Returns
    -------
        Specific humidity (g/kg).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate vapour pressure (hPa).
    vapour_pressure = calc_vapour_pressure(temperature)

    # Calculate specific humidity.
    spec_humidity = (0.622 * vapour_pressure) / (pressure - 0.378 * vapour_pressure)

    return spec_humidity


def gen_dataset_sh(p_d2m, p_sp, p_sh, n_years):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a dataset for the variable "specific humidity".

    Parameters
    ----------
    p_d2m : str
        Path of hourly dataset containing temperature values.
    p_sp : str
        Path of hourly dataset containing atmospheric pressure values.
    p_sh : str
        Path of hourly dataset containing specific humidity values.
    n_years : int
        Number of years in the datasets.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    ds_d2m = xr.open_mfdataset(p_d2m, chunks={cfg.dim_time: n_years})[cfg.var_era5_d2m]
    ds_sp  = xr.open_dataset(p_sp, chunks={cfg.dim_time: n_years})[cfg.var_era5_sp]

    # DEBUG: Specific day.
    # date   = "1979-01-01"
    # ds_d2m = ds_d2m.sel(time=date)
    # ds_sp  = ds_sp.sel(time=date)

    # Calculate specific humidity values.
    ds_sh = calc_spec_humidity(ds_d2m - cfg.d_KC, ds_sp / 100.0)

    # Update meta information.
    ds_sh.name = cfg.var_era5_sh
    ds_sh[cfg.attrs_lname] = "specific humidity"
    ds_sh[cfg.attrs_units] = "1"
    ds_sh.attrs[cfg.attrs_lname] = "specific humidity"
    ds_sh.attrs[cfg.attrs_units] = "1"

    # Save NetCDF file.
    utils.save_dataset(ds_sh, p_sh)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List variables.
    vars = os.listdir(cfg.d_ra_raw)
    vars.sort()
    if not(cfg.var_era5_sh in vars):
        vars.append(cfg.var_era5_sh)

    # Loop through variables.
    for var in vars:

        # Loop through files.
        p_raw_lst = glob.glob(cfg.d_ra_raw + var + "/*.nc")
        p_raw_lst.sort()
        n_years = len(p_raw_lst)
        for p_raw in p_raw_lst:

            # Create an hourly dataset for specific humidity.
            # This requires cfg.var_era5_t2m and cfg.var_era5_sp.
            if var == cfg.var_era5_d2m:
                p_raw_d2m = p_raw
                p_raw_sp  = p_raw_d2m.replace(cfg.var_era5_d2m, cfg.var_era5_sp)
                p_raw_sh  = p_raw_d2m.replace(cfg.var_era5_d2m, cfg.var_era5_sh)
                if os.path.exists(p_raw_d2m) and os.path.exists(p_raw_sp) and not os.path.exists(p_raw_sh):
                    gen_dataset_sh(p_raw_d2m, p_raw_sp, p_raw_sh, n_years)

            # Perform aggregation.
            if var != cfg.var_era5_d2m:
                p_day = cfg.d_ra_day + os.path.basename(p_raw).replace("hour", "day")
                aggregate(p_raw, p_day, cfg.obs_src, var)


if __name__ == "__main__":
    run()
