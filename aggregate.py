# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Aggregation of a dataset (hourly to daily).
# The algorithm is only compatible with def_varidx.ens_era5 and def_varid.ens_era5_land datasets.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import constants as const
import clisops.core.subset as subset
import datetime
import file_utils as fu
import glob
import os
import plot as plot
import xarray as xr
import warnings
from config import cfg

import sys
sys.path.append("dashboard")
from dashboard import def_stat, def_varidx as vi


def aggregate(
    p_hour: str,
    p_day: str,
    set_name: str,
    var: str
):

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
    if set_name == vi.ens_era5:
        dbg_date = "1979-01-01"
    elif set_name == vi.ens_era5_land:
        dbg_date = "1981-01-01"
    dbg_latitude  = 0
    dbg_longitude = 0

    # Hourly data.
    ds_hour = fu.open_netcdf(p_hour)[var]

    # Daily data.
    dir_day = os.path.dirname(p_day) + cfg.sep
    fn_day  = os.path.basename(p_day)

    # Loop through statistics.
    for stat in [def_stat.code_mean, def_stat.code_min, def_stat.code_max, def_stat.code_sum]:

        # Output file name.
        var_stat = var + stat
        if (var in [vi.v_era5_t2m, vi.v_era5_u10, vi.v_era5_v10, vi.v_era5_uv10]) and\
           (stat in [def_stat.code_min, def_stat.code_max]):
            p_day_stat = dir_day + var_stat + cfg.sep + fn_day.replace(var + "_", var_stat + "_")
        else:
            p_day_stat = dir_day + var + cfg.sep + fn_day

        # Aggregate only if output file does not exist.
        if (not os.path.exists(p_day_stat)) or cfg.opt_force_overwrite:

            # Aggregation.
            ds_day = None
            save = False
            if stat == def_stat.code_mean:
                if (var in [vi.v_era5_d2m, vi.v_era5_sh]) or\
                   ((var == vi.v_era5_t2m) and (vi.v_tas in cfg.variables)) or\
                   (var == vi.v_era5_u10) or (var == vi.v_era5_v10):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=const.freq_D).mean()
                    save = True
            elif stat == def_stat.code_min:
                if (var == vi.v_era5_t2m) and (vi.v_tasmin in cfg.variables):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=const.freq_D).min()
                    save = True
            elif stat == def_stat.code_max:
                if ((var == vi.v_era5_t2m) and (vi.v_tasmax in cfg.variables)) or\
                   ((var == vi.v_era5_uv10) and (vi.v_sfcwindmax in cfg.variables)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=const.freq_D).max()
                    save = True
            elif stat == def_stat.code_sum:
                if (var in [vi.v_era5_tp, vi.v_era5_e, vi.v_era5_pev]) or (var == vi.v_era5_ssrd):
                    if cfg.obs_src == vi.ens_era5:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            ds_day = ds_hour.resample(time=const.freq_D).sum()
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            ds_day = ds_hour.sel(time=datetime.time(23)).resample(time=const.freq_D).sum()
                    save = True

            # Save NetCDF file.
            if save:
                fu.save_netcdf(ds_day, p_day_stat)

        # Numerical test and plot for a given day of year.
        if opt_debug and os.path.exists(p_day_stat):

            ds_day = fu.open_netcdf(p_day_stat)[var]

            # Plot #1: Time-series.
            # Hourly data.
            ds_hour_sel = subset.subset_gridpoint(ds_hour, lat=dbg_latitude, lon=dbg_longitude, tolerance=10 * 1000)
            # Daily data.
            ds_day_sel = subset.subset_gridpoint(ds_day, lat=dbg_latitude, lon=dbg_longitude, tolerance=10 * 1000)
            plot.plot_year(ds_hour_sel, ds_day_sel, var)

            # Verify values.
            # Hourly data.
            # DEBUG: ds_hour_sel = ds_hour.sel(time=dbg_date, latitude=dbg_latitude, longitude=dbg_longitude)
            # DEBUG: val_hour = ds_hour_sel.data.mean()
            # Daily data.
            # DEBUG: ds_day_sel = ds_day.sel(time=dbg_date, latitude=dbg_latitude, longitude=dbg_longitude)
            # DEBUG: val_day    = ds_day_sel.data.mean()

            # Plot #2: Map.
            plot.plot_dayofyear(ds_day, var, dbg_date)


def calc_vapour_pressure(
    da_tas: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate vapour pressure or saturation vapour pressure (hPa).
    Source: http://www.reahvac.com/tools/humidity-formulas/

    Parameters
    ----------
    da_tas : xr.DataArray
        Temperature or dew temperature (C).

    Returns
    -------
    Vapour pressure (hPa)
    If a temperature is passed, saturation vapour pressure is calculated.
    If a dew point temperature is passed, actual vapour pressure is calculated.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate vapour pressure  (hPa).
    da_vp = 6.11 * 10.0 ** (7.5 * da_tas / (237.7 + da_tas))

    return da_vp


def calc_spec_humidity(
    da_tas: xr.DataArray,
    da_ps: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate specific humidity.
    Source: https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html#eq:2

    Parameters
    ----------
    da_tas : xr.DataArray
        Temperature or dew temperature (C).
    da_ps : xr.DataArray
        Barometric pressure (hPa).

    Returns
    -------
    Specific humidity (g/kg).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate vapour pressure (hPa).
    da_vp = calc_vapour_pressure(da_tas)

    # Calculate specific humidity.
    da_sh = (0.622 * da_vp) / (da_ps - 0.378 * da_vp)

    return da_sh


def gen_dataset_sh(
    p_d2m: str,
    p_sp: str,
    p_sh: str,
    n_years: int
):

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
    da_d2m = fu.open_netcdf(p_d2m, chunks={const.dim_time: n_years})[vi.v_era5_d2m]
    da_sp  = fu.open_netcdf(p_sp, chunks={const.dim_time: n_years})[vi.v_era5_sp]

    # Calculate specific humidity values.
    da_sh = calc_spec_humidity(da_d2m - const.d_KC, da_sp / 100.0)

    # Update meta information.
    da_sh.name = vi.v_era5_sh
    da_sh.attrs[const.attrs_lname] = "specific humidity"
    da_sh.attrs[const.attrs_units] = const.unit_1
    da_sh.attrs[const.attrs_lname] = "specific humidity"
    da_sh.attrs[const.attrs_units] = const.unit_1

    # Save NetCDF file.
    fu.save_netcdf(da_sh, p_sh)


def gen_dataset_uv10(
    p_u10: str,
    p_v10: str,
    p_uv10: str,
    n_years: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a dataset for the variable "specific humidity".

    Parameters
    ----------
    p_u10 : str
        Path of hourly dataset containing u-component wind.
    p_v10 : str
        Path of hourly dataset containing v-component wind.
    p_uv10 : str
        Path of hourly dataset containing wind.
    n_years : int
        Number of years in the datasets.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    da_u10 = fu.open_netcdf(p_u10, chunks={const.dim_time: n_years})[vi.v_era5_u10]
    da_v10  = fu.open_netcdf(p_v10, chunks={const.dim_time: n_years})[vi.v_era5_v10]

    # Calculate specific humidity values.
    da_uv10 = ((da_u10 ** 2) + (da_v10 ** 2)) ** 0.5

    # Update meta information.
    da_uv10.name = vi.v_era5_uv10
    da_uv10.attrs[const.attrs_lname] = "wind"
    da_uv10.attrs[const.attrs_units] = const.unit_m_s
    da_uv10.attrs[const.attrs_lname] = "wind"
    da_uv10.attrs[const.attrs_units] = const.unit_m_s

    # Save NetCDF file.
    fu.save_netcdf(da_uv10, p_uv10)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List variables.
    var_l = []
    for var in cfg.variables:
        if var == vi.v_huss:
            var_l.append(vi.v_era5_d2m)
            var_l.append(vi.v_era5_sp)
        elif var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
            if vi.v_era5_t2m not in var_l:
                var_l.append(vi.v_era5_t2m)
        elif var == vi.v_sfcwindmax:
            var_l.append(vi.v_era5_u10)
            var_l.append(vi.v_era5_v10)
        else:
            var_ra = vi.VarIdx(var).convert_name(vi.ens_era5)
            if var_ra is not None:
                var_l.append(var_ra)

    # Loop through variables.
    for var in var_l:

        # Loop through files.
        p_raw_lst = glob.glob(cfg.d_ra_raw + var + cfg.sep + "*" + fu.f_ext_nc)
        p_raw_lst.sort()
        n_years = len(p_raw_lst)
        for i_raw in range(len(p_raw_lst)):
            p_raw = p_raw_lst[i_raw]
            p_day = cfg.d_ra_day + os.path.basename(p_raw).replace("hour", "day")

            fu.log("Processing: " + p_raw, True)

            # Perform aggregation.
            if (not os.path.exists(p_day)) or cfg.opt_force_overwrite:
                aggregate(p_raw, p_day, cfg.obs_src, var)

            # Calculate specific humidity.
            if (var == vi.v_era5_d2m) or (var == vi.v_era5_sp):
                if var == vi.v_era5_d2m:
                    p_raw_d2m = p_raw
                    p_raw_sp  = p_raw.replace(vi.v_era5_d2m, vi.v_era5_sp)
                else:
                    p_raw_sp  = p_raw
                    p_raw_d2m = p_raw.replace(vi.v_era5_sp, vi.v_era5_d2m)
                p_raw_sh = p_raw_sp.replace(vi.v_era5_sp, vi.v_era5_sh)
                p_day_sh = cfg.d_ra_day + os.path.basename(p_raw_sh).replace("hour", "day")
                if os.path.exists(p_raw_d2m) and os.path.exists(p_raw_sp) and\
                   ((not os.path.exists(p_raw_sh)) or cfg.opt_force_overwrite):
                    gen_dataset_sh(p_raw_d2m, p_raw_sp, p_raw_sh, n_years)
                if os.path.exists(p_raw_sh) and (not os.path.exists(p_day_sh) or cfg.opt_force_overwrite):
                    aggregate(p_raw_sh, p_day_sh, cfg.obs_src, vi.v_era5_sh)

            # Calculate wind speed.
            if (var == vi.v_era5_u10) or (var == vi.v_era5_v10):
                if var == vi.v_era5_u10:
                    p_raw_u10 = p_raw
                    p_raw_v10 = p_raw.replace(vi.v_era5_u10, vi.v_era5_v10)
                else:
                    p_raw_v10 = p_raw
                    p_raw_u10 = p_raw.replace(vi.v_era5_v10, vi.v_era5_u10)
                p_raw_uv10 = p_raw_v10.replace(vi.v_era5_v10, vi.v_era5_uv10)
                p_day_uv10 = cfg.d_ra_day + os.path.basename(p_raw_uv10).replace("hour", "day")
                if os.path.exists(p_raw_u10) and os.path.exists(p_raw_v10) and\
                   ((not os.path.exists(p_raw_uv10)) or cfg.opt_force_overwrite):
                    gen_dataset_uv10(p_raw_u10, p_raw_v10, p_raw_uv10, n_years)
                if os.path.exists(p_raw_uv10) and (not os.path.exists(p_day_uv10) or cfg.opt_force_overwrite):
                    aggregate(p_raw_uv10, p_day_uv10, cfg.obs_src, vi.v_era5_uv10)


if __name__ == "__main__":
    run()
