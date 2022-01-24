# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Aggregation of a dataset (hourly to daily).
# The algorithm is only compatible with def_varidx.ens_era5 and def_varid.ens_era5_land datasets.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import clisops.core.subset as subset
import datetime
import glob
import os
import xarray as xr
import warnings

# Workflow libraries.
import file_utils as fu
import plot
from def_constant import const as c
from def_context import cntx

# Dashboard libraries.
import sys
sys.path.append("dashboard")
from dashboard import def_varidx as vi


def aggregate(
    p_hour: str,
    p_day: str,
    ens: str,
    var: vi.VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a NetCDF file by aggregating hourly to daily frequency.

    Parameters
    ----------
    p_hour: str
        Path of a NetCDF file containing hourly data (read).
    p_day: str
        Path of a NetCDF file containing daily data (written).
    ens: str
        Ensemble.
    var: vi.VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    # DEBUG: Select the date that will be used to visualize data.
    opt_debug = False
    dbg_date = ""
    if ens == c.ens_era5:
        dbg_date = "1979-01-01"
    elif ens == c.ens_era5_land:
        dbg_date = "1981-01-01"
    dbg_latitude  = 0
    dbg_longitude = 0

    # Hourly data.
    ds_hour = fu.open_netcdf(p_hour)[var.name]

    # Daily data.
    dir_day = os.path.dirname(p_day) + cntx.sep
    fn_day  = os.path.basename(p_day)

    # Loop through statistics.
    for stat in [c.stat_mean, c.stat_min, c.stat_max, c.stat_sum]:

        # Output file name.
        var_stat = var.name + stat
        if (var.name in [c.v_era5_t2m, c.v_era5_u10, c.v_era5_v10, c.v_era5_uv10]) and\
           (stat in [c.stat_min, c.stat_max]):
            p_day_stat = dir_day + var_stat + cntx.sep + fn_day.replace(var.name + "_", var_stat + "_")
        else:
            p_day_stat = dir_day + var.name + cntx.sep + fn_day

        # Aggregate only if output file does not exist.
        if (not os.path.exists(p_day_stat)) or cntx.opt_force_overwrite:

            # Aggregation.
            ds_day = None
            save = False
            if stat == c.stat_mean:
                if (var.name in [c.v_era5_d2m, c.v_era5_sh]) or\
                   ((var.name == c.v_era5_t2m) and (c.v_tas in cntx.vars.code_l)) or\
                   (var.name == c.v_era5_u10) or (var.name == c.v_era5_v10):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=c.freq_D).mean()
                    save = True
            elif stat == c.stat_min:
                if (var.name == c.v_era5_t2m) and (c.v_tasmin in cntx.vars.code_l):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=c.freq_D).min()
                    save = True
            elif stat == c.stat_max:
                if ((var.name == c.v_era5_t2m) and (c.v_tasmax in cntx.vars.code_l)) or\
                   ((var.name == c.v_era5_uv10) and (c.v_sfcwindmax in cntx.vars.code_l)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=c.freq_D).max()
                    save = True
            elif stat == c.stat_sum:
                if (var.name in [c.v_era5_tp, c.v_era5_e, c.v_era5_pev]) or (var.name == c.v_era5_ssrd):
                    if cntx.obs_src == c.ens_era5:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            ds_day = ds_hour.resample(time=c.freq_D).sum()
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            ds_day = ds_hour.sel(time=datetime.time(23)).resample(time=c.freq_D).sum()
                    save = True

            # Save NetCDF file.
            if save:
                fu.save_netcdf(ds_day, p_day_stat)

        # Numerical test and plot for a given day of year.
        if opt_debug and os.path.exists(p_day_stat):

            ds_day = fu.open_netcdf(p_day_stat)[var.name]

            # Plot #1: Time-series.
            # Hourly data.
            ds_hour_sel = subset.subset_gridpoint(ds_hour, lat=dbg_latitude, lon=dbg_longitude, tolerance=10 * 1000)
            # Daily data.
            ds_day_sel = subset.subset_gridpoint(ds_day, lat=dbg_latitude, lon=dbg_longitude, tolerance=10 * 1000)
            plot.plot_year(ds_hour_sel, ds_day_sel, var)

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
    da_vp = xr.DataArray(calc_vapour_pressure(da_tas))

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
    da_d2m = fu.open_netcdf(p_d2m, chunks={c.dim_time: n_years})[c.v_era5_d2m]
    da_sp  = fu.open_netcdf(p_sp, chunks={c.dim_time: n_years})[c.v_era5_sp]

    # Calculate specific humidity values.
    da_sh = xr.DataArray(calc_spec_humidity(da_d2m - c.d_KC, da_sp / 100.0))

    # Update meta information.
    da_sh.name = c.v_era5_sh
    da_sh.attrs[c.attrs_lname] = "specific humidity"
    da_sh.attrs[c.attrs_units] = c.unit_1
    da_sh.attrs[c.attrs_lname] = "specific humidity"
    da_sh.attrs[c.attrs_units] = c.unit_1

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
    da_u10 = fu.open_netcdf(p_u10, chunks={c.dim_time: n_years})[c.v_era5_u10]
    da_v10  = fu.open_netcdf(p_v10, chunks={c.dim_time: n_years})[c.v_era5_v10]

    # Calculate specific humidity values.
    da_uv10 = ((da_u10 ** 2) + (da_v10 ** 2)) ** 0.5

    # Update meta information.
    da_uv10.name = c.v_era5_uv10
    da_uv10.attrs[c.attrs_lname] = "wind"
    da_uv10.attrs[c.attrs_units] = c.unit_m_s
    da_uv10.attrs[c.attrs_lname] = "wind"
    da_uv10.attrs[c.attrs_units] = c.unit_m_s

    # Save NetCDF file.
    fu.save_netcdf(da_uv10, p_uv10)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List variables.
    var_name_l = []
    for var in cntx.vars.items:
        if var.name == c.v_huss:
            var_name_l.append(c.v_era5_d2m)
            var_name_l.append(c.v_era5_sp)
        elif var.name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
            if c.v_era5_t2m not in var_name_l:
                var_name_l.append(c.v_era5_t2m)
        elif var.name == c.v_sfcwindmax:
            var_name_l.append(c.v_era5_u10)
            var_name_l.append(c.v_era5_v10)
        else:
            var_ra_name = vi.VarIdx(var.name).convert_name(c.ens_era5)
            if var_ra_name is not None:
                var_name_l.append(var_ra_name)

    # Loop through variables.
    for var_name in var_name_l:

        # Loop through files.
        p_raw_lst = glob.glob(cntx.d_ra_raw + var_name + cntx.sep + "*" + c.f_ext_nc)
        p_raw_lst.sort()
        n_years = len(p_raw_lst)
        for i_raw in range(len(p_raw_lst)):
            p_raw = p_raw_lst[i_raw]
            p_day = cntx.d_ra_day + os.path.basename(p_raw).replace("hour", "day")

            fu.log("Processing: " + p_raw, True)

            # Perform aggregation.
            if (not os.path.exists(p_day)) or cntx.opt_force_overwrite:
                aggregate(p_raw, p_day, cntx.obs_src, vi.VarIdx(var_name))

            # Calculate specific humidity.
            if var_name in [c.v_era5_d2m, c.v_era5_sp]:
                if var_name == c.v_era5_d2m:
                    p_raw_d2m = p_raw
                    p_raw_sp  = p_raw.replace(c.v_era5_d2m, c.v_era5_sp)
                else:
                    p_raw_sp  = p_raw
                    p_raw_d2m = p_raw.replace(c.v_era5_sp, c.v_era5_d2m)
                p_raw_sh = p_raw_sp.replace(c.v_era5_sp, c.v_era5_sh)
                p_day_sh = cntx.d_ra_day + os.path.basename(p_raw_sh).replace("hour", "day")
                if os.path.exists(p_raw_d2m) and os.path.exists(p_raw_sp) and\
                   ((not os.path.exists(p_raw_sh)) or cntx.opt_force_overwrite):
                    gen_dataset_sh(p_raw_d2m, p_raw_sp, p_raw_sh, n_years)
                if os.path.exists(p_raw_sh) and (not os.path.exists(p_day_sh) or cntx.opt_force_overwrite):
                    aggregate(p_raw_sh, p_day_sh, cntx.obs_src, vi.VarIdx(c.v_era5_sh))

            # Calculate wind speed.
            if var_name in [c.v_era5_u10, c.v_era5_v10]:
                if var_name == c.v_era5_u10:
                    p_raw_u10 = p_raw
                    p_raw_v10 = p_raw.replace(c.v_era5_u10, c.v_era5_v10)
                else:
                    p_raw_v10 = p_raw
                    p_raw_u10 = p_raw.replace(c.v_era5_v10, c.v_era5_u10)
                p_raw_uv10 = p_raw_v10.replace(c.v_era5_v10, c.v_era5_uv10)
                p_day_uv10 = cntx.d_ra_day + os.path.basename(p_raw_uv10).replace("hour", "day")
                if os.path.exists(p_raw_u10) and os.path.exists(p_raw_v10) and\
                   ((not os.path.exists(p_raw_uv10)) or cntx.opt_force_overwrite):
                    gen_dataset_uv10(p_raw_u10, p_raw_v10, p_raw_uv10, n_years)
                if os.path.exists(p_raw_uv10) and (not os.path.exists(p_day_uv10) or cntx.opt_force_overwrite):
                    aggregate(p_raw_uv10, p_day_uv10, cntx.obs_src, vi.VarIdx(c.v_era5_uv10))


if __name__ == "__main__":
    run()
