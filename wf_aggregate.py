# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Aggregation of a dataset (hourly to daily).
# The algorithm is only compatible with cl_varidx.ens_era5 and cl_varidx.ens_era5_land datasets.
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
from typing import Optional

# Workflow libraries.
import wf_file_utils as fu
import wf_plot
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
import sys
sys.path.append("dashboard")
from dashboard.cl_varidx import VarIdx


def aggregate(
    p_hour: str,
    p_day: str,
    ens: str,
    var: VarIdx
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
    var: VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    # DEBUG: Select the date that will be used to visualize data.
    opt_debug = False
    if ens in [c.ENS_ERA5, c.ENS_ERA5_LAND]:
        dbg_date = str(cntx.opt_download_per[0]) + "-01-01"
    else:
        dbg_date = ""
    dbg_latitude  = 0
    dbg_longitude = 0

    # Daily data.
    dir_day = os.path.dirname(os.path.dirname(p_day)) + cntx.sep
    fn_day  = os.path.basename(p_day)

    # Loop through statistics.
    ds_hour = None
    for stat in [c.STAT_MEAN, c.STAT_MIN, c.STAT_MAX, c.STAT_SUM]:

        # Output file name.
        var_stat = var.name + stat
        if (var.name in [c.V_ECMWF_T2M, c.V_ECMWF_U10, c.V_ECMWF_V10, c.V_ECMWF_UV10]) and\
           (stat in [c.STAT_MIN, c.STAT_MAX]):
            p_day_stat = dir_day + var_stat + cntx.sep + fn_day.replace(var.name + "_", var_stat + "_")
        else:
            p_day_stat = dir_day + var.name + cntx.sep + fn_day

        # Aggregate only if output file does not exist.
        if (not os.path.exists(p_day_stat)) or cntx.opt_force_overwrite:

            # Hourly data.
            if ds_hour is None:
                ds_hour = fu.open_netcdf(p_hour)[var.name]

            # Aggregation.
            ds_day = None
            save = False
            if stat == c.STAT_MEAN:
                if var.name in [c.V_ECMWF_LSM, c.V_ECMWF_D2M, c.V_ECMWF_SH, c.V_ECMWF_T2M, c.V_ECMWF_U10,
                                c.V_ECMWF_V10, c.V_ECMWF_UV10]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=c.FREQ_D).mean()
                        if var.name == c.V_ECMWF_LSM:
                            ds_day = ds_day.resample(time=c.FREQ_YS).mean().squeeze()
                    save = True
            elif stat == c.STAT_MIN:
                if var.name == c.V_ECMWF_T2M:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=c.FREQ_D).min()
                    save = True
            elif stat == c.STAT_MAX:
                if var.name in [c.V_ECMWF_T2M, c.V_ECMWF_U10, c.V_ECMWF_V10, c.V_ECMWF_UV10]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        ds_day = ds_hour.resample(time=c.FREQ_D).max()
                    save = True
            elif stat == c.STAT_SUM:
                if var.name in [c.V_ECMWF_TP, c.V_ECMWF_E, c.V_ECMWF_PEV, c.V_ECMWF_SSRD]:
                    if cntx.obs_src == c.ENS_ERA5:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            ds_day = ds_hour.resample(time=c.FREQ_D).sum()
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            ds_day = ds_hour.sel(time=datetime.time(23)).resample(time=c.FREQ_D).sum()
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
            wf_plot.plot_year(ds_hour_sel, ds_day_sel, var)

            # Plot #2: Map.
            wf_plot.plot_dayofyear(ds_day, var, dbg_date)


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
    da_d2m = fu.open_netcdf(p_d2m, chunks={c.DIM_TIME: n_years})[c.V_ECMWF_D2M]
    da_sp  = fu.open_netcdf(p_sp, chunks={c.DIM_TIME: n_years})[c.V_ECMWF_SP]

    # Calculate specific humidity values.
    da_sh = xr.DataArray(calc_spec_humidity(da_d2m - c.d_KC, da_sp / 100.0))

    # Update meta information.
    da_sh.name = c.V_ECMWF_SH
    da_sh.attrs[c.ATTRS_LNAME] = "specific humidity"
    da_sh.attrs[c.ATTRS_UNITS] = c.UNIT_1
    da_sh.attrs[c.ATTRS_LNAME] = "specific humidity"
    da_sh.attrs[c.ATTRS_UNITS] = c.UNIT_1

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
    da_u10 = fu.open_netcdf(p_u10, chunks={c.DIM_TIME: n_years})[c.V_ECMWF_U10]
    da_v10  = fu.open_netcdf(p_v10, chunks={c.DIM_TIME: n_years})[c.V_ECMWF_V10]

    # Calculate specific humidity values.
    da_uv10 = ((da_u10 ** 2) + (da_v10 ** 2)) ** 0.5

    # Update meta information.
    da_uv10.name = c.V_ECMWF_UV10
    da_uv10.attrs[c.ATTRS_LNAME] = "wind"
    da_uv10.attrs[c.ATTRS_UNITS] = c.UNIT_m_s
    da_uv10.attrs[c.ATTRS_LNAME] = "wind"
    da_uv10.attrs[c.ATTRS_UNITS] = c.UNIT_m_s

    # Save NetCDF file.
    fu.save_netcdf(da_uv10, p_uv10)


def p_ra(
    var_name: str,
    freq: str,
    year: int,
    stat_name: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the path of ECMWF reanalysis data file.

    Parameters
    ----------
    var_name: str
        Variable name.
    freq: str
        Frequency = {"c.FREQ_H", "c.FREQ_D"}
    year: int
        Year.
    stat_name: Optional[str]
        Statistics name = {c.STAT_MEAN, c.STAT_MIN, c.STAT_MAX, c.STAT_SUM}
    --------------------------------------------------------------------------------------------------------------------
    """

    return (cntx.d_ra_raw if freq == c.FREQ_H else cntx.d_ra_day) + var_name + stat_name + cntx.sep + \
        var_name + stat_name + "_" + cntx.obs_src + "_" + ("day" if freq == c.FREQ_D else "hour") + "_" +\
        str(year) + c.F_EXT_NC


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List variables.
    var_name_l = []
    for var in cntx.vars.items:
        if var.name == c.V_HUSS:
            var_name_l.append(c.V_ECMWF_D2M)
            var_name_l.append(c.V_ECMWF_SP)
        elif var.name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
            if c.V_ECMWF_T2M not in var_name_l:
                var_name_l.append(c.V_ECMWF_T2M)
        elif var.name == c.V_SFCWINDMAX:
            var_name_l.append(c.V_ECMWF_U10)
            var_name_l.append(c.V_ECMWF_V10)
        else:
            var_ra_name = VarIdx(var.name).convert_name(c.ENS_ERA5)
            if var_ra_name is not None:
                var_name_l.append(var_ra_name)
    var_name_l.sort()

    # Loop through variables.
    for var_name in var_name_l:

        # Skip if variable does not exist in the current dataset.
        if (cntx.obs_src == c.ENS_ERA5_LAND) and (var_name == c.V_ECMWF_LSM):
            continue

        # Loop through files.
        p_raw_lst = glob.glob(cntx.d_ra_raw + var_name + cntx.sep + "*" + c.F_EXT_NC)
        p_raw_lst.sort()
        n_years = len(p_raw_lst)
        for i_raw in range(len(p_raw_lst)):

            # Path of hourly data.
            p_raw = p_raw_lst[i_raw]

            # Extract year.
            tokens = p_raw.replace(c.F_EXT_NC, "").split("_")
            year = int(tokens[len(tokens) - 1])

            # Path of daily data.
            p_day = p_ra(var_name, c.FREQ_D, year)

            fu.log("Processing: " + p_raw, True)

            # Perform aggregation (if not all files exist).
            f_exist = os.path.exists(p_ra(var_name, c.FREQ_D, year))
            if var_name == c.V_ECMWF_T2M:
                p_day_t2mmin = p_ra(c.V_ECMWF_T2MMIN, c.FREQ_D, year)
                p_day_t2mmax = p_ra(c.V_ECMWF_T2MMAX, c.FREQ_D, year)
                f_exist = f_exist and os.path.exists(p_day_t2mmin) and os.path.exists(p_day_t2mmax)
            elif var_name == c.V_ECMWF_U10:
                p_day_u10max = p_ra(c.V_ECMWF_U10MAX, c.FREQ_D, year)
                f_exist = f_exist and os.path.exists(p_day_u10max)
            elif var_name == c.V_ECMWF_V10:
                p_day_v10max = p_ra(c.V_ECMWF_V10MAX, c.FREQ_D, year)
                f_exist = f_exist and os.path.exists(p_day_v10max)
            if (not f_exist) or cntx.opt_force_overwrite:
                aggregate(p_raw, p_day, cntx.obs_src, VarIdx(var_name))

            # Calculate specific humidity.
            if var_name in [c.V_ECMWF_D2M, c.V_ECMWF_SP]:
                p_raw_d2m = p_ra(c.V_ECMWF_D2M, c.FREQ_H, year)
                p_raw_sp  = p_ra(c.V_ECMWF_SP, c.FREQ_H, year)
                p_raw_sh  = p_ra(c.V_ECMWF_SH, c.FREQ_H, year)
                p_day_sh  = p_ra(c.V_ECMWF_SH, c.FREQ_D, year)
                if os.path.exists(p_raw_d2m) and os.path.exists(p_raw_sp) and\
                   ((not os.path.exists(p_raw_sh)) or cntx.opt_force_overwrite):
                    gen_dataset_sh(p_raw_d2m, p_raw_sp, p_raw_sh, n_years)
                if os.path.exists(p_raw_sh) and (not os.path.exists(p_day_sh) or cntx.opt_force_overwrite):
                    aggregate(p_raw_sh, p_day_sh, cntx.obs_src, VarIdx(c.V_ECMWF_SH))

            # Calculate wind speed.
            if var_name in [c.V_ECMWF_U10, c.V_ECMWF_V10]:
                p_raw_u10  = p_ra(c.V_ECMWF_U10, c.FREQ_H, year)
                p_raw_v10  = p_ra(c.V_ECMWF_V10, c.FREQ_H, year)
                p_raw_uv10 = p_ra(c.V_ECMWF_UV10, c.FREQ_H, year)
                p_day_uv10 = p_ra(c.V_ECMWF_UV10, c.FREQ_D, year)
                p_day_uv10max = p_ra(c.V_ECMWF_UV10MAX, c.FREQ_D, year)
                if os.path.exists(p_raw_u10) and os.path.exists(p_raw_v10) and\
                   ((not os.path.exists(p_raw_uv10)) or cntx.opt_force_overwrite):
                    gen_dataset_uv10(p_raw_u10, p_raw_v10, p_raw_uv10, n_years)
                if (os.path.exists(p_raw_uv10) and
                    ((not os.path.exists(p_day_uv10) or (not os.path.exists(p_day_uv10max))) or
                     cntx.opt_force_overwrite)):
                    aggregate(p_raw_uv10, p_day_uv10, cntx.obs_src, VarIdx(c.V_ECMWF_UV10))


if __name__ == "__main__":
    run()
