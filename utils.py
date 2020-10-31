# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import clisops.core.subset as subset
import config as cfg
import datetime
import glob
import math
import matplotlib.pyplot
import numpy as np
import os
import pandas as pd
import re
import xarray as xr
import warnings
from cmath import rect, phase
from collections import defaultdict
from itertools import compress
from math import radians, degrees, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Union, List, Tuple


def natural_sort(values: Union[float, int]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Sorts a list numerically ('sorted([1,2,10,11]) would result in [1,10,11,2]).

    Parameters
    ----------
    values : List of values that require numerical sorting.
    --------------------------------------------------------------------------------------------------------------------
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(values, key=alphanum_key)


def uas_vas_2_sfc(uas: xr.DataArray, vas: xr.DataArray):

    """
    --------------------------------------------------------------------------------------------------------------------
    Transforms wind components to sfcwind and direction.
    TODO: Support other data formats.

    Parameters
    ----------
    uas : xr.DataArray
        Wind component (x-axis).
    vas : xr.DataArray
        Wind component (y-axis).
    --------------------------------------------------------------------------------------------------------------------
    """

    sfcwind = np.sqrt(np.square(uas) + np.square(vas))

    # Calculate the angle, then aggregate to daily.
    sfcwind_dirmath = np.degrees(np.arctan2(vas, uas))
    sfcwind_dirmet = (270 - sfcwind_dirmath) % 360.0

    # Calm winds have a direction of 0°. Northerly winds have a direction of 360°.
    # According to the Beaufort scale, "calm" winds are < 0.5 m/s
    sfcwind_dirmet.values[(sfcwind_dirmet.round() == 0) & (sfcwind >= 0.5)] = 360
    sfcwind_dirmet.values[sfcwind < 0.5] = 0

    return sfcwind, sfcwind_dirmet


def sfcwind_2_uas_vas(sfcwind: xr.DataArray, winddir: np.array, resample=None, nb_per_day=None):

    """
    --------------------------------------------------------------------------------------------------------------------
    Transforms sfcWind and direction as the wind components uas and vas
    TODO: Support other data formats.

    Parameters
    ----------
    sfcwind : xr.DataArray
        Wind speed.
    winddir : np.array
        Direction from which the wind blows (using the meteorological standard).
    resample : str
        Whether or not the data needs to be resampled.
    nb_per_day : int
        Number of time steps per day.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Transform wind direction from the meteorological standard to the mathematical standard.
    sfcwind_dirmath = (-winddir + 270) % 360.0

    daily_avg_angle = 0.0
    if resample is not None:

        sfcwind = sfcwind.resample(time=resample).mean(dim=cfg.dim_time, keep_attrs=True)

        # TODO.MAB: Remove nb_per_day and calculate it.

        # TODO.MAB: Improve the following line because it is very dirty.
        sfcwind_angle_per_day = sfcwind_dirmath.reshape((len(sfcwind.time), nb_per_day))

        # TODO.MAB: Improve the following line because it is also very dirty.
        daily_avg_angle = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
                                          for angles in sfcwind_angle_per_day])

    uas = sfcwind.values * np.cos(np.radians(daily_avg_angle))
    vas = sfcwind.values * np.sin(np.radians(daily_avg_angle))

    return uas, vas


def list_cordex(p_ds: str, rcps: [str]):

    """
    --------------------------------------------------------------------------------------------------------------------
     Lists CORDEX simulations.

    Parameters
    ----------
    p_ds : str
        Path of data_source.
    rcps : [str]
        List of RCP scenarios.
    --------------------------------------------------------------------------------------------------------------------
    """

    list_f = {}

    # Find all the available simulations for a given RCP.
    for r in range(len(rcps)):

        d_format = p_ds + "*/*/AFR-*{r}".format(r=rcps[r]) + "/*/atmos/*/"
        d_list = glob.glob(d_format)
        d_list = [i for i in d_list if "day" in i]
        d_list.sort()

        # Remove timestep information.
        for i in range(0, len(d_list)):
            tokens = d_list[i].split("/")
            d_list[i] = d_list[i].replace(tokens[len(tokens) - 4], "*")

        # Keep only the unique simulation folders (with a * as the timestep).
        d_list = list(set(d_list))
        d_list_valid = [True] * len(d_list)

        # Keep only the simulations with all the variables we need.
        d_list = list(compress(d_list, d_list_valid))

        list_f[rcps[r] + "_historical"] = [w.replace(rcps[r], "historical") for w in d_list]
        list_f[rcps[r]] = d_list

    return list_f


def info_cordex(d_ds: str):

    """
    --------------------------------------------------------------------------------------------------------------------
     Creates an array that contains information about CORDEX simulations.
     A verification is made to ensure that there is at least one NetCDF file available for each variable.

    Parameters
    ----------
    d_ds : str
        Directory of data_source.

    Return
    ------
    sets : [str, str, str, str, [str]]
        List of simulation sets.
        sets[0] is the institute that created this simulation.
        sets[1] is the regional circulation model (RCM).
        sets[2] is the global circulation model (GCM).
        sets[3] is the simulation name.
        sets[4] is a list of variables available.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Results.
    sets = []

    # List directories containing simulation sets.
    d_format = d_ds + "*/*/AFR-*/day/atmos/*/"
    n_token = len(d_ds.split("/")) - 2

    # Loop through simulations sets.
    for i in glob.glob(d_format):
        tokens_i = i.split("/")

        # Extract institute, RGM, CGM and emission scenario.
        inst = tokens_i[n_token + 1]
        rgm  = tokens_i[n_token + 2]
        cgm  = tokens_i[n_token + 3].split("_")[1]
        scen = tokens_i[n_token + 3].split("_")[2]

        # Extract variables and ensure that there is at least one NetCDF file available for each  one.
        vars_i = []
        for j in glob.glob(i + "*/"):
            n_netcdf = len(glob.glob(j + "*.nc"))
            if n_netcdf > 0:
                tokens_j = j.split("/")
                var      = tokens_j[len(tokens_j) - 2]
                vars_i.append(var)

        sets.append([inst, rgm, cgm, scen, vars_i])

    return sets


def calendar(x: Union[xr.Dataset, xr.DataArray], n_days_old=360, n_days_new=365):

    """
    --------------------------------------------------------------------------------------------------------------------
    Interpolates data from a 360- to 365-day calendar.

    Parameters
    ----------
    x : xr.Dataset
        Dataset.
    n_days_old: int
        Number of days in the old calendar.
    n_days_new: int
        Number of days in the new calendar.
    --------------------------------------------------------------------------------------------------------------------
    """

    x["backup"] = x.time

    # Put 360 on 365 calendar.
    ts      = x.assign_coords(time=x.time.dt.dayofyear / n_days_old * n_days_new)
    ts_year = (ts.backup.dt.year.values - ts.backup.dt.year.values[0]) * n_days_new
    ts_time = ts.time.values
    ts[cfg.dim_time] = ts_year + ts_time

    nb_year  = (ts.backup.dt.year.values[-1] - ts.backup.dt.year.values[0])+1
    time_new = np.arange(1, (nb_year*n_days_new)+1)

    # Create new times series.
    year_0  = str(ts.backup.dt.year.values[0])
    month_0 = str(ts.backup.dt.month.values[0]).zfill(2)
    day_0   = str(min(ts.backup.dt.day.values[0], 31)).zfill(2)
    year_1  = str(ts.backup.dt.year[-1].values)
    month_1 = str(ts.backup.dt.month[-1].values).zfill(2)
    if int(month_1) == 11:
        day_1 = str(min(ts.backup.dt.day[-1].values, 31)).zfill(2)
    else:
        day_1 = str(min(ts.backup.dt.day[-1].values + 1, 31)).zfill(2)
    date_0 = year_0 + "-" + month_0 + "-" + day_0
    date_1 = year_1 + "-" + month_1 + "-" + day_1
    time_date = pd.date_range(start=date_0, end=date_1)

    # Remove February 29th.
    time_date = time_date[~((time_date.month == 2) & (time_date.day == 29))]

    # Interpolate 365 days time series.
    ref_365 = ts.interp(time=time_new, kwargs={"fill_value": "extrapolate"}, method="nearest")

    # Recreate 365 time series.
    ref_365[cfg.dim_time] = time_date

    # DEBUG: Plot data.
    # DEBUG: plt.plot(np.arange(1,n_days_new+1),ref_365[:n_days_new].values)
    # DEBUG: plt.plot((np.arange(1, n_days_old+1)/n_days_old*n_days_new), ts[0:n_days_old].values,alpha=0.5)
    # DEBUG: plt.show()

    return ref_365


def reset_calendar(ds: Union[xr.Dataset, xr.DataArray], year_1=-1, year_n=-1, freq=cfg.freq_D) -> pd.DatetimeIndex:

    """
    --------------------------------------------------------------------------------------------------------------------
    Fix calendar using a start year, period and frequency.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.DataArray]
        Dataset.
    year_1 : int
        First year.
    year_n : int
        Last year.
    freq : str
        Frequency: cfg.freq_D=daily; cfg.freq_YS=annual
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract year.
    def extract_date(val) -> [int, int, int]:

        try:
            year  = val.year
            month = val.month
            day   = val.day
        except:
            year  = int(str(val)[0:4])
            month = int(str(val)[5:7])
            day   = int(str(val)[8:10])

        return [year, month, day]

    # First year.
    val_1 = ds.time.values[0]
    if year_1 == -1:
        year_1 = extract_date(val_1)[0]

    # Last year.
    val_n = ds.time.values[len(ds.time.values) - 1]
    if year_n == -1:
        year_n = extract_date(val_n)[0]

    # Exactly the right number of time items.
    n_time = len(ds.time.values)
    if (freq != cfg.freq_D) or (n_time == ((year_n - year_1 + 1) * 365)):
        mult = 365 if freq == cfg.freq_D else 1
        new_time = pd.date_range(str(year_1) + "-01-01", periods=(year_n - year_1 + 1) * mult, freq=freq)
    else:
        arr_time = []
        for val in ds.time.values:
            ymd = extract_date(val)
            arr_time.append(str(ymd[0]) + "-" + str(ymd[1]).rjust(2, "0") + "-" + str(ymd[2]).rjust(2, "0"))
        new_time = pd.DatetimeIndex(arr_time, dtype="datetime64[ns]")

    return new_time


def reset_calendar_list(years: [int]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Fix calendar using a list of years.
    This is only working with a list of years at the moment.

    Parameters
    ----------
    years : [int]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    arr_str = []
    for year in years:
        arr_str.append(str(year) + "-01-01")
    new_time = pd.DatetimeIndex(arr_str)

    return new_time


def convert_to_365_calender(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert calendar to a 365-day calendar.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.DataArray]
        Dataset or data array
    --------------------------------------------------------------------------------------------------------------------
    """

    if isinstance(ds.time.values[0], np.datetime64):
        ds_365 = ds
    else:
        cf = ds.time.values[0].calendar
        if cf in [cfg.cal_noleap, cfg.cal_365day]:
            ds_365 = ds
        elif cf in [cfg.cal_360day]:
            ds_365 = calendar(ds)
        else:
            ds_365 = None
            log("Calendar type not recognized", True)

    # DEBUG: Plot 365 versus 360 calendar.
    # DEBUG: if cfg.opt_plt_365vs360:
    # DEBUG:     plot.plot_360_vs_365(ds, ds_365, var)

    return ds_365


def list_files(p: str) -> [str]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Lists files in a directory.

    Parameters
    ----------
    p : str
        Path of directory.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List.
    p_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for p in f:
            if ".nc" in p:
                p_list.append(os.path.join(r, p))

    # Sort.
    p_list.sort()

    return p_list


def create_multi_dict(n: int, data_type: type) -> dict:

    """
    --------------------------------------------------------------------------------------------------------------------
    Create directory.

    Parameters
    ----------
    n : int
        Number of dimensions.
    data_type : type
        Data type.
    --------------------------------------------------------------------------------------------------------------------
    """

    if n == 1:
        return defaultdict(data_type)
    else:
        return defaultdict(lambda: create_multi_dict(n - 1, data_type))


def calc_error(values_obs: [float], values_pred: [float]) -> float:

    """
    -------------------------------------------------------------------------------------------------------------------
    Calculate the error between observed and predicted values.
    The methods and equations are presented in the following thesis:
    Parishkura D (2009) Évaluation de méthodes de mise à l'échelle statistique: reconstruction des extrêmes et de la
    variabilité du régime de mousson au Sahel (mémoire de maîtrise). UQAM.

    Parameters
    ----------
    values_obs : [float]
        Observed values.
    values_pred : [float]
        Predicted values.
    -------------------------------------------------------------------------------------------------------------------
    """

    error = -1

    # Method #1: Coefficient of determination.
    if cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_r2:
        error = r2_score(values_obs, values_pred)

    # Method #2: Mean absolute error.
    elif cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_mae:
        error = mean_absolute_error(values_obs, values_pred)

    # Method #3: Root mean square error.
    elif cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_rmse:
        error = sqrt(mean_squared_error(values_obs, values_pred))

    # Method #4: Relative root mean square error.
    elif cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_rrmse:
        error = np.sqrt(np.sum(np.square((values_obs - values_pred) / np.std(values_obs))) / len(values_obs))

    return error


def get_datetime_str() -> str:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get date and time.
    --------------------------------------------------------------------------------------------------------------------
    """

    dt = datetime.datetime.now()
    dt_str = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + "_" + \
        str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + str(dt.second).zfill(2)

    return dt_str


def log(msg: str, indent=False):

    """
    --------------------------------------------------------------------------------------------------------------------
    Log message (to console and into a file.

    Parameters
    ----------
    msg : str
        Message.
    indent : bool
        If True, indent text.
    --------------------------------------------------------------------------------------------------------------------
    """

    ln = ""

    # Start line with a timestamp, unless this is a divide.
    if (msg != "-") and (msg != "="):
        ln = get_datetime_str()

    if indent:
        ln += " " * cfg.log_n_blank
    if (msg == "-") or (msg == "="):
        if indent:
            ln += msg * cfg.log_sep_len
        else:
            ln += msg * (cfg.log_sep_len + cfg.log_n_blank)
    else:
        ln += " " + msg

    # Print to console.
    pid_current = os.getpid()
    if pid_current == cfg.pid:
        print(ln)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(cfg.p_log)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Print to file.
    p_log = cfg.p_log
    if pid_current != cfg.pid:
        p_log = p_log.replace(".log", "_" + str(pid_current) + ".log")
    if p_log != "":
        f = open(p_log, "a")
        f.writelines(ln + "\n")
        f.close()


def open_netcdf(p: Union[str, List[str]], drop_variables: [str] = None, chunks: dict = None, combine: str = None,
                concat_dim: str = None, desc: str = "") -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Open a NetCDF file.

    Parameters
    ----------
    p : Union[str, [str]]
        Path of file to be created.
    drop_variables : [str]
        Drop-variables parameter.
    chunks : dict
        Chunks parameter
    combine : str
        Combine parameter.
    concat_dim : str
        Concatenate dimension.
    desc : str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = (os.path.basename(p) if isinstance(p, str) else os.path.basename(p[0]))

    if cfg.opt_trace:
        log("Opening NetCDF file: " + desc, True)

    if isinstance(p, str):
        ds = xr.open_dataset(p, drop_variables=drop_variables, chunks=chunks).copy()
        close_netcdf(ds)
    else:
        ds = xr.open_mfdataset(p, drop_variables=drop_variables, chunks=chunks, combine=combine, concat_dim=concat_dim)

    if cfg.opt_trace:
        log("Opened NetCDF file", True)

    return ds


def save_netcdf(ds: Union[xr.Dataset, xr.DataArray], p, desc=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a dataset or a data array to a NetCDF file.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.DataArray]
        Dataset.
    p : str
        Path of file to be created.
    desc : str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cfg.opt_trace:
        log("Saving NetCDF file: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save NetCDF file.
    ds.to_netcdf(p, "w")

    if cfg.opt_trace:
        log("Saved NetCDF file", True)


def close_netcdf(ds: Union[xr.Dataset, xr.DataArray]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Close a NetCDF file.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.DataArray]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    if ds is not None:
        ds.close()


def save_plot(plt: matplotlib.pyplot, p: str, desc=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a plot to a .png file.

    Parameters
    ----------
    plt : matplotlib.pyplot
        Plot.
    p : str
        Path of file to be created.
    desc : str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cfg.opt_trace:
        log("Saving plot: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Create PNG file.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt.savefig(p)

    if cfg.opt_trace:
        log("Saving plot", True)


def save_csv(df: pd.DataFrame, p: str, desc=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a dataset or a data array to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    p : str
        Path of file to be created.
    desc : str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cfg.opt_trace:
        log("Saving CSV file: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save CSV file.
    df.to_csv(p)

    if cfg.opt_trace:
        log("Saved CSV file", True)


def squeeze_lon_lat(ds: Union[xr.Dataset, xr.Dataset], var: str = "") -> Union[xr.Dataset, xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Squeeze a 3D Dataset to remove longitude and latitude. This will calculate the mean value (all coordinates) for
    each time step.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.Dataset]
        Dataset or DataArray.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_squeeze = ds

    # Squeeze data.
    if cfg.dim_lon in ds.dims:
        ds_squeeze = ds.mean([cfg.dim_lon, cfg.dim_lat])
    elif cfg.dim_rlon in ds.dims:
        ds_squeeze = ds.mean([cfg.dim_rlon, cfg.dim_rlat])
    elif cfg.dim_longitude in ds.dims:
        ds_squeeze = ds.mean([cfg.dim_longitude, cfg.dim_latitude])

    # Transfer units.
    ds_squeeze = copy_attributes(ds, ds_squeeze, var)

    return ds_squeeze


def subset_ctrl_pt(ds: Union[xr.Dataset, xr.Dataset]) -> Union[xr.Dataset, xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Select the center cell.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.Dataset]
        Dataset or DataArray.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_ctr = None

    # Determine control point.
    lon = cfg.lon_bnds.mean()
    lat = cfg.lat_bnds.mean()
    if cfg.ctrl_pt is not None:
        lon = cfg.ctrl_pt[0]
        lat = cfg.ctrl_pt[1]
    else:
        if cfg.dim_rlon in ds.dims:
            if (len(ds.rlat) > 1) or (len(ds.rlon) > 1):
                lon = round(len(ds.rlon) / 2.0)
                lat = round(len(ds.rlat) / 2.0)
        elif cfg.dim_lon in ds.dims:
            if (len(ds.lat) > 1) or (len(ds.lon) > 1):
                lon = round(len(ds.lon) / 2.0)
                lat = round(len(ds.lat) / 2.0)
        else:
            if (len(ds.latitude) > 1) or (len(ds.longitude) > 1):
                lon = round(len(ds.longitude) / 2.0)
                lat = round(len(ds.latitude) / 2.0)

    # Perform subset.
    if cfg.dim_rlon in ds.dims:
        if (len(ds.rlat) > 1) or (len(ds.rlon) > 1):
            ds_ctr = ds.isel(rlon=lon, rlat=lat, drop=True)
    elif cfg.dim_lon in ds.dims:
        if (len(ds.lat) > 1) or (len(ds.lon) > 1):
            ds_ctr = ds.isel(lon=lon, lat=lat, drop=True)
    else:
        if (len(ds.latitude) > 1) or (len(ds.longitude) > 1):
            ds_ctr = ds.isel(longitude=lon, latitude=lat, drop=True)

    return ds_ctr


def subset_lon_lat(ds: xr.Dataset, lon_bnds=None, lat_bnds=None) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Subset a dataset using a box described by a range of longitudes and latitudes.
    That's probably not the best way to do it, but using 'xarray.sel' and 'xarray.where' did not work.
    The rank of cells is increasing for longitude and is decreasing for latitude.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset.
    lon_bnds : [float], optional
        Longitude boundaries.
    lat_bnds : [float], optional
        Latitude boundaries.
    --------------------------------------------------------------------------------------------------------------------
    """

    if lon_bnds is None:
        lon_bnds = cfg.lon_bnds
    if lat_bnds is None:
        lat_bnds = cfg.lat_bnds

    # Latitude.
    if cfg.dim_latitude in ds.dims:
        n_lat = len(ds.latitude)
        lat_min = ds.latitude.min()
        lat_max = ds.latitude.max()
    else:
        n_lat = len(ds.rlat)
        lat_min = ds.rlat.min()
        lat_max = ds.rlat.max()

    # Longitude.
    if cfg.dim_longitude in ds.dims:
        n_lon = len(ds.longitude)
        lon_min = ds.longitude.min()
        lon_max = ds.longitude.max()
    else:
        n_lon = len(ds.rlon)
        lon_min = ds.rlon.min()
        lon_max = ds.rlon.max()

    # Calculate latitude.
    lat_range = lat_max - lat_min
    i_lat_min = math.floor((lat_max - lat_bnds[1]) / lat_range * n_lat)
    i_lat_max = math.ceil((lat_max - lat_bnds[0]) / lat_range * n_lat)
    if lat_bnds[0] == lat_bnds[1]:
        i_lat_max = i_lat_min

    # Calculate longitude.
    lon_range = lon_max - lon_min
    i_lon_min = math.floor((lon_bnds[0] - lon_min) / lon_range * n_lon)
    i_lon_max = math.ceil((lon_bnds[1] - lon_min) / lon_range * n_lon)
    if lon_bnds[0] == lon_bnds[1]:
        i_lon_max = i_lon_min

    # Slice.
    if cfg.dim_latitude in ds.dims:
        ds = ds.isel(latitude=slice(i_lat_min, i_lat_max), longitude=slice(i_lon_min, i_lon_max))
    else:
        ds = ds.isel(rlat=slice(i_lat_min, i_lat_max), rlon=slice(i_lon_min, i_lon_max))

    return ds


def remove_feb29(ds: xr.Dataset) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Remove February 29th from a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

    return ds


def sel_period(ds: xr.Dataset, per: [float]) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Select a period.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    per: [float]
        Selected period, ex: [1981, 2010]
    --------------------------------------------------------------------------------------------------------------------
    """

    ds = ds.where((ds.time.dt.year >= per[0]) & (ds.time.dt.year <= per[1]), drop=True)

    return ds


def get_coordinates(ds: Union[xr.Dataset, xr.DataArray]) -> Tuple[List[float], List[float]]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract coordinates from a dataset.

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to copy coordinates from.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract longitude and latitude.
    if cfg.dim_longitude in list(ds.dims):
        nd_lon_vals = ds[cfg.dim_longitude].values
        nd_lat_vals = ds[cfg.dim_latitude].values
    elif cfg.dim_lon in list(ds.dims):
        nd_lon_vals = ds[cfg.dim_lon].values
        nd_lat_vals = ds[cfg.dim_lat].values
    else:
        nd_lon_vals = ds[cfg.dim_rlon].values
        nd_lat_vals = ds[cfg.dim_rlat].values

    # Convert to a float array.
    lon_vals = []
    for i in range(len(nd_lon_vals)):
        lon_vals.append(float(nd_lon_vals[i]))
    lat_vals = []
    for i in range(len(nd_lat_vals)):
        lat_vals.append(float(nd_lat_vals[i]))

    return lon_vals, lat_vals


def copy_coordinates(ds_from: Union[xr.Dataset, xr.DataArray], ds_to: Union[xr.Dataset, xr.DataArray]) ->\
        Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Copy coordinates from one dataset to another one.

    Parameters
    ----------
    ds_from: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to copy coordinates from.
    ds_to: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to copy coordinates to.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract longitude and latitude values.
    lon_vals, lat_vals = get_coordinates(ds_from)

    # Assign coordinates.
    if cfg.dim_longitude in list(ds_to.dims):
        ds_to[cfg.dim_longitude] = lon_vals
        ds_to[cfg.dim_latitude] = lat_vals
    elif cfg.dim_rlon in list(ds_to.dims):
        ds_to[cfg.dim_rlon] = lon_vals
        ds_to[cfg.dim_rlat] = lat_vals
    else:
        ds_to[cfg.dim_lon] = lon_vals
        ds_to[cfg.dim_lat] = lat_vals

    return ds_to


def copy_attributes(ds_from: Union[xr.Dataset, xr.Dataset], ds_to: Union[xr.Dataset, xr.Dataset], var: str = "") -> \
        Union[xr.Dataset, xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Copy attributes.
    TODO: Other attributes should be transferred as well.

    Parameters
    ----------
    ds_from: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to copy coordinates from.
    ds_to: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to copy coordinates to.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    if cfg.attrs_units in ds_from.attrs:
        ds_to.attrs[cfg.attrs_units] = ds_from.attrs[cfg.attrs_units]
    if isinstance(ds_from, xr.Dataset) and (var in ds_from.data_vars):
        if cfg.attrs_units in ds_from[var].attrs:
            ds_to[var].attrs[cfg.attrs_units] = ds_from[var].attrs[cfg.attrs_units]

    return ds_to


def subset_shape(ds: xr.Dataset, var: str = "") -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Subset based on a shape.

    Parameters
    ----------
    ds_from: xr.Dataset
        Dataset.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    if cfg.d_bounds != "":
        try:
            reset_rlon_rlat = False
            if (cfg.dim_lon not in list(ds.dims)) and (cfg.dim_rlon in list(ds.dims)):
                ds = ds.rename({cfg.dim_rlon: cfg.dim_lon, cfg.dim_rlat: cfg.dim_lat})
                reset_rlon_rlat = True
            if var != "":
                if cfg.attrs_gmap not in ds[var].attrs:
                    ds[var].attrs[cfg.attrs_gmap] = "regular_lon_lat"
            ds = subset.subset_shape(ds, cfg.d_bounds)
            if reset_rlon_rlat:
                ds = ds.rename({cfg.dim_lon: cfg.dim_rlon, cfg.dim_lat: cfg.dim_rlat})
        except TypeError:
            log("Unable to use a mask.", True)

    return ds
