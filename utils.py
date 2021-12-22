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
import logging
import math
import matplotlib.pyplot
import numpy as np
import os
import pandas as pd
import re
import time
import xarray as xr
import warnings
from cmath import rect, phase
from collections import defaultdict
from itertools import compress
from math import radians, degrees, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Union, List, Tuple

import sys
sys.path.append("dashboard")
from dashboard import def_varidx as vi


def natural_sort(
    values: Union[float, int]
):

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


def uas_vas_2_sfc(
    uas: xr.DataArray,
    vas: xr.DataArray
):

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


def sfcwind_2_uas_vas(
    sfcwind: xr.DataArray,
    winddir: np.array,
    resample=None,
    nb_per_day=None
):

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
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


def angle_diff(
    alpha: float,
    beta: float
) -> float:

    """
    --------------------------------------------------------------------------------------------------------------------
     Calculate the difference between two angles.

    Parameters
    ----------
    alpha : float
        Angle #1.
    beta : float
        Angle #2.
    --------------------------------------------------------------------------------------------------------------------
    """

    phi = abs(beta - alpha) % 360
    distance = 360 - phi if (phi > 180) else phi

    return distance


def list_cordex(
    p_ds: str,
    rcps: [str]
):

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

        d_format = p_ds + "*" + cfg.sep + "*" + cfg.sep + "AFR-*{r}".format(r=rcps[r]) + cfg.sep + "*" +\
                   cfg.sep + "atmos" + cfg.sep + "*" + cfg.sep
        d_l = glob.glob(d_format)
        d_l = [i for i in d_l if "day" in i]
        d_l.sort()

        # Remove timestep information.
        for i in range(0, len(d_l)):
            tokens = d_l[i].split(cfg.sep)
            d_l[i] = d_l[i].replace(tokens[len(tokens) - 4], "*")

        # Keep only the unique simulation folders (with a * as the timestep).
        d_l = list(set(d_l))
        d_l_valid = [True] * len(d_l)

        # Keep only the simulations with all the variables we need.
        d_l = list(compress(d_l, d_l_valid))

        list_f[rcps[r] + "_historical"] = [w.replace(rcps[r], "historical") for w in d_l]
        list_f[rcps[r]] = d_l

    return list_f


def info_cordex(
    d_ds: str
):

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
    d_format =\
        d_ds + "*" + cfg.sep + "*" + cfg.sep + "AFR-*" + cfg.sep + "day" + cfg.sep + "atmos" + cfg.sep + "*" + cfg.sep
    n_token = len(d_ds.split(cfg.sep)) - 2

    # Loop through simulations sets.
    for i in glob.glob(d_format):
        tokens_i = i.split(cfg.sep)

        # Extract institute, RGM, CGM and emission scenario.
        inst = tokens_i[n_token + 1]
        rgm  = tokens_i[n_token + 2]
        cgm  = tokens_i[n_token + 3].split("_")[1]
        scen = tokens_i[n_token + 3].split("_")[2]

        # Extract variables and ensure that there is at least one NetCDF file available for each  one.
        vars_i = []
        for j in glob.glob(i + "*" + cfg.sep):
            n_netcdf = len(glob.glob(j + "*" + cfg.f_ext_nc))
            if n_netcdf > 0:
                tokens_j = j.split(cfg.sep)
                var      = tokens_j[len(tokens_j) - 2]
                vars_i.append(var)

        sets.append([inst, rgm, cgm, scen, vars_i])

    return sets


def calendar(
    x: Union[xr.Dataset, xr.DataArray],
    n_days_old=360,
    n_days_new=365
):

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

    nb_year  = (ts.backup.dt.year.values[-1] - ts.backup.dt.year.values[0]) + 1
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


def extract_date(
    val: pd.DatetimeIndex
) -> [int]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract date (year, month, day).

    Parameters
    ----------
    val : pd_DatetimeIndex
        Date.
    --------------------------------------------------------------------------------------------------------------------
    """

    try:
        year = val.year
        month = val.month
        day = val.day
    except:
        year = int(str(val)[0:4])
        month = int(str(val)[5:7])
        day = int(str(val)[8:10])

    return [year, month, day]


def extract_date_field(
    ds: Union[xr.DataArray, xr.Dataset],
    field: str = None
) -> [int]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract year, month, day and doy (dayofyear) for each time step.

    Parameters
    ----------
    ds : Union[xr.DataArray, xr.Dataset]
        DataArray or Dataset
    field : str, optional
        Field = {'year','month','day','doy'}
    --------------------------------------------------------------------------------------------------------------------
    """

    res = []

    # Loop through days.
    time_l = list(ds.time.values)
    for i in range(len(time_l)):

        # Extract fields.
        try:
            year  = time_l[i].year
            month = time_l[i].month
            day   = time_l[i].day
            doy   = time_l[i].dayofyear
        except:
            year  = int(str(time_l[i])[0:4])
            month = int(str(time_l[i])[5:7])
            day   = int(str(time_l[i])[8:10])
            doy = day
            doy += 31 if month > 1 else 0
            doy += 28 if (month > 2) and (year % 4 > 0) else 0
            doy += 29 if (month > 2) and (year % 4 == 0) else 0
            doy += 31 if month > 3 else 0
            doy += 30 if month > 4 else 0
            doy += 31 if month > 5 else 0
            doy += 30 if month > 6 else 0
            doy += 31 if month > 7 else 0
            doy += 31 if month > 8 else 0
            doy += 30 if month > 9 else 0
            doy += 31 if month > 10 else 0
            doy += 30 if month > 11 else 0

        # Add field(s) to list.
        if field == "year":
            res.append(year)
        elif field == "month":
            res.append(month)
        elif field == "day":
            res.append(day)
        elif field == "doy":
            res.append(doy)
        else:
            res.append([year, month, day, doy])

    return res


def reset_calendar(
    ds: Union[xr.Dataset, xr.DataArray],
    year_1=-1,
    year_n=-1,
    freq=cfg.freq_D
) -> pd.DatetimeIndex:

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

    # Extract first year.
    val_1 = ds.time.values[0]
    if year_1 == -1:
        year_1 = extract_date(val_1)[0]

    # Extract last year.
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


def reset_calendar_l(
    years: [int]
):

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


def convert_to_365_calender(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:

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


def list_files(
    p: str
) -> [str]:

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
    p_l = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for p in f:
            if cfg.f_ext_nc in p:
                p_l.append(os.path.join(r, p))

    # Sort.
    p_l.sort()

    return p_l


def create_multi_dict(
    n: int,
    data_type: type
) -> dict:

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


def calc_error(
    da_obs: xr.DataArray,
    da_pred: xr.DataArray
) -> float:

    """
    -------------------------------------------------------------------------------------------------------------------
    Calculate the error between observed and predicted values.
    The methods and equations are presented in the following thesis:
    Parishkura D (2009) Évaluation de méthodes de mise à l'échelle statistique: reconstruction des extrêmes et de la
    variabilité du régime de mousson au Sahel (mémoire de maîtrise). UQAM.

    Parameters
    ----------
    da_obs : xr.DataArray
        DataArray containing observed values.
    da_pred : xr.DataArray
        DataArray containing predicted values.
    -------------------------------------------------------------------------------------------------------------------
    """

    error = -1.0

    # TODO: Ensure that the length of datasets is the same in both datasets.
    #       The algorithm is extremely non-efficient. There must be a better way to do it.
    if len(da_obs[cfg.dim_time]) != len(da_pred[cfg.dim_time]):

        # Extract dates (year, month, day).
        dates_obs  = extract_date_field(da_obs)
        dates_pred = extract_date_field(da_pred)

        # Select indices from 'da_obs' to keep.
        sel_obs = []
        for i in range(len(dates_obs)):
            if dates_obs[i] in dates_pred:
                sel_obs.append(i)

        # Select indices from 'da_pred' to keep.
        sel_pred = []
        for i in range(len(dates_pred)):
            if dates_pred[i] in dates_obs:
                sel_pred.append(i)

        # Subset data arrays.
        da_obs  = da_obs[sel_obs]
        da_pred = da_pred[sel_pred]

    # Extract values.
    values_obs  = da_obs.values.ravel()
    values_pred = da_pred.values.ravel()

    # Remove values that are nan in at least one of the two datasets.
    sel = (np.isnan(values_obs) == False) & (np.isnan(values_pred) == False)
    da_obs = xr.DataArray(values_obs)
    da_obs = da_obs[sel]
    da_pred = xr.DataArray(values_pred)
    da_pred = da_pred[sel]
    values_obs  = da_obs.values
    values_pred = da_pred.values

    if len(values_obs) == len(values_pred):

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
            if np.std(values_obs) != 0:
                error = np.sqrt(np.sum(np.square((values_obs - values_pred) / np.std(values_obs))) / len(values_obs))
            else:
                error = 0

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


def get_current_time():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get current time (in seconds).
    --------------------------------------------------------------------------------------------------------------------
    """

    return time.time()


def log(
    msg: str,
    indent=False
):

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
        p_log = p_log.replace(cfg.f_ext_log, "_" + str(pid_current) + cfg.f_ext_log)
    if p_log != "":
        f = open(p_log, "a")
        f.writelines(ln + "\n")
        f.close()


def open_netcdf(
    p: Union[str, List[str]],
    drop_variables: [str] = None,
    chunks: Union[int, dict] = None,
    combine: str = None,
    concat_dim: str = None,
    desc: str = ""
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Open a NetCDF file.

    Parameters
    ----------
    p : Union[str, [str]]
        Path of file to be created.
    drop_variables : [str]
        Drop-variables parameter.
    chunks : Union[int,dict]
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

        # Open file normally.
        ds = xr.open_dataset(p, drop_variables=drop_variables).load()
        close_netcdf(ds)

        # Determine the number of chunks.
        if cfg.use_chunks and (cfg.n_proc == 1) and (chunks is None) and ("scen" in p) and (cfg.dim_time in ds.dims):
            chunks = {cfg.dim_time: len(ds[cfg.dim_time])}

        # Reopen file using chunks.
        if chunks is not None:
            ds = xr.open_dataset(p, drop_variables=drop_variables, chunks=chunks).copy(deep=True).load()
            close_netcdf(ds)
    else:
        ds = xr.open_mfdataset(p, drop_variables=drop_variables, chunks=chunks, combine=combine, concat_dim=concat_dim)

    if cfg.opt_trace:
        log("Opened NetCDF file", True)

    return ds


def save_netcdf(
    ds: Union[xr.Dataset, xr.DataArray],
    p: str,
    desc: str = ""
):

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
    if not (os.path.exists(d)):
        os.makedirs(d)

    # Create a temporary file to indicate that writing is in progress.
    p_inc = p.replace(cfg.f_ext_nc, ".incomplete")
    if not os.path.exists(p_inc):
        open(p_inc, 'a').close()

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save NetCDF file.
    ds.to_netcdf(p, "w")

    # Discard the temporary file.
    if os.path.exists(p_inc):
        os.remove(p_inc)

    if cfg.opt_trace:
        log("Saved NetCDF file", True)


def clean_netcdf(
    d: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Clean NetCDF files.
    A .nc file will be removed if there's an .incomplete file with the same name. The .incomplete file is removed as
    well. This is done to avoid having potentially incomplete NetCDF files, which can result in a crash.

    Parameters
    ----------
    d : str
        Base directory to search from.
    --------------------------------------------------------------------------------------------------------------------
    """

    log("Cleaning NetCDF file: " + d, True)

    # List temporary files.
    if d[len(d) - 1] != cfg.sep:
        d = d + cfg.sep
    p_inc_l = glob.glob(d + "**" + cfg.sep + "*.incomplete", recursive=True)

    # Loop through temporary files.
    for p_inc in p_inc_l:

        # Attempt removing an associated NetCDF file.
        p_nc = p_inc.replace(".incomplete", cfg.f_ext_nc)
        if os.path.exists(p_nc):
            log("Removing: " + p_nc)
            os.remove(p_nc)

        # Remove the temporary file.
        if os.path.exists(p_inc):
            log("Removing: " + p_inc)
            os.remove(p_inc)


def close_netcdf(
    ds: Union[xr.Dataset, xr.DataArray]
):

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
        try:
            ds.close()
        except:
            pass


def save_plot(
    plt: matplotlib.pyplot,
    p: str,
    desc=""
):

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


def save_csv(
    df: pd.DataFrame,
    p: str,
    desc=""
):

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
    df.to_csv(p, index=False)

    if cfg.opt_trace:
        log("Saved CSV file", True)


def squeeze_lon_lat(
    ds: Union[xr.Dataset, xr.Dataset],
    varidx_name: str = ""
) -> Union[xr.Dataset, xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Squeeze a 3D Dataset to remove longitude and latitude. This will calculate the mean value (all coordinates) for
    each time step.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.Dataset]
        Dataset or DataArray.
    varidx_name : str
        Variable or index.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = ds.copy(deep=True)

    # Squeeze data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if cfg.dim_lon in ds.dims:
            ds_res = ds.mean([cfg.dim_lon, cfg.dim_lat])
        elif cfg.dim_rlon in ds.dims:
            ds_res = ds.mean([cfg.dim_rlon, cfg.dim_rlat])
        elif cfg.dim_longitude in ds.dims:
            ds_res = ds.mean([cfg.dim_longitude, cfg.dim_latitude])

    # Transfer units.
    ds_res = copy_attributes(ds, ds_res, varidx_name)

    return ds_res


def subset_ctrl_pt(
    ds: Union[xr.Dataset, xr.Dataset]
) -> Union[xr.Dataset, xr.Dataset]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Select the control point. If none was specified by the user, the cell at the center of the domain is considered.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.Dataset]
        Dataset or DataArray.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = None

    # Determine control point.
    lon = np.mean(cfg.lon_bnds)
    lat = np.mean(cfg.lat_bnds)
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
            ds_res = ds.isel(rlon=lon, rlat=lat, drop=True)
    elif cfg.dim_lon in ds.dims:
        if (len(ds.lat) > 1) or (len(ds.lon) > 1):
            ds_res = ds.isel(lon=lon, lat=lat, drop=True)
    else:
        if (len(ds.latitude) > 1) or (len(ds.longitude) > 1):
            ds_res = ds.isel(longitude=lon, latitude=lat, drop=True)

    return ds_res


def subset_doy(
    da_or_ds: Union[xr.DataArray, xr.Dataset],
    doy_min: int,
    doy_max: int
) -> Union[xr.DataArray, xr.Dataset]:

    """
    Subset based on day of year.

    Parameters
    ----------
    da_or_ds: Union[xr.DataArray, xr.Dataset]
        DataArray or Dataset.
    doy_min: int
        Minimum day of year to consider.
    doy_max: int
        Maximum day of year to consider.
    --------------------------------------------------------------------------------------------------------------------
    """

    da_or_ds_res = da_or_ds.copy(deep=True)

    if (doy_min > -1) or (doy_max > -1):
        if doy_min == -1:
            doy_min = 1
        if doy_max == -1:
            doy_max = 365
        if doy_max >= doy_min:
            cond = (da_or_ds_res.time.dt.dayofyear >= doy_min) & (da_or_ds_res.time.dt.dayofyear <= doy_max)
        else:
            cond = (da_or_ds_res.time.dt.dayofyear <= doy_max) | (da_or_ds_res.time.dt.dayofyear >= doy_min)
        da_or_ds_res = da_or_ds_res[cond]

    return da_or_ds_res


def subset_lon_lat_time(
    ds: xr.Dataset,
    vi_name: str,
    lon: List[float] = [],
    lat: List[float] = [],
    time: List[int] = []
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Subset a dataset using a box described by a range of longitudes and latitudes.
    That's probably not the best way to do it, but using 'xarray.sel' and 'xarray.where' did not work.
    The rank of cells is increasing for longitude and is decreasing for latitude.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset.
    vi_name : str
        Variable or index name.
    lon : List[float]
        Longitude.
    lat : List[float]
        Latitude.
    time : List[int]
        Time.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = ds.copy(deep=True)

    # Longitude.
    if len(lon) > 0:
        if cfg.dim_longitude in ds_res.dims:
            lon_min = max(ds_res.longitude.min(), min(lon))
            lon_max = min(ds_res.longitude.max(), max(lon))
            ds_res = ds_res.sel(longitude=slice(lon_min, lon_max))
        else:
            lon_min = max(ds_res.rlon.min(), min(lon))
            lon_max = min(ds_res.rlon.max(), max(lon))
            ds_res = ds_res.sel(rlon=slice(lon_min, lon_max))

    # Latitude.
    if len(lat) > 0:
        if cfg.dim_latitude in ds_res.dims:
            lat_min = max(ds_res.latitude.min(), min(lat))
            lat_max = min(ds_res.latitude.max(), max(lat))
            ds_res = ds_res.sel(latitude=slice(lat_min, lat_max))
        else:
            lat_min = max(ds_res.rlat.min(), min(lat))
            lat_max = min(ds_res.rlat.max(), max(lat))
            ds_res = ds_res.sel(rlat=slice(lat_min, lat_max))

    # Time.
    if (len(time) > 0) and (cfg.dim_time in ds_res.dims):
        time_l = list(np.unique(ds_res.time.dt.year))
        time_min = max(min(time_l), min(time))
        time_max = min(max(time_l), max(time))
        ds_res = ds_res.sel(time=slice(str(time_min), str(time_max)))

    # Adjust grid.
    if (len(lon) > 0) or (len(lat) > 0) or (len(time) > 0):
        try:
            grid = ds[vi_name].attrs[cfg.attrs_gmap]
            ds_res[grid] = ds[grid]
        except KeyError:
            pass

    return ds_res


def remove_feb29(
    ds: xr.Dataset
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Remove February 29th from a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = ds.copy(deep=True)

    ds_res = ds_res.sel(time=~((ds_res.time.dt.month == 2) & (ds_res.time.dt.day == 29)))

    return ds_res


def sel_period(
    ds: xr.Dataset,
    per: [float]
) -> xr.Dataset:

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

    ds_res = ds.copy(deep=True)

    ds_res = ds_res.where((ds_res.time.dt.year >= per[0]) & (ds_res.time.dt.year <= per[1]), drop=True)

    return ds_res


def get_coordinates(
    ds: Union[xr.Dataset, xr.DataArray],
    array_format: bool = False
) -> Tuple[Union[List[float], xr.DataArray], Union[List[float], xr.DataArray]]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract coordinates from a dataset.

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to copy coordinates from.
    array_format: bool
        If True, return an array. If False, return xr.DataArray.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract longitude and latitude.
    if cfg.dim_longitude in list(ds.dims):
        da_lon = ds[cfg.dim_longitude]
        da_lat = ds[cfg.dim_latitude]
    elif cfg.dim_lon in list(ds.dims):
        da_lon = ds[cfg.dim_lon]
        da_lat = ds[cfg.dim_lat]
    else:
        da_lon = ds[cfg.dim_rlon]
        da_lat = ds[cfg.dim_rlat]

    # Need to return an array.
    if array_format:

        # Extract values.
        nd_lon_vals = da_lon.values
        nd_lat_vals = da_lat.values

        # Convert to a float array.
        lon_vals = []
        for i in range(len(nd_lon_vals)):
            lon_vals.append(float(nd_lon_vals[i]))
        lat_vals = []
        for i in range(len(nd_lat_vals)):
            lat_vals.append(float(nd_lat_vals[i]))

    else:
        lon_vals = da_lon
        lat_vals = da_lat

    return lon_vals, lat_vals


def copy_coordinates(
    ds_from: Union[xr.Dataset, xr.DataArray],
    ds_to: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:

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
    lon_vals, lat_vals = get_coordinates(ds_from, True)

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


def copy_attributes(
    ds_from: Union[xr.Dataset, xr.Dataset],
    ds_to: Union[xr.Dataset, xr.Dataset],
    var: str = ""
) -> Union[xr.Dataset, xr.Dataset]:

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


def subset_shape(
    ds: xr.Dataset,
    var: str = ""
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Subset based on a shape.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = ds.copy(deep=True)

    if cfg.p_bounds != "":

        try:
            # Memorize dimension names and attributes.
            dim_lon, dim_lat = get_coord_names(ds_res)
            if dim_lat != cfg.dim_lat:
                ds_res = ds_res.rename({dim_lon: cfg.dim_lon, dim_lat: cfg.dim_lat})
            if var != "":
                if cfg.attrs_gmap not in ds_res[var].attrs:
                    ds_res[var].attrs[cfg.attrs_gmap] = "regular_lon_lat"

            # Subset by shape.
            logger = logging.getLogger()
            level = logger.level
            logger.setLevel(logging.CRITICAL)
            ds_res = subset.subset_shape(ds_res, cfg.p_bounds)
            logger.setLevel(level)

            # Recover initial dimension names.
            if dim_lat != cfg.dim_lat:
                ds_res = ds_res.rename({cfg.dim_lon: dim_lon, cfg.dim_lat: dim_lat})

        except (TypeError, ValueError):
            return ds.copy(deep=True)

    return ds_res


def apply_mask(
    da: xr.DataArray,
    da_mask: xr.DataArray
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Apply a mask.

    Parameters
    ----------
    da: xr.DataArray
        Dataset or DataArray on which to apply the mask.
    da_mask: xr.DataArray
        Mask (contains 0, 1 and nan).
    --------------------------------------------------------------------------------------------------------------------
    """

    da_res = da.copy(deep=True)

    # Get coordinates names.
    dims_data = get_coord_names(da_res)
    dims_mask = get_coord_names(da_mask)

    # Record units.
    units = None
    if cfg.attrs_units in da_res.attrs:
        units = da_res.attrs[cfg.attrs_units]

    # Rename spatial dimensions.
    if dims_data != dims_mask:
        da_res = da_res.rename({list(dims_data)[0]: list(dims_mask)[0], list(dims_data)[1]: list(dims_mask)[1]})

    # Apply mask.
    n_time = len(da[cfg.dim_time])
    da_res[0:n_time, :, :] = da_res[0:n_time, :, :].values * da_mask.values

    # Restore spatial dimensions.
    if dims_data != dims_mask:
        da_res = da_res.rename({list(dims_mask)[0]: list(dims_data)[0], list(dims_mask)[1]: list(dims_data)[1]})

    # Restore units.
    if units is not None:
        da_res.attrs[cfg.attrs_units] = units

    # Drop coordinates.
    try:
        da_res = da_res.reset_coords(names=cfg.dim_time, drop=True)
    except:
        pass

    return da_res


def get_coord_names(
    ds_or_da: Union[xr.Dataset, xr.DataArray]
) -> List[str]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get coordinate names (dictionary).

    Parameters
    ----------
    ds_or_da: Union[xr.Dataset, xr.DataArray]
        Dataset.

    Returns
    -------
    coord_dict: [str]
        [<longitude_field>, <latitude_field>]
    --------------------------------------------------------------------------------------------------------------------
    """

    if cfg.dim_lat in ds_or_da.dims:
        coord_dict = [cfg.dim_lon, cfg.dim_lat]
    elif cfg.dim_rlat in ds_or_da.dims:
        coord_dict = [cfg.dim_rlon, cfg.dim_rlat]
    else:
        coord_dict = [cfg.dim_longitude, cfg.dim_latitude]

    return coord_dict


def rename_dimensions(
    da: xr.DataArray,
    lat_name: str = cfg.dim_latitude,
    lon_name: str = cfg.dim_longitude
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    # Function that renames dimensions.

    Parameters
    ----------
    da: xr.DataArray
        DataArray.
    lat_name: str
        Latitude name to transform to.
    lon_name: str
        Longitude name to transform to.
    --------------------------------------------------------------------------------------------------------------------
    """

    if (lat_name not in da.dims) or (lon_name not in da.dims):

        if "dim_0" in list(da.dims):
            da = da.rename({"dim_0": cfg.dim_time})
            da = da.rename({"dim_1": lat_name, "dim_2": lon_name})
        elif (cfg.dim_lat in list(da.dims)) or (cfg.dim_lon in list(da.dims)):
            da = da.rename({cfg.dim_lat: lat_name, cfg.dim_lon: lon_name})
        elif (cfg.dim_rlat in list(da.dims)) or (cfg.dim_rlon in list(da.dims)):
            da = da.rename({cfg.dim_rlat: lat_name, cfg.dim_rlon: lon_name})
        elif (lat_name not in list(da.dims)) and (lon_name not in list(da.dims)):
            if lat_name == cfg.dim_latitude:
                da = da.expand_dims(latitude=1)
            if lon_name == cfg.dim_longitude:
                da = da.expand_dims(longitude=1)

    return da


def interpolate_na_fix(
    ds_or_da: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Interpolate values considering both longitude and latitude.
    This is a wrapper of the pandas DataFrame interpolate function.
    The idea behind flip is that the x-axis (latitude or longitude steps) must be monotonically increasing when
    using the xr.interpolate_na function (not used anymore).

    Parameters
    ----------
    ds_or_da: Union[xr.Dataset, xr.DataArray]
        Dataset or data array.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_or_da_res = ds_or_da.copy(deep=True)

    # Extract coordinates and determine if they are increasing.
    lon_vals = ds_or_da_res[cfg.dim_longitude]
    lat_vals = ds_or_da_res[cfg.dim_latitude]
    lon_monotonic_inc = bool(lon_vals[0] < lon_vals[len(lon_vals) - 1])
    lat_monotonic_inc = bool(lat_vals[0] < lat_vals[len(lat_vals) - 1])

    # Flip values.
    if not lon_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(cfg.dim_longitude, ascending=True)
    if not lat_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(cfg.dim_latitude, ascending=True)

    # Interpolate, layer by layer (limit=1).
    for t in range(len(ds_or_da_res[cfg.dim_time])):
        n_i = max(len(ds_or_da_res[t].longitude), len(ds_or_da_res[t].latitude))
        for i in range(n_i):
            df_t = ds_or_da_res[t].to_pandas()
            ds_or_da_lat_t = ds_or_da_res[t].copy(deep=True)
            ds_or_da_lon_t = ds_or_da_res[t].copy(deep=True)
            ds_or_da_lat_t.values = df_t.interpolate(axis=0, limit=1, limit_direction="both")
            ds_or_da_lon_t.values = df_t.interpolate(axis=1, limit=1, limit_direction="both")
            ds_or_da_res[t] = (ds_or_da_lon_t + ds_or_da_lat_t) / 2.0
            ds_or_da_res[t].values[np.isnan(ds_or_da_res[t].values)] =\
                ds_or_da_lon_t.values[np.isnan(ds_or_da_res[t].values)]
            ds_or_da_res[t].values[np.isnan(ds_or_da_res[t].values)] =\
                ds_or_da_lat_t.values[np.isnan(ds_or_da_res[t].values)]
            n_nan = np.isnan(ds_or_da_res[t].values).astype(float).sum()
            if n_nan == 0:
                break

    # Unflip values.
    if not lon_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(cfg.dim_longitude, ascending=False)
    if not lat_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(cfg.dim_latitude, ascending=False)

    return ds_or_da_res


def create_mask() -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a mask, based on climate scenarios for the temperature or precipitation variable.
    All values with a value are attributed a value of 1. Other values are assigned 'nan'.
    --------------------------------------------------------------------------------------------------------------------
    """

    da_mask = None

    f_l = glob.glob(cfg.d_stn + "*/*" + cfg.f_ext_nc)
    for i in range(len(f_l)):

        # Open NetCDF file.
        ds = open_netcdf(f_l[i])
        var = list(ds.data_vars)[0]
        if var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:

            # Create mask.
            da_mask = ds[var][0] * 0 + 1

            break

    return da_mask


def doy_str_to_doy(
    doy_str: str
) -> int:

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert from DayOfYearStr to DayOfYear.

    Parameters
    ----------
    doy_str: str
        Day of year, as a string ("mm-dd").

    Returns
    -------
    doy: int
        Day of year, as an integer (value between 1 and 366).
    --------------------------------------------------------------------------------------------------------------------
    """

    doy = datetime.datetime.strptime(doy_str, "%m-%d").timetuple().tm_yday

    return doy


def doy_to_doy_str(
    doy: int
) -> str:

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert from DayOfYearStr to DayOfYear.

    Parameters
    ----------
    doy: int
        Day of year, as an integer (value between 1 and 366).

    Returns
    -------
    doy_str: str
        Day of year, as a string ("mm-dd").
    --------------------------------------------------------------------------------------------------------------------
    """

    tt = datetime.datetime.strptime(str.zfill(str(int(doy)), 3), "%j").timetuple()
    doy_str = str(tt.tm_mon).zfill(2) + "-" + str(tt.tm_mday).zfill(2)

    return doy_str


def reorder_dims(da_idx: xr.DataArray, ds_or_da_src: Union[xr.DataArray, xr.Dataset]) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    # Reorder dimensions to fit input data.
    # There is not guarantee that the dimensions of a DataArray of indices will be the same at each run.

    Parameters
    ----------
    da_idx : xr.DataArray
        DataArray whose dimensions need to be reordered.
    ds_or_da_src : Union[xr.DataArray, xr.Dataset]
        DataArray serving as template.

    Returns
    -------
    xr.DataArray :
        DataArray with a potentially altered order of dimensions.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract dimensions.
    if isinstance(ds_or_da_src, xr.Dataset):
        dims = list(ds_or_da_src[list(ds_or_da_src.data_vars.variables.mapping)[0]].dims)
    else:
        dims = list(ds_or_da_src.dims)

    # Rename dimensions.
    for i in range(len(dims)):
        dims[i] = dims[i].replace(cfg.dim_rlat, cfg.dim_latitude).replace(cfg.dim_rlon, cfg.dim_longitude)
        if dims[i] == cfg.dim_lat:
            dims[i] = cfg.dim_latitude
        if dims[i] == cfg.dim_lon:
            dims[i] = cfg.dim_longitude

    # Reorder dimensions.
    if cfg.dim_location in dims:
        da_idx_new = da_idx.transpose(dims[0], dims[1]).copy()
    else:
        da_idx_new = da_idx.transpose(dims[0], dims[1], dims[2]).copy()

    return da_idx_new
