# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020-2022 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import clisops.core.subset as subset
import datetime
import logging
import numpy as np
import pandas as pd
import re
import time
import xarray as xr
import warnings
from cmath import rect, phase
from collections import defaultdict
from math import radians, degrees, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Tuple, Union

# Workflow libraries.
import file_utils as fu
from def_constant import const as c
from def_context import cntx


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
            sfcwind = sfcwind.resample(time=resample).mean(dim=c.dim_time, keep_attrs=True)

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
    ts[c.dim_time] = ts_year + ts_time

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
    ref_365[c.dim_time] = time_date

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
    except Exception as e:
        fu.log(str(e))
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
        except AttributeError as e:
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
    freq=c.freq_D
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
        Frequency: const.freq_D=daily; const.freq_YS=annual
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
    if (freq != c.freq_D) or (n_time == ((year_n - year_1 + 1) * 365)):
        mult = 365 if freq == c.freq_D else 1
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
        if cf in [c.cal_noleap, c.cal_365day]:
            ds_365 = ds
        elif cf in [c.cal_360day]:
            ds_365 = calendar(ds)
        else:
            ds_365 = None

    return ds_365


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
    #       The algorithm is extremely inefficient. There must be a better way to do it.
    if len(da_obs[c.dim_time]) != len(da_pred[c.dim_time]):

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
        if cntx.opt_calib_bias_meth == c.opt_calib_bias_meth_r2:
            error = r2_score(values_obs, values_pred)

        # Method #2: Mean absolute error.
        elif cntx.opt_calib_bias_meth == c.opt_calib_bias_meth_mae:
            error = mean_absolute_error(values_obs, values_pred)

        # Method #3: Root mean square error.
        elif cntx.opt_calib_bias_meth == c.opt_calib_bias_meth_rmse:
            error = sqrt(mean_squared_error(values_obs, values_pred))

        # Method #4: Relative root mean square error.
        elif cntx.opt_calib_bias_meth == c.opt_calib_bias_meth_rrmse:
            if np.std(values_obs) != 0:
                error = np.sqrt(np.sum(np.square((values_obs - values_pred) / np.std(values_obs))) / len(values_obs))
            else:
                error = 0

    return error


def datetime_str() -> str:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get date and time.
    --------------------------------------------------------------------------------------------------------------------
    """

    dt = datetime.datetime.now()
    dt_str = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + "_" + \
        str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + str(dt.second).zfill(2)

    return dt_str


def current_time():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get current time (in seconds).
    --------------------------------------------------------------------------------------------------------------------
    """

    return time.time()


def squeeze_lon_lat(
    ds: Union[xr.Dataset, xr.DataArray],
    varidx_name: str = ""
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Squeeze a 3D Dataset to remove longitude and latitude. This will calculate the mean value (all coordinates) for
    each time step.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray.
    varidx_name : str
        Variable or index.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = ds.copy(deep=True)

    # Squeeze data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if c.dim_lon in ds.dims:
            ds_res = ds.mean([c.dim_lon, c.dim_lat])
        elif c.dim_rlon in ds.dims:
            ds_res = ds.mean([c.dim_rlon, c.dim_rlat])
        elif c.dim_longitude in ds.dims:
            ds_res = ds.mean([c.dim_longitude, c.dim_latitude])

    # Transfer units.
    ds_res = copy_attributes(ds, ds_res, varidx_name)

    return ds_res


def subset_ctrl_pt(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Select the control point. If none was specified by the user, the cell at the center of the domain is considered.

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = None

    # Determine control point.
    lon = np.mean(cntx.lon_bnds)
    lat = np.mean(cntx.lat_bnds)
    if cntx.ctrl_pt is not None:
        lon = cntx.ctrl_pt[0]
        lat = cntx.ctrl_pt[1]
    else:
        if c.dim_rlon in ds.dims:
            if (len(ds.rlat) > 1) or (len(ds.rlon) > 1):
                lon = round(len(ds.rlon) / 2.0)
                lat = round(len(ds.rlat) / 2.0)
        elif c.dim_lon in ds.dims:
            if (len(ds.lat) > 1) or (len(ds.lon) > 1):
                lon = round(len(ds.lon) / 2.0)
                lat = round(len(ds.lat) / 2.0)
        else:
            if (len(ds.latitude) > 1) or (len(ds.longitude) > 1):
                lon = round(len(ds.longitude) / 2.0)
                lat = round(len(ds.latitude) / 2.0)

    # Perform subset.
    if c.dim_rlon in ds.dims:
        if (len(ds.rlat) > 1) or (len(ds.rlon) > 1):
            ds_res = ds.isel(rlon=lon, rlat=lat, drop=True)
    elif c.dim_lon in ds.dims:
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
        if c.dim_longitude in ds_res.dims:
            lon_min = max(ds_res.longitude.min(), min(lon))
            lon_max = min(ds_res.longitude.max(), max(lon))
            ds_res = ds_res.sel(longitude=slice(lon_min, lon_max))
        else:
            lon_min = max(ds_res.rlon.min(), min(lon))
            lon_max = min(ds_res.rlon.max(), max(lon))
            ds_res = ds_res.sel(rlon=slice(lon_min, lon_max))

    # Latitude.
    if len(lat) > 0:
        if c.dim_latitude in ds_res.dims:
            lat_min = max(ds_res.latitude.min(), min(lat))
            lat_max = min(ds_res.latitude.max(), max(lat))
            ds_res = ds_res.sel(latitude=slice(lat_min, lat_max))
        else:
            lat_min = max(ds_res.rlat.min(), min(lat))
            lat_max = min(ds_res.rlat.max(), max(lat))
            ds_res = ds_res.sel(rlat=slice(lat_min, lat_max))

    # Time.
    if (len(time) > 0) and (c.dim_time in ds_res.dims):
        time_l = list(np.unique(ds_res.time.dt.year))
        time_min = max(min(time_l), min(time))
        time_max = min(max(time_l), max(time))
        ds_res = ds_res.sel(time=slice(str(time_min), str(time_max)))

    # Adjust grid.
    if (len(lon) > 0) or (len(lat) > 0) or (len(time) > 0):
        try:
            grid = ds[vi_name].attrs[c.attrs_gmap]
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


def coords(
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
    if c.dim_longitude in list(ds.dims):
        da_lon = ds[c.dim_longitude]
        da_lat = ds[c.dim_latitude]
    elif c.dim_lon in list(ds.dims):
        da_lon = ds[c.dim_lon]
        da_lat = ds[c.dim_lat]
    else:
        da_lon = ds[c.dim_rlon]
        da_lat = ds[c.dim_rlat]

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


def copy_coords(
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
    lon_vals, lat_vals = coords(ds_from, True)

    # Assign coordinates.
    if c.dim_longitude in list(ds_to.dims):
        ds_to[c.dim_longitude] = lon_vals
        ds_to[c.dim_latitude] = lat_vals
    elif c.dim_rlon in list(ds_to.dims):
        ds_to[c.dim_rlon] = lon_vals
        ds_to[c.dim_rlat] = lat_vals
    else:
        ds_to[c.dim_lon] = lon_vals
        ds_to[c.dim_lat] = lat_vals

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

    if c.attrs_units in ds_from.attrs:
        ds_to.attrs[c.attrs_units] = ds_from.attrs[c.attrs_units]
    if isinstance(ds_from, xr.Dataset) and (var in ds_from.data_vars):
        if c.attrs_units in ds_from[var].attrs:
            ds_to[var].attrs[c.attrs_units] = ds_from[var].attrs[c.attrs_units]

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

    if cntx.p_bounds != "":

        try:
            # Memorize dimension names and attributes.
            dim_lon, dim_lat = coord_names(ds_res)
            if dim_lat != c.dim_lat:
                ds_res = ds_res.rename({dim_lon: c.dim_lon, dim_lat: c.dim_lat})
            if var != "":
                if c.attrs_gmap not in ds_res[var].attrs:
                    ds_res[var].attrs[c.attrs_gmap] = "regular_lon_lat"

            # Subset by shape.
            logger = logging.getLogger()
            level = logger.level
            logger.setLevel(logging.CRITICAL)
            ds_res = subset.subset_shape(ds_res, cntx.p_bounds)
            logger.setLevel(level)

            # Recover initial dimension names.
            if dim_lat != c.dim_lat:
                ds_res = ds_res.rename({c.dim_lon: dim_lon, c.dim_lat: dim_lat})

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
    dims_data = coord_names(da_res)
    dims_mask = coord_names(da_mask)

    # Record units.
    units = None
    if c.attrs_units in da_res.attrs:
        units = da_res.attrs[c.attrs_units]

    # Rename spatial dimensions.
    if dims_data != dims_mask:
        da_res = da_res.rename({list(dims_data)[0]: list(dims_mask)[0], list(dims_data)[1]: list(dims_mask)[1]})

    # Apply mask.
    n_time = len(da[c.dim_time])
    da_res[0:n_time, :, :] = da_res[0:n_time, :, :].values * da_mask.values

    # Restore spatial dimensions.
    if dims_data != dims_mask:
        da_res = da_res.rename({list(dims_mask)[0]: list(dims_data)[0], list(dims_mask)[1]: list(dims_data)[1]})

    # Restore units.
    if units is not None:
        da_res.attrs[c.attrs_units] = units

    # Drop coordinates.
    try:
        da_res = da_res.reset_coords(names=c.dim_time, drop=True)
    except ValueError:
        pass

    return da_res


def coord_names(
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

    if c.dim_lat in ds_or_da.dims:
        coord_dict = [c.dim_lon, c.dim_lat]
    elif c.dim_rlat in ds_or_da.dims:
        coord_dict = [c.dim_rlon, c.dim_rlat]
    else:
        coord_dict = [c.dim_longitude, c.dim_latitude]

    return coord_dict


def rename_dimensions(
    da: xr.DataArray,
    lat_name: str = c.dim_latitude,
    lon_name: str = c.dim_longitude
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Function that renames dimensions.

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
            da = da.rename({"dim_0": c.dim_time})
            da = da.rename({"dim_1": lat_name, "dim_2": lon_name})
        elif (c.dim_lat in list(da.dims)) or (c.dim_lon in list(da.dims)):
            da = da.rename({c.dim_lat: lat_name, c.dim_lon: lon_name})
        elif (c.dim_rlat in list(da.dims)) or (c.dim_rlon in list(da.dims)):
            da = da.rename({c.dim_rlat: lat_name, c.dim_rlon: lon_name})
        elif (lat_name not in list(da.dims)) and (lon_name not in list(da.dims)):
            if lat_name == c.dim_latitude:
                da = da.expand_dims(latitude=1)
            if lon_name == c.dim_longitude:
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
    lon_vals = ds_or_da_res[c.dim_longitude]
    lat_vals = ds_or_da_res[c.dim_latitude]
    lon_monotonic_inc = bool(lon_vals[0] < lon_vals[len(lon_vals) - 1])
    lat_monotonic_inc = bool(lat_vals[0] < lat_vals[len(lat_vals) - 1])

    # Flip values.
    if not lon_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(c.dim_longitude, ascending=True)
    if not lat_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(c.dim_latitude, ascending=True)

    # Interpolate, layer by layer (limit=1).
    for t in range(len(ds_or_da_res[c.dim_time])):
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
        ds_or_da_res = ds_or_da_res.sortby(c.dim_longitude, ascending=False)
    if not lat_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(c.dim_latitude, ascending=False)

    return ds_or_da_res


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


def sort_dims(
    ds_da: Union[xr.Dataset, xr.DataArray],
    vi_name: Optional[str] = "",
    template: Optional[Union[xr.DataArray, xr.Dataset, List[str]]] = None,
    rename: Optional[bool] = False
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Reorder dimensions to fit input data.
    There is not guarantee that the dimensions of a DataArray of indices will be the same at each run.

    Parameters
    ----------
    ds_da: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray whose dimensions need to be reordered and potentially renamed.
    vi_name: Optional[str]
        Climate variable or index name.
    template: Union[xr.DataArray, xr.Dataset, List[str]]
        DataArray serving as template.
    rename: Optional[bool]
        If True, rename dimensions according to template.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]:
        Dataset or DataArray with a potentially altered order of dimensions.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine current dimensions (referring to names in 'ds_da').
    if isinstance(ds_da, xr.Dataset):
        dims_current = list(ds_da[vi_name].dims)
    else:
        dims_current = list(ds_da.dims)

    # Determine template dimensions (referring to names in 'template').
    if isinstance(template, xr.Dataset):
        dims_template = list(template[vi_name].dims)
    elif isinstance(template, xr.DataArray):
        dims_template = list(template.dims)
    elif isinstance(template, List):
        dims_template = list(template)
    else:
        dims_template = [c.dim_time, c.dim_latitude, c.dim_longitude]

    # Determine ordered dimensions (referring to the names in 'ds_da').
    dims_order = []
    for dim in dims_template:
        dim = str(dim).replace(c.dim_rlat, c.dim_lat).replace(c.dim_rlon, c.dim_lon).\
            replace(c.dim_latitude, c.dim_lat).replace(c.dim_longitude, c.dim_lon)
        dim_current = [i for i in dims_current if dim in i][0]
        dims_order.append(dim_current)

    # Sort dimensions.
    if isinstance(ds_da, xr.Dataset):
        if c.dim_location in dims_order:
            ds_da[vi_name] = ds_da[vi_name].transpose(dims_order[0], dims_order[1])
        else:
            ds_da[vi_name] = ds_da[vi_name].transpose(dims_order[0], dims_order[1], dims_order[2])
    else:
        if c.dim_location in dims_order:
            ds_da = ds_da.transpose(dims_order[0], dims_order[1])
        else:
            ds_da = ds_da.transpose(dims_order[0], dims_order[1], dims_order[2])

    # Rename dimensions.
    if rename:
        for dim in [c.dim_lat, c.dim_lon]:
            dim_current = [i for i in dims_current if dim in i][0]
            dim_template = [i for i in dims_template if dim in i][0]
            ds_da = ds_da.rename({dim_current: dim_template})

    return ds_da
