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
import wf_file_utils as fu
from cl_constant import const as c
from cl_context import cntx


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
            sfcwind = sfcwind.resample(time=resample).mean(dim=c.DIM_TIME, keep_attrs=True)

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
    ts[c.DIM_TIME] = ts_year + ts_time

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
    ref_365[c.DIM_TIME] = time_date

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
    freq=c.FREQ_D
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
        Frequency: c.FREQ_D=daily; c.FREQ_YS=annual
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
    if (freq != c.FREQ_D) or (n_time == ((year_n - year_1 + 1) * 365)):
        mult = 365 if freq == c.FREQ_D else 1
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
        if cf in [c.CAL_NOLEAP, c.CAL_365DAY]:
            ds_365 = ds
        elif cf in [c.CAL_360DAY]:
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
    if len(da_obs[c.DIM_TIME]) != len(da_pred[c.DIM_TIME]):

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
    sel = (np.isnan(values_obs).astype(int) == 0) & (np.isnan(values_pred).astype(int) == 0)
    da_obs = xr.DataArray(values_obs)
    da_obs = da_obs[sel]
    da_pred = xr.DataArray(values_pred)
    da_pred = da_pred[sel]
    values_obs  = da_obs.values
    values_pred = da_pred.values

    if len(values_obs) == len(values_pred):

        # Method #1: Coefficient of determination.
        if cntx.opt_bias_err_meth == c.OPT_BIAS_ERR_METH_R2:
            error = r2_score(values_obs, values_pred)

        # Method #2: Mean absolute error.
        elif cntx.opt_bias_err_meth == c.OPT_BIAS_ERR_METH_MAE:
            error = mean_absolute_error(values_obs, values_pred)

        # Method #3: Root mean square error.
        elif cntx.opt_bias_err_meth == c.OPT_BIAS_ERR_METH_RMSE:
            error = sqrt(mean_squared_error(values_obs, values_pred))

        # Method #4: Relative root mean square error.
        elif cntx.opt_bias_err_meth == c.OPT_BIAS_ERR_METH_RRMSE:
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
        if c.DIM_LON in ds.dims:
            ds_res = ds.mean([c.DIM_LON, c.DIM_LAT])
        elif c.DIM_RLON in ds.dims:
            ds_res = ds.mean([c.DIM_RLON, c.DIM_RLAT])
        elif c.DIM_LONGITUDE in ds.dims:
            ds_res = ds.mean([c.DIM_LONGITUDE, c.DIM_LATITUDE])

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
        if c.DIM_RLON in ds.dims:
            if (len(ds.rlat) > 1) or (len(ds.rlon) > 1):
                lon = round(len(ds.rlon) / 2.0)
                lat = round(len(ds.rlat) / 2.0)
        elif c.DIM_LON in ds.dims:
            if (len(ds.lat) > 1) or (len(ds.lon) > 1):
                lon = round(len(ds.lon) / 2.0)
                lat = round(len(ds.lat) / 2.0)
        else:
            if (len(ds.latitude) > 1) or (len(ds.longitude) > 1):
                lon = round(len(ds.longitude) / 2.0)
                lat = round(len(ds.latitude) / 2.0)

    # Perform subset.
    if c.DIM_RLON in ds.dims:
        if (len(ds.rlat) > 1) or (len(ds.rlon) > 1):
            ds_res = ds.isel(rlon=lon, rlat=lat, drop=True)
    elif c.DIM_LON in ds.dims:
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


def grid_properties(
    ds_da: Union[xr.Dataset, xr.DataArray]
) -> Tuple[float, float, int, int]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate grid properties.

    This calculates cell size and cell count along each direction (longitude and latitude).

    Parameters
    ----------
    ds_da: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray.

    Returns
    -------
    Tuple[float, float, int, int]
        Cell size (longitude), cell size (latitude), cell count (longitude), cell count (latitude).
        A value of zero is returned if resolution could not be calculated along a dimension.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Longitude: List values.
    if c.DIM_LONGITUDE in ds_da.dims:
        lon_vals = ds_da.longitude
    elif c.DIM_LON in ds_da.dims:
        lon_vals = ds_da.lon
    else:
        lon_vals = ds_da.rlon

    # Longitude: Extract lower an upper values, along with the number of cells between center cells.
    lon_min, lon_max = 0, 0
    lon_n = len(lon_vals)
    if lon_n > 0:
        lon_min = float(lon_vals[0])
        lon_max = float(lon_vals[lon_n - 1])

    # Latitude: List values.
    if c.DIM_LATITUDE in ds_da.dims:
        lat_vals = ds_da.latitude
    elif c.DIM_LAT in ds_da.dims:
        lat_vals = ds_da.lat
    else:
        lat_vals = ds_da.rlat

    # Latitude: Extract lower an upper values, along with the number of cells between center cells.
    lat_min, lat_max = 0, 0
    lat_n = len(lat_vals)
    if lat_n > 0:
        lat_min = float(lat_vals[0])
        lat_max = float(lat_vals[lat_n - 1])

    # Calculate cell size.
    lon_cs = abs((lon_max - lon_min) / (lon_n - 1) if lon_n > 0 else 0)
    lat_cs = abs((lat_max - lat_min) / (lat_n - 1) if lat_n > 0 else 0)

    return lon_cs, lat_cs, lon_n, lat_n


def subset_lon_lat_time(
    ds_da: Union[xr.Dataset, xr.DataArray],
    vi_name: str,
    lon: List[float] = [],
    lat: List[float] = [],
    t: List[int] = [],
    force_2x2_grid: Optional[bool] = True
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Perform a spatiotemporal subset using a box described by a range of longitudes and latitudes.

    Parameters
    ----------
    ds_da: Union[xr.Dataset, xr.DataArray]
        Dataset.
    vi_name: str
        Variable or index name.
    lon: List[float]
        Longitude.
    lat: List[float]
        Latitude.
    t: List[int]
        Time.
    force_2x2_grid: Optional[bool]
         If True, the algorithm attempts to produce a grid that is at least 2x2. This is done by enlarging the search
         box in each direction by 1/2 x cell size, then by 1.0 x cell size (if required).
         Between 4 and 9 cells should be selected, given that the dataset comprises these cells.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        Spatiotemporal subset of Dataset or DataArray.

    Notes
    -----
    The 'force_2x2_grid' option enlarges the search box by 0.5 to 1.0 x cell size in each direction.
    This option will have an effect only if a single cell is selected along an axis (longitude or latitude) prior
    to the extension of the search box.

    Initial search box (only cell 5)     Enlarged once (cells 2, 3, 5, 6)     Enlarged twice (all cells)
    +---------+---------+---------+      +---------+---------+---------+      +---------+---------+---------+
    |         |         |         |      |         |         |         |      |    #======================# |
    |         |         |         |      |         |         |         |      |    #    |         |       # |
    |         |         |         |      |         |#==========#       |      |    #    |         |       # |
    | 1       | 2       | 3       |      | 1       |# 2      | #     3 |      | 1  #    | 2       | 3     # |
    +---------+---------+---------+      +---------+#--------+-#-------+      +----#----+---------+-------#-+
    |         |    #==# |         |      |         |#      5 | #       |      |    #    |         |       # |
    |         |    #==# |         | ===> |         |#        | #       | ===> |    #    |         |       # |
    |         |         |         |      |         |#        | #       |      |    #    |         |       # |
    | 4       | 5       | 6       |      | 4       |#==========#     6 |      | 4  #    | 5       | 6     # |
    +---------+---------+---------+      +---------+---------+---------+      +----#----+---------+-------#-+
    |         |         |         |      |         |         |         |      |    #    |         |       # |
    |         |         |         |      |         |         |         |      |    #======================# |
    |         |         |         |      |         |         |         |      |         |         |         |
    | 7       | 8       | 9       |      | 7       | 8       | 9       |      | 7       | 8       | 9       |
    +---------+---------+---------+      +---------+---------+---------+      +---------+---------+---------+

    Examples
    --------
    Assuming that 'ds_da' has a 3x3 grid.

    +---+---+---+    Initial search box ===> Enlarged search box
    | 1 | 2 | 3 |    (1,4)              ===> (1,4) and (2,5)
    +---+---+---+    (2,5)              ===> (2,5) and ((1,4) or (3,6))
    | 4 | 5 | 6 |    (5)                ===> (5) and ((1,2,3,4,6,7,8,9) or (1,2,4) or (4,7,8) or (2,3,6) or (6,8,9))
    +---+---+---+    (6)                ===> (6) and ((2,3,5,8,9) or (2,3,5) or (5,8,9))
    | 7 | 8 | 9 |    (3)                ===> (3) and (2,5,6)
    +---+---+---+
    The 'or' in the returned cells depends on how centered the selection box is with respect to the grid.
    --------------------------------------------------------------------------------------------------------------------
    """

    lon_buffer, lat_buffer = 0.0, 0.0

    # Calculate resolution.
    lon_cs, lat_cs, lon_n, lat_n = grid_properties(ds_da)

    # Loop twice: without a buffer (first attempt) and with a bufffer (if a single cell was selected).
    ds_da_sub = None
    n_pass = 3
    for i_pass in range(n_pass):

        # Create a copy of the dataset.
        ds_da_sub = ds_da.copy(deep=True)

        # Longitude.
        if len(lon) > 0:

            # List values.
            if c.DIM_LONGITUDE in ds_da_sub.dims:
                lon_vals = ds_da_sub.longitude
            elif c.DIM_LON in ds_da_sub.dims:
                lon_vals = ds_da_sub.lon
            else:
                lon_vals = ds_da_sub.rlon

            # Determine minimum an maximum values.
            lon_min_data = float(lon_vals[0])
            lon_max_data = float(lon_vals[len(lon_vals) - 1])
            lon_min = max(float(lon_vals.min()), min(lon) - lon_buffer)
            lon_max = min(float(lon_vals.max()), max(lon) + lon_buffer)

            # Invert coordinates, then add minimum and maximum values in array.
            invert_coords = (lon_min_data > lon_max_data) and (lon_min < lon_max)
            lon_l = [lon_min, lon_max] if not invert_coords else [lon_max, lon_min]

            # Select cells.
            if c.DIM_LONGITUDE in ds_da_sub.dims:
                ds_da_sub = ds_da_sub.sel(longitude=slice(*lon_l))
                lon_n = len(ds_da_sub.longitude)
            elif c.DIM_LON in ds_da_sub.dims:
                ds_da_sub = ds_da_sub.sel(lon=slice(*lon_l))
                lon_n = len(ds_da_sub.lon)
            else:
                ds_da_sub = ds_da_sub.sel(rlon=slice(*lon_l))
                lon_n = len(ds_da_sub.rlon)

        # Latitude.
        if len(lat) > 0:

            # List values.
            if c.DIM_LATITUDE in ds_da_sub.dims:
                lat_vals = ds_da_sub.latitude
            elif c.DIM_LAT in ds_da_sub.dims:
                lat_vals = ds_da_sub.lat
            else:
                lat_vals = ds_da_sub.rlat

            # Determine minimum an maximum values.
            lat_min_data = float(lat_vals[0])
            lat_max_data = float(lat_vals[len(lat_vals) - 1])
            lat_min = max(float(lat_vals.min()), min(lat) - lat_buffer)
            lat_max = min(float(lat_vals.max()), max(lat) + lat_buffer)

            # Invert coordinates, then add minimum and maximum values in array.
            invert_coords = (((lat_min_data > lat_max_data) and (lat_min < lat_max)) or
                             ((lat_min_data < lat_max_data) and (lat_min > lat_max)))
            lat_l = [lat_min, lat_max] if not invert_coords else [lat_max, lat_min]

            # Select cells.
            if c.DIM_LATITUDE in ds_da_sub.dims:
                ds_da_sub = ds_da_sub.sel(latitude=slice(*lat_l))
                lat_n = len(ds_da_sub.latitude)
            elif c.DIM_LAT in ds_da_sub.dims:
                ds_da_sub = ds_da_sub.sel(lat=slice(*lat_l))
                lat_n = len(ds_da_sub.lat)
            else:
                ds_da_sub = ds_da_sub.sel(rlat=slice(*lat_l))
                lat_n = len(ds_da_sub.rlat)

        # Perform subset again, this time with a buffer (equivalent to data resolution).
        if (i_pass < n_pass - 1) and force_2x2_grid:
            if (lon_n <= 1) or (lat_n <= 1):
                if lon_n <= 1:
                    lon_buffer = lon_cs * 0.5 * float(i_pass + 1)
                if lat_n <= 1:
                    lat_buffer = lat_cs * 0.5 * float(i_pass + 1)
            else:
                break

        else:
            break

    # Time.
    if (len(t) > 0) and (c.DIM_TIME in ds_da_sub.dims):
        time_l = list(np.unique(ds_da_sub.time.dt.year))
        time_min = max(min(time_l), min(t))
        time_max = min(max(time_l), max(t))
        ds_da_sub = ds_da_sub.sel(time=slice(str(time_min), str(time_max)))

    # Adjust grid.
    if (len(lon) > 0) or (len(lat) > 0) or (len(t) > 0):
        try:
            grid = ds_da[vi_name].attrs[c.ATTRS_GMAP] if isinstance(ds_da, xr.Dataset) else ds_da.attrs[c.ATTRS_GMAP]
            ds_da_sub[grid] = ds_da[grid]
        except KeyError:
            pass

    return ds_da_sub


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
    per: [float],
    drop: Optional[bool] = True
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
    drop: Optional[bool]
        If True, drops dimensions of length 1.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_res = ds.copy(deep=True)

    ds_res = ds_res.where((ds_res.time.dt.year >= per[0]) & (ds_res.time.dt.year <= per[1]), drop=drop)

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
    if c.DIM_LONGITUDE in list(ds.dims):
        da_lon = ds[c.DIM_LONGITUDE]
        da_lat = ds[c.DIM_LATITUDE]
    elif c.DIM_LON in list(ds.dims):
        da_lon = ds[c.DIM_LON]
        da_lat = ds[c.DIM_LAT]
    else:
        da_lon = ds[c.DIM_RLON]
        da_lat = ds[c.DIM_RLAT]

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
    if c.DIM_LONGITUDE in list(ds_to.dims):
        ds_to[c.DIM_LONGITUDE] = lon_vals
        ds_to[c.DIM_LATITUDE] = lat_vals
    elif c.DIM_RLON in list(ds_to.dims):
        ds_to[c.DIM_RLON] = lon_vals
        ds_to[c.DIM_RLAT] = lat_vals
    else:
        ds_to[c.DIM_LON] = lon_vals
        ds_to[c.DIM_LAT] = lat_vals

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

    if c.ATTRS_UNITS in ds_from.attrs:
        ds_to.attrs[c.ATTRS_UNITS] = ds_from.attrs[c.ATTRS_UNITS]
    if isinstance(ds_from, xr.Dataset) and (var in ds_from.data_vars):
        if c.ATTRS_UNITS in ds_from[var].attrs:
            ds_to[var].attrs[c.ATTRS_UNITS] = ds_from[var].attrs[c.ATTRS_UNITS]

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
            if dim_lat != c.DIM_LAT:
                ds_res = ds_res.rename({dim_lon: c.DIM_LON, dim_lat: c.DIM_LAT})
            if var != "":
                if c.ATTRS_GMAP not in ds_res[var].attrs:
                    ds_res[var].attrs[c.ATTRS_GMAP] = "regular_lon_lat"

            # Subset by shape.
            logger = logging.getLogger()
            level = logger.level
            logger.setLevel(logging.CRITICAL)
            ds_res = subset.subset_shape(ds_res, cntx.p_bounds)
            logger.setLevel(level)

            # Recover initial dimension names.
            if dim_lat != c.DIM_LAT:
                ds_res = ds_res.rename({c.DIM_LON: dim_lon, c.DIM_LAT: dim_lat})

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
    _units = None
    if c.ATTRS_UNITS in da_res.attrs:
        _units = da_res.attrs[c.ATTRS_UNITS]

    # Rename spatial dimensions.
    if dims_data != dims_mask:
        da_res = da_res.rename({list(dims_data)[0]: list(dims_mask)[0], list(dims_data)[1]: list(dims_mask)[1]})

    # Apply mask.
    if c.DIM_TIME in da.dims:
        n_time = len(da[c.DIM_TIME])
        da_res[0:n_time, :, :] = da_res[0:n_time, :, :].values * da_mask.values
    else:
        da_res[:, :] = da_res[:, :].values * da_mask.values

    # Restore spatial dimensions.
    if dims_data != dims_mask:
        da_res = da_res.rename({list(dims_mask)[0]: list(dims_data)[0], list(dims_mask)[1]: list(dims_data)[1]})

    # Restore units.
    if _units is not None:
        da_res.attrs[c.ATTRS_UNITS] = _units

    # Drop coordinates.
    try:
        da_res = da_res.reset_coords(names=c.DIM_TIME, drop=True)
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

    if c.DIM_LAT in ds_or_da.dims:
        coord_dict = [c.DIM_LON, c.DIM_LAT]
    elif c.DIM_RLAT in ds_or_da.dims:
        coord_dict = [c.DIM_RLON, c.DIM_RLAT]
    else:
        coord_dict = [c.DIM_LONGITUDE, c.DIM_LATITUDE]

    return coord_dict


def rename_dimensions(
    da: xr.DataArray,
    lat_name: str = c.DIM_LATITUDE,
    lon_name: str = c.DIM_LONGITUDE
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
            da = da.rename({"dim_0": c.DIM_TIME})
            da = da.rename({"dim_1": lat_name, "dim_2": lon_name})
        elif (c.DIM_LAT in list(da.dims)) or (c.DIM_LON in list(da.dims)):
            da = da.rename({c.DIM_LAT: lat_name, c.DIM_LON: lon_name})
        elif (c.DIM_RLAT in list(da.dims)) or (c.DIM_RLON in list(da.dims)):
            da = da.rename({c.DIM_RLAT: lat_name, c.DIM_RLON: lon_name})
        elif (lat_name not in list(da.dims)) and (lon_name not in list(da.dims)):
            if lat_name == c.DIM_LATITUDE:
                da = da.expand_dims(latitude=1)
            if lon_name == c.DIM_LONGITUDE:
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
    lon_vals = ds_or_da_res[c.DIM_LONGITUDE]
    lat_vals = ds_or_da_res[c.DIM_LATITUDE]
    lon_monotonic_inc = bool(lon_vals[0] < lon_vals[len(lon_vals) - 1])
    lat_monotonic_inc = bool(lat_vals[0] < lat_vals[len(lat_vals) - 1])

    # Flip values.
    if not lon_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(c.DIM_LONGITUDE, ascending=True)
    if not lat_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(c.DIM_LATITUDE, ascending=True)

    # Interpolate, layer by layer (limit=1).
    for t in range(len(ds_or_da_res[c.DIM_TIME])):
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
        ds_or_da_res = ds_or_da_res.sortby(c.DIM_LONGITUDE, ascending=False)
    if not lat_monotonic_inc:
        ds_or_da_res = ds_or_da_res.sortby(c.DIM_LATITUDE, ascending=False)

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


def standardize_netcdf(
    ds_da: Union[xr.Dataset, xr.DataArray],
    vi_name: Optional[str] = "",
    template: Optional[Union[xr.Dataset, xr.DataArray, List[str]]] = None,
    sort: Optional[bool] = True,
    rename: Optional[bool] = True,
    drop: Optional[bool] = True
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Standardize a Dataset or DataArray by sorting dimensions, renaming dimensions and droping variables.

    Dimensions are renamed based on a template, which can be a DataSet, a DataArray, or a list of strings. If no
    template is provided but the sort option is enabled, columns are sorted by time, latitude and longitude.

    The columns that can be dropped are the following: 'lat' and 'lon' (if they are not dimensions), 'lat_vertices',
    'lon_vertices', 'rotated_latitude_longitude', 'height'.

    Parameters
    ----------
    ds_da: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray whose dimensions need to be reordered and potentially renamed.
    vi_name: Optional[str]
        Climate variable or index name.
    template: Union[xr.Dataset, xr.DataArray, List[str]]
        DataArray serving as template.
    sort: Optional[bool]
        If True, sort dimensions according to template
    rename: Optional[bool]
        If True, rename dimensions according to template.
    drop: Optional[bool]
        If True, remove unncessary columns.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]:
        Dataset or DataArray with a potentially altered order of dimensions.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Exit if the types of 'ds_da' and 'template' are different.
    # if (type(ds_da) != type(template):
    #     return ds_da

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
        dims_template = [c.DIM_TIME] if c.DIM_TIME in ds_da.dims else []
        dims_template = dims_template + [c.DIM_LATITUDE, c.DIM_LONGITUDE]

    # Sort dimensions.
    if sort:

        # Determine the required order of dimensions (referring to the names in 'ds_da'), then drop 'lon' and 'lat'.
        sort_needed = False
        dims_order = []
        for dim_template in dims_template:
            dim = str(dim_template).replace(c.DIM_RLAT, c.DIM_LAT).replace(c.DIM_RLON, c.DIM_LON).\
                replace(c.DIM_LATITUDE, c.DIM_LAT).replace(c.DIM_LONGITUDE, c.DIM_LON)
            dim_current = [i for i in dims_current if dim in i][0]
            dims_order.append(dim_current)

        # Sort dimensions.
        if sort_needed:
            if isinstance(ds_da, xr.Dataset):
                if c.DIM_LOCATION in dims_order:
                    ds_da[vi_name] = ds_da[vi_name].transpose(dims_order[0], dims_order[1])
                else:
                    ds_da[vi_name] = ds_da[vi_name].transpose(dims_order[0], dims_order[1], dims_order[2])
            else:
                if c.DIM_LOCATION in dims_order:
                    ds_da = ds_da.transpose(dims_order[0], dims_order[1])
                else:
                    ds_da = ds_da.transpose(dims_order[0], dims_order[1], dims_order[2])

    # Rename dimensions.
    if rename:
        for dim in [c.DIM_LAT, c.DIM_LON]:
            dim_current = [i for i in dims_current if dim in i][0]
            dim_template = [i for i in dims_template if dim in i][0]
            if dim_current != dim_template:
                ds_da = ds_da.rename({dim_current: dim_template})
            if drop and (dim in ds_da.dims) and (dim != dim_template):
                ds_da = ds_da.drop_vars(dim)

    # Drop columns.
    if drop and isinstance(ds_da, xr.Dataset):
        columns = ["lat", "lon", "lat_vertices", "lon_vertices", "rotated_latitude_longitude", "height"]
        for column in columns:
            if column in list(ds_da.variables):
                ds_da = ds_da.drop_vars([column])

    # Sort by time dimension to avoid non-monotonic issue during resampling.
    if c.DIM_TIME in ds_da.dims:
        ds_da = ds_da.sortby(c.DIM_TIME)

    return ds_da


def get_logger(
    logger_name: str
) -> Union[logging.Logger, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get logger.

    Parameters
    ----------
    logger_name: str
         Logger name.

    Returns
    -------
    Union[logging.Logger, None]
        Logger.
    --------------------------------------------------------------------------------------------------------------------
    """

    if logger_name == "root":
        return logging.root
    else:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for i in range(len(loggers)):
            if loggers[i].name == logger_name:
                return loggers[i]

    return None


def get_logger_level(
    logger_name: str
) -> int:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get logger level.

    Returns
    -------
    int
         Logger level.
    --------------------------------------------------------------------------------------------------------------------
    """

    logger = get_logger(logger_name)
    if logger is not None:
        return logger.level

    return -1


def set_logger_level(
    logger_name: str,
    logger_level: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Set logger level.

    Parameters
    ----------
    logger_name: str
        Logger name.
    logger_level: str
         Logger level.
    --------------------------------------------------------------------------------------------------------------------
    """

    if logger_name == "root":
        logging.root.setLevel(logger_level)
    else:
        logger = get_logger(logger_name)
        logger.setLevel(logger_level)

    return


def units(
    ds_da: Union[xr.Dataset, xr.DataArray],
    layer_name: str
) -> Union[str, int]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get units.

    Parameters
    ----------
    ds_da: Uniont[xr.Dataset, xr.DataArray]
        Dataset.
    layer_name: str
        Layer name.

    Returns
    -------
    Union[str, int]
        Units.
    --------------------------------------------------------------------------------------------------------------------
    """

    if isinstance(ds_da, xr.Dataset):
        if c.ATTRS_UNITS in ds_da[layer_name].attrs:
            _units = ds_da[layer_name].attrs[c.ATTRS_UNITS]
        elif c.ATTRS_UNITS in ds_da.data_vars:
            _units = ds_da[c.ATTRS_UNITS]
        else:
            _units = ""
    else:
        if c.ATTRS_UNITS in ds_da.attrs:
            _units = ds_da.attrs[c.ATTRS_UNITS]
        else:
            _units = ""

    return _units


def set_units(
    ds_da: Union[xr.Dataset, xr.DataArray],
    vi_name: str,
    units_new: Optional[Union[str, int]] = ""
) -> Union[xr.Dataset, xr.DataArray]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert units.

    Parameters
    ----------
    ds_da: Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray.
    vi_name: str
        Variable or index name.
    units_new: Optional[Union[str, int]]
        Units to convert to. If units are not provided the units are those of the generated visual elements.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray with updated units.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get old units.
    units_old = units(ds_da, vi_name)

    # Save attributes.
    attrs = ds_da[vi_name].attrs if isinstance(ds_da, xr.Dataset) else ds_da.attrs
    if len(attrs) == 0:
        attrs = {c.ATTRS_UNITS: ""}

    # Assign new units.
    if units_new == "":
        if vi_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
            units_new = c.UNIT_C
        elif vi_name in [c.V_PR, c.V_EVSPSBL, c.V_EVSPSBLPOT]:
            units_new = c.UNIT_mm
        elif vi_name in [c.V_UAS, c.V_VAS, c.V_SFCWINDMAX]:
            units_new = c.UNIT_km_h

    # Temperature: K -> C.
    if vi_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
        if (units_old == c.UNIT_K) and (units_new == c.UNIT_C):
            ds_da = ds_da - c.d_KC
        elif (units_old == c.UNIT_C) and (units_new == c.UNIT_K):
            ds_da = ds_da + c.d_KC

    # Precipitation, evaporation, evapotranspiration: kg/m2s2 -> mm.
    elif vi_name in [c.V_PR, c.V_EVSPSBL, c.V_EVSPSBLPOT]:
        if (units_old == c.UNIT_kg_m2s1) and (units_new == c.UNIT_mm):
            ds_da = ds_da * c.SPD
        elif (units_old == c.UNIT_mm) and (units_new == c.UNIT_kg_m2s1):
            ds_da = ds_da / c.SPD

    # Wind: m/s -> km/h.
    elif vi_name in [c.V_UAS, c.V_VAS, c.V_SFCWINDMAX]:
        if (units_old == c.UNIT_m_s) and (units_new == c.UNIT_km_h):
            ds_da = ds_da * c.KM_H_PER_M_S
        elif (units_old == c.UNIT_km_h) and (units_new == c.UNIT_m_s):
            ds_da = ds_da / c.KM_H_PER_M_S

    # Restore attributes and set new units.
    if units_old != units_new:
        if isinstance(ds_da, xr.Dataset):
            ds_da[vi_name].attrs = attrs
            ds_da[vi_name].attrs[c.ATTRS_UNITS] = units_new
        else:
            ds_da.attrs = attrs
            ds_da.attrs[c.ATTRS_UNITS] = units_new

    return ds_da
