# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Unit tester.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2021 Ouranos Inc., Canada
#
# Legend (case description):
#   |  = Year separator
#   A  = start_date
#   B  = end_date
#   .  = dry day(s)
#   T  = threshold
#   Tw = wet threshold
#   Td = dry threshold
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import copy
import datetime
import indices
import numpy as np
import pandas as pd

import utils
import xarray as xr
import xclim.indices as xindices
import xclim.testing._utils as xutils
from typing import List, Union
from xclim.core.units import convert_units_to, rate2amount, to_agg_units


def get_sample_data(var: str) -> Union[xr.DataArray, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Get sample dataset.

    Parameters
    ----------
    var : str
        Variable.

    Returns
    -------
    Union[xr.DataArray, None]
        Sample dataset for a given variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = ""
    if var == cfg.var_cordex_pr:
        path = "ERA5/daily_surface_cancities_1990-1993.nc"

    if path != "":
        return xutils.open_dataset(path).pr
    else:
        return None


def generate(
    var_or_idx: str,
    start_year: int,
    n_years: int,
    val: float,
    freq: str = "D",
    locations: Union[str, List[str]] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a scenario.

    Parameters
    ----------
    var_or_idx : str
        Variable or index
    start_year : int
        First year.
    n_years : int
        Number of years
    val : float
        Value to assign to all cells.
    freq : str
        Frequency.
    locations : Union[str, List[str]]
        Locations.
    --------------------------------------------------------------------------------------------------------------------
    """

    n_loc = 0 if locations == "" else len(locations)

    # Build arrays of values and time.
    dpy = 365 if freq == "D" else 1
    if n_loc == 0:
        arr = [[[0] * dpy * n_years]]
    else:
        arr = [[0] * dpy * n_years] * (n_loc if n_loc > 0 else 1)
    time = pd.date_range(str(start_year) + "-01-01", periods=n_years * dpy, freq=freq)

    # Variable.
    if var_or_idx == cfg.var_cordex_pr:
        description = "Precipitation"
        units = "mm"
    else:
        description = var_or_idx
        units = cfg.get_unit(var_or_idx)

    # Coordinates.
    if n_loc == 0:
        dims = [cfg.dim_longitude, cfg.dim_latitude, cfg.dim_time]
        longitude = [0]
        latitude = [0]
        coords = dict(
            longitude=([cfg.dim_longitude], longitude),
            latitude=([cfg.dim_latitude], latitude),
            time=time
        )
    else:
        dims = [cfg.dim_location, cfg.dim_time]
        coords = dict(
            location=([cfg.dim_location], locations),
            time=time
        )

    # Create data array.
    da = xr.DataArray(
        data=arr,
        dims=dims,
        coords=coords,
        attrs=dict(
            description=description,
            units=units,
        )
    )

    # Reorder dimensions.
    if n_loc == 0:
        da = da.transpose(cfg.dim_time, cfg.dim_latitude, cfg.dim_longitude)
    else:
        da = da.transpose(cfg.dim_location, cfg.dim_time)

    # Assign values.
    da = xr.ones_like(da).astype(bool) * val
    da.attrs[cfg.attrs_units] = units

    return da


def assign(
    da: xr.DataArray,
    start: Union[int, List[int]],
    end: Union[int, List[int]],
    vals: Union[float, List[float]],
    loc: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Assign values.

    Parameters
    ----------
    da: xr.DataArray
        Data array.
    start: int[]
        Start date.
        Option #1: year, e.g.: 1981
        Option #2: year, month and day, e.g.: [1981, 3, 1] for March 1st 1981.
    end: int[]
        End date.
        Same options as {start}
    vals: Union[float, List[float]]
        Value to assign.
    loc: str, optional
        Location.

    Returns
    -------
    None
    --------------------------------------------------------------------------------------------------------------------
    """

    def extract_year_doy(date_str: str):
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").timetuple()
        year = date.tm_year
        doy = date.tm_yday
        return year, doy

    if loc == "":

        # Assemble dates (in a string format).
        if isinstance(start, int):
            n = 1
            start_str = str(date(start, 1, 1))
        else:
            n = 365
            start_str = str(date(start[0], start[1], start[2]))
        if isinstance(end, int):
            end_str = str(date(end, 1, 1))
        else:
            end_str = str(date(end[0], end[1], end[2]))

        year_1, doy_1 = extract_year_doy(start_str)
        year_n, doy_n = extract_year_doy(end_str)
        t1 = (year_1 - int(da["time"].dt.year.min())) * n + (doy_1 - 1)
        tn = (year_n - int(da["time"].dt.year.min())) * n + (doy_n - 1)
        da.loc[slice(da[cfg.dim_time][t1], da[cfg.dim_time][tn])] = vals

    else:
        da[da.location == loc] = vals

    return da


def res_is_valid(res, res_expected) -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine if a result is valid.

    res : [Union[int, float]]
        Actual result (to be verified).
    res_expected : [Union[int, float]]
        Expected result.

    Returns
    -------
    bool
        True if the actual and expected results are equivalent.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert both arrays to float.
    try:
        for i in range(len(res)):
            res[i] = float(res[i])
            res_expected[i] = float(res_expected[i])
    except TypeError:
        pass

    # Compare arrays.
    try:
        np.testing.assert_equal(res, res_expected)
    except AssertionError:
        return False

    return True


def date(
    y: int,
    m: int,
    d: int
) -> str:

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine year, month and day into a string.

    Parameters
    ----------
    y : int
        Year.
    m : int
        Month.
    d : int
        Day.

    Returns
    -------
    str
        Date in a string format.
    --------------------------------------------------------------------------------------------------------------------
    """

    y_str = (str(y).zfill(4) if y != -1 else "")
    m_str = str(m).zfill(2)
    d_str = str(d).zfill(2)
    date_str = y_str + ("-" if y_str != "" else "") + m_str + "-" + d_str

    return date_str


def pr_series(values, start="7/1/2000", units="kg m-2 s-1"):

    """
    --------------------------------------------------------------------------------------------------------------------
    This is the same as the fixture in xclim.testing.tests.conftests.py.
    --------------------------------------------------------------------------------------------------------------------
    """

    coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
    return xr.DataArray(
        values,
        coords=[coords],
        dims="time",
        name="pr",
        attrs={
            "standard_name": "precipitation_flux",
            "cell_methods": "time: mean within days",
            "units": units
        }
    )


def dry_spell_total_length() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Test indices.dry_spell_total_length.

    Returns
    -------
    bool
        True if all tests are successful.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Algorithm (1= old/current; 2= new/proposed).
    algo = 2

    # Variable.
    var = cfg.var_cordex_pr

    # Operators.
    op_max = "max"
    op_sum = "sum"
    op_max_data = "max_data"
    op_sum_data = "sum_data"

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Loop through cases.
    error = False
    n_cases = 30
    for i in range(0, n_cases + 1):

        for op in [op_max, op_sum, op_max_data, op_sum_data]:

            # Parameters.
            freq = "YS"
            start_date = ""
            end_date = ""
            is_synthetic = op in [op_max, op_sum]

            # xclim ----------------------------------------------------------------------------------------------------

            # Case #0: Default xclim testing function.
            if i == 0:
                thresh, window = 3, 7
                if op in [op_max, op_sum]:
                    da_pr = pr_series(np.array([1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 0.01, 0.01, 0.01, 0.51,
                                                0.51, 0.75, 0.75, 0.51, 0.01, 0.01, 0.01, 1.01, 1.01, 1.01]))
                    da_pr.attrs["units"] = "mm/day"
                    res_expected = [12] if op == op_sum else [20]
                else:
                    da_pr = get_sample_data(cfg.var_cordex_pr)
                    if op == op_sum_data:
                        res_expected = [[50, 60, 73, 65],
                                        [67, 118, 87, 93],
                                        [142, 160, 166, 172],
                                        [227, 223, 203, 234],
                                        [145, 164, 202, 222]]
                    else:
                        res_expected = [[76, 97, 90, 85],
                                        [90, 144, 144, 133],
                                        [244, 250, 277, 267],
                                        [298, 266, 256, 282],
                                        [160, 208, 210, 250]]

            # {op} = "max" ---------------------------------------------------------------------------------------------

            # Case #1: | T A 14x. T B T | T |
            elif (i == 1) and (op == op_max):
                start_date, end_date = "03-01", "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                res_expected = [14, 0]

            # Case #2: | T A 13x. T B T | T |
            elif (i == 2) and (op == op_max):
                start_date, end_date = "03-01", "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 13], 0)
                res_expected = [0, 0]

            # Case #3: | T A 7x. T 7x. T B T | T |
            elif (i == 3) and (op == op_max):
                start_date, end_date = "03-01", "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 7], 0)
                assign(da_pr, [y1, 3, 9], [y1, 3, 15], 0)
                res_expected = [0, 0]

            # Case #4: | T . A 13x. T B T | T |
            elif (i == 4) and (op == op_max):
                start_date, end_date = "03-01", "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 2, 28], [y1, 3, 13], 0)
                res_expected = [13, 0]

            # Case #5: | T A T 13x. B . T | T |
            elif (i == 5) and (op == op_max):
                start_date, end_date = "03-01", "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 11, 18], [y1, 12, 1], 0)
                res_expected = [13, 0]

            # Case #6: | T A 14x. T 14x. B T | T |
            elif (i == 6) and (op == op_max):
                start_date, end_date = "03-01", "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                assign(da_pr, [y1, 8, 1], [y1, 8, 14], 0)
                res_expected = [28, 0]

            # Case #7: | T B 14x. T A T | T |
            elif (i == 7) and (op == op_max):
                start_date, end_date = "12-01", "02-28"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                res_expected = [0, 0]

            # Case #8: | T B 13x. T A T | T |
            elif (i == 8) and (op == op_max):
                start_date, end_date = "12-01", "02-28"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 13], 0)
                res_expected = [0, 0]

            # Case #9: | T B T 7x. T 7x. T A T | T |
            elif (i == 9) and (op == op_max):
                start_date, end_date = "12-01", "02-28"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 7], 0)
                assign(da_pr, [y1, 3, 9], [y1, 3, 15], 0)
                res_expected = [0, 0]

            # Case #10: | T B 14x. T A T | T |
            elif (i == 10) and (op == op_max):
                start_date, end_date = "12-01", "02-28"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 2, 28], [y1, 3, 13], 0)
                res_expected = [1, 0]

            # Case 11: | T B T 14x. A T | T |
            elif (i == 11) and (op == op_max):
                start_date, end_date = "12-01", "02-28"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 11, 18], [y1, 12, 1], 0)
                res_expected = [1, 0]

            # Case #12: | T B T 14x. T 14x. T A T | T |
            elif (i == 12) and (op == op_max):
                start_date, end_date = "12-01", "02-28"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                assign(da_pr, [y1, 8, 1], [y1, 8, 14], 0)
                res_expected = [0, 0]

            # Case #13: | T 14x. T | T |
            elif (i == 13) and (op == op_max):
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                res_expected = [14, 0]

            # Case #14: | T A 14x. T | T |
            elif (i == 14) and (op == op_max):
                start_date = "03-01"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                res_expected = [14, 0]

            # Case #15: | T 14x. T B T | T |
            elif (i == 15) and (op == op_max):
                end_date = "11-30"
                thresh, window   = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], 0)
                res_expected = [14, 0]

            # Case #16: | T 7x. A 7x. T | T |
            elif (i == 16) and (op == op_max):
                start_date = "03-01"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 2, 22], [y1, 3, 7], 0)
                res_expected = [7, 0]

            # Case #17: | T 7x. T B 7x. T | T |
            elif (i == 17) and (op == op_max):
                end_date = "11-30"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 11, 24], [y1, 12, 7], 0)
                res_expected = [7, 0]

            # Case #18: | T B T 7x. | 7x. T B T |
            elif (i == 18) and (op == op_max):
                start_date, end_date = "12-01", "01-31"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 12, 25], [y1 + 1, 1, 7], 0)
                res_expected = [7, 7]

            # Case #19: | 14x. T B T A T | T |
            elif (i == 19) and (op == op_max):
                start_date, end_date = "12-01", "01-31"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1, 1, 1], [y1, 1, 14], 0)
                res_expected = [14, 0]

            # Case #20: | T | T B T A T 14x. |
            elif (i == 20) and (op == op_max):
                start_date, end_date = "12-01", "01-31"
                thresh, window = 1, 14
                da_pr = generate(var, y1, n_years, thresh)
                assign(da_pr, [y1 + 1, 12, 18], [y1 + 1, 12, 31], 0)
                res_expected = [0, 14]

            # Case #21: | T 3x<T T 2x<T T 3x<T T | T 3x<T T 2x<T T 3x<T T |
            elif (i == 21) and (op == op_max):
                thresh, window = 3, 3
                da_pr = generate(var, y1, n_years, thresh)
                pr = thresh / window
                assign(da_pr, [y1, 3, 1], [y1, 3, 3], pr)
                assign(da_pr, [y1, 6, 1], [y1, 6, 2], pr)
                assign(da_pr, [y1, 9, 1], [y1, 9, 3], pr)
                assign(da_pr, [y2, 3, 1], [y2, 3, 3], pr)
                assign(da_pr, [y2, 6, 1], [y2, 6, 2], pr)
                assign(da_pr, [y2, 9, 1], [y2, 9, 3], pr)
                res_expected = [6, 6]

            # {op} = "sum" ---------------------------------------------------------------------------------------------

            # Case #22: | . | . |
            elif (i == 22) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                res_expected = [365, 365]

            # Case #23: | . T*13/14 . | . |
            elif (i == 23) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 3, 1], [y1, 3, 1], thresh - 1.0)
                res_expected = [365, 365]

            # Case #24: | . T . | . |
            elif (i == 24) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 3, 1], [y1, 3, 1], thresh)
                res_expected = [364, 365]

            # Case #25: | . T/2x3 . | . |
            elif (i == 25) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 3, 1], [y1, 3, 3], thresh / 2.0)
                res_expected = [364, 365]

            # Case #26: | . 14xT/14 . | . |
            elif (i == 26) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 3, 1], [y1, 3, 14], thresh / window)
                res_expected = [365, 365]

            # Case #27: | . 7xT/14 | 7xT/14 . |
            elif (i == 27) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 12, 25], [y2, 1, 7], thresh / window)
                res_expected = [365, 365]

            # Case #28: | . 7xT | 7xT . |
            elif (i == 28) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 12, 25], [y2, 1, 7], thresh)
                res_expected = [358, 358]

            # Case #29: | . 15xT/14 . | . |
            elif (i == 29) and (op == op_sum):
                thresh, window = 15, 15
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 3, 1], [y1, 3, 15], thresh / window * 3)
                res_expected = [358, 365]

            # Case #30: | . 9xT/3 . | . |
            elif (i == 30) and (op == op_sum):
                thresh, window = 3, 3
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 3, 1], [y1, 3, 9], thresh / window)
                res_expected = [360, 365]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Convert from precipitation amount to rate.
            if is_synthetic:
                da_pr = da_pr / cfg.spd
                da_pr.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1
            else:
                op = op_sum if op == op_sum_data else op_max

            # Exit if case does not apply.
            if (algo == 1) and ((start_date != "") or (end_date != "")) and\
               (i not in [1, 2, 3, 6, 13, 14, 15, 18, 19, 20, 21]):
                continue

            # Calculate indices using the old algorithm.
            if algo == 1:
                pram = rate2amount(da_pr, out_units="mm")
                thresh = convert_units_to(str(thresh) + " mm", pram)
                if op == op_max:
                    mask = xr.DataArray(pram.rolling(time=window, center=True).max() < thresh)
                else:
                    mask = xr.DataArray(pram.rolling(time=window, center=True).sum() < thresh)
                out = (mask.rolling(time=window, center=True).sum() >= 1).resample(time=freq).sum()
                da_idx = to_agg_units(out, pram, "count")

            # Calculate indices using the new algorithm.
            else:
                da_idx =\
                    indices.dry_spell_total_length(da_pr, str(thresh) + " mm", window, op, freq, start_date, end_date)

            # Reorder dimensions.
            if len(list(da_idx.dims)) > 1:
                da_idx = utils.reorder_dims(da_idx, da_pr)

            # Extract results.
            res = np.array(da_idx.squeeze())
            if not res_is_valid(res, res_expected):
                error = True

    return error


def rain_season_start() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Test indices.rain_season_start.

    Returns
    -------
    bool
        True if all tests are successful.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable.
    var = cfg.var_cordex_pr

    # Years.
    n_years = 2
    y1 = 1981

    # Operators.
    op_synthetic = "synthetic"
    op_data      = "data"

    # Parameters.
    thresh_wet   = 15  # Tw
    window_wet   = 3
    thresh_dry   = 1   # Td
    dry_days     = 10
    window_dry   = 30

    # Loop through cases.
    error = False
    n_cases = 9
    for i in range(0, n_cases + 1):

        for op in [op_synthetic, op_data]:

            is_synthetic = (op == op_synthetic)

            # xclim ----------------------------------------------------------------------------------------------------

            if i == 0:

                # Case #0: | . A . 3xTw/3 10xTd 9x. 11xTd . B | . |
                if op == op_synthetic:
                    start_date, end_date = "03-01", "12-31"  # A, B
                    da_pr = generate(var, y1, n_years, 0.0)
                    assign(da_pr, [y1, 1, 1], [y1, 3, 31], 1.01)
                    assign(da_pr, [y1, 4, 1], [y1, 4, 3], 5.01)
                    assign(da_pr, [y1, 4, 4], [y1, 4, 13], 1.01)
                    assign(da_pr, [y1, 4, 14], [y1, 4, 22], 0.99)
                    assign(da_pr, [y1, 4, 23], [y1, 5, 3], 1.01)
                    res_expected = [91, np.nan]

                # Case #0: real dataset.
                else:
                    start_date, end_date = "03-01", "05-31"  # A, B
                    thresh_wet = 25  # Tw
                    da_pr = get_sample_data(cfg.var_cordex_pr)
                    res_expected = [[89, 61, 66, 63],
                                    [92, 97, 70, 90],
                                    [np.nan, np.nan, np.nan, np.nan],
                                    [np.nan, 115, 130, np.nan],
                                    [np.nan, 60, 106, 62]]

            # workflow -------------------------------------------------------------------------------------------------

            # Case #1: | . A . 3xTw/3 30xTd . B | . |
            elif (i == 1) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 5, 3], thresh_dry)
                res_expected = [91, np.nan]

            # Case #2: | . A . 3xTw/3 10xTd 9x. 11xTd . B | . |
            elif (i == 2) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 4, 13], thresh_dry)
                assign(da_pr, [y1, 4, 23], [y1, 5, 3], thresh_dry)
                res_expected = [91, np.nan]

            # Case #3: | . A . 3xTw/3 10xTd 10x. 10xTd . B | . |
            elif (i == 3) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 4, 13], thresh_dry)
                assign(da_pr, [y1, 4, 24], [y1, 5, 3], thresh_dry)
                res_expected = [np.nan, np.nan]

            # Case #4: | . A . 2xTw*2/3 . 10xTd 9x. 11xTd . B | . |
            elif (i == 4) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 2], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 4, 13], thresh_dry)
                assign(da_pr, [y1, 4, 23], [y1, 5, 3], thresh_dry)
                res_expected = [np.nan, np.nan]

            # Case #5: | . A . 3xTw/3 25xTd . B | . |
            elif (i == 5) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 4, 28], thresh_dry)
                res_expected = [91, np.nan]

            # Case #6: | . A . 3xTw/3 5xTd 5x. 10xTd 5x. 5xTd . B | . |
            elif (i == 6) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 4, 8], thresh_dry)
                assign(da_pr, [y1, 4, 14], [y1, 4, 23], thresh_dry)
                assign(da_pr, [y1, 4, 29], [y1, 5, 3], thresh_dry)
                res_expected = [91, np.nan]

            # Case #7: | . A . 2xTw/3 . 30xTd . 3xTw/3 30xTd . B | . |
            elif (i == 7) and is_synthetic:
                start_date, end_date = "03-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 2], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 5, 3], thresh_dry)
                assign(da_pr, [y1, 6, 1], [y1, 6, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 6, 4], [y1, 7, 3], thresh_dry)
                res_expected = [152, np.nan]

            # Case #8: | . 3xTw/3 A 30xTd . B | . |
            elif (i == 8) and is_synthetic:
                start_date, end_date = "04-04", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 4, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 4, 4], [y1, 5, 3], thresh_dry)
                res_expected = [np.nan, np.nan]

            # Case #9: | . A . 3xTw/3 30xTd . | . B . |
            elif (i == 9) and is_synthetic:
                start_date, end_date = "09-01", "06-30"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 10, 1], [y1, 10, 3], thresh_wet / window_wet)
                assign(da_pr, [y1, 10, 4], [y1, 11, 2], thresh_dry)
                res_expected = [274, np.nan]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Convert from precipitation amount to rate.
            if is_synthetic:
                da_pr = da_pr / cfg.spd
                da_pr.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1

            # Calculate index.
            da_start = xr.DataArray(
                xindices.rain_season_start(da_pr, str(thresh_wet) + " mm", window_wet,
                                           str(thresh_dry) + " mm", dry_days, window_dry, start_date, end_date)
            )

            # Reorder dimensions.
            if len(list(da_start.dims)) > 1:
                da_start = utils.reorder_dims(da_start, da_pr)

            # Verify results.
            res = np.array(da_start.squeeze())
            if not res_is_valid(res, res_expected):
                error = True

    return error


def rain_season_end() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Test indices.rain_season_end.

    Returns
    -------
    bool
        True if all tests are successful.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable.
    var = cfg.var_cordex_pr

    # Methods.
    op_max      = "max"
    op_sum      = "sum"
    op_etp      = "etp"
    op_max_data = "max_data"
    op_sum_data = "sum_data"
    op_etp_data = "etp_data"

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Parameters:
    da_etp        = None
    da_start      = None
    da_start_next = None

    # Loop through cases.
    error = False
    n_cases = 19
    for i in range(0, n_cases + 1):

        for op in [op_max, op_sum, op_etp, op_max_data, op_sum_data, op_etp_data]:

            # Parameters.
            etp_rate   = 5
            etp_rate_i = etp_rate
            pr_rate    = etp_rate
            window     = 14
            if op == op_max:
                thresh = pr_rate            # T
            elif op  == op_sum:
                thresh = pr_rate * window   # T
            else:
                thresh = etp_rate * window  # T
            is_synthetic = op in [op_max, op_sum, op_etp]

            # xclim ----------------------------------------------------------------------------------------------------

            if i == 0:

                # Case #0: | . T A T 4T/5 3T/5 2T/5 1T/5 . B . | . |
                if op in [op_max, op_sum, op_etp]:
                    start_date, end_date = "09-01", "12-31"  # A, B
                    da_pr = generate(var, y1, n_years, 0.0)
                    assign(da_pr, [y1, 5, 4], [y1, 10, 15], pr_rate)
                    assign(da_pr, [y1, 10, 16], [y1, 10, 16], pr_rate * 4/5)
                    assign(da_pr, [y1, 10, 17], [y1, 10, 17], pr_rate * 3/5)
                    assign(da_pr, [y1, 10, 18], [y1, 10, 18], pr_rate * 2/5)
                    assign(da_pr, [y1, 10, 19], [y1, 10, 19], pr_rate * 1/5)
                    if op == op_max:
                        res_expected = [288, np.nan]
                    elif op == op_sum:
                        res_expected = [275, np.nan]
                    else:
                        res_expected = [306, np.nan]

                # Case #0: real dataset.
                else:
                    start_date, end_date = "06-01", "09-30"  # A, B
                    if op == op_max_data:
                        thresh = 5.0   # T
                    elif op == op_sum_data:
                        thresh = 10.0  # T
                    else:
                        thresh = 20.0  # T
                    window = 10
                    da_pr = get_sample_data(cfg.var_cordex_pr)
                    if op == op_max_data:
                        res_expected = [[190, 152, 256, 217],
                                        [204, 152, 202, 179],
                                        [176, 152, 152, 152],
                                        [162, 196, 156, 152],
                                        [161, 152, 152, 165]]
                    elif op == op_sum_data:
                        res_expected = [[186, 152, 232, 185],
                                        [204, 152, 160, 179],
                                        [176, 152, 152, 153],
                                        [163, 196, 156, 152],
                                        [161, 152, 152, 165]]
                    else:
                        res_expected = [[195, 153, 238, 166],
                                        [152, 156, 259, 178],
                                        [200, 154, 157, 152],
                                        [269, 152, 228, 152],
                                        [201, 154, 157, 216]]

            # {op} = any -----------------------------------------------------------------------------------------------

            # Case #1: | . A . | . B . |
            elif (i == 1) and (op in [op_max, op_sum, op_etp]):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                res_expected = [np.nan, np.nan]

            # Case #2: | T/14 A T/14 | T/14 B T/14 |
            elif (i == 2) and (op in [op_max, op_sum, op_etp]):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, pr_rate)
                res_expected = [np.nan, np.nan]

            # {op} = "etp" ---------------------------------------------------------------------------------------------

            # Case #3: | . A 30xT/14 . B | . |
            elif (i == 3) and (op == op_etp):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate)
                res_expected = [287, np.nan]

            # Case #4: | . A 30xT/14 T/28 B | . |
            elif (i == 4) and (op == op_etp):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate)
                assign(da_pr, [y1, 10, 1], [y1, 12, 31], pr_rate / 2)
                res_expected = [301, np.nan]

            # Case #5: | . A 30xT/14 . B | . | (no etp)
            elif (i == 5) and (op == op_etp):
                etp_rate_i = 0.0
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate)
                res_expected = [np.nan, np.nan]

            # Case #6: # | . A 30xT/14 . B | . | (2 x etp)
            elif (i == 6) and (op == op_etp):
                etp_rate_i = 2 * etp_rate
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate)
                res_expected = [280, np.nan]

            # Case #7: | . A 30xT/14 2x. 2xT/14 . B | . | (2 x etp)
            elif (i == 7) and (op == op_etp):
                etp_rate_i = 2 * etp_rate
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate)
                assign(da_pr, [y1, 10, 3], [y1, 10, 4], 2 * pr_rate)
                res_expected = [284, np.nan]

            # Case #8: | . A 15xT/14 | 15xT/14 . B . |
            elif (i == 8) and (op == op_etp):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 10, 1], [y1, 12, 16], pr_rate)
                assign(da_pr, [y2, 1, 1], [y2, 1, 15], pr_rate)
                res_expected = [np.nan, 29]

            # {op} = "max" ---------------------------------------------------------------------------------------------

            # Case #9: | . T A 30xT . B | . |
            elif (i == 9) and (op == op_max):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate)
                res_expected = [273, np.nan]

            # Case #10: | . T A . B | . |
            elif (i == 10) and (op == op_max):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 8, 31], pr_rate)
                res_expected = [np.nan, np.nan]

            # Case #11: | . T/2 A 30xT/2 . B | . |
            elif (i == 11) and (op == op_max):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate / 2)
                res_expected = [244, np.nan]

            # Case #12: | . T A 15xT . B | . |
            elif (i == 12) and (op == op_max):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 10, 15], pr_rate)
                res_expected = [288, np.nan]

            # Case #13: | . T A . B 15xT | . |
            elif (i == 13) and (op == op_max):
                etp_rate_i = etp_rate
                start_date, end_date = "09-01", "12-16"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 8, 31], pr_rate)
                assign(da_pr, [y1, 12, 17], [y1, 12, 31], pr_rate)
                res_expected = [np.nan, np.nan]

            # Case #14: | . T A 24xT/14 7x. | 6x. T . B . |
            elif (i == 14) and (op == op_max):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 12, 24], pr_rate)
                assign(da_pr, [y2, 1, 7], [y2, 1, 7], pr_rate)
                res_expected = [np.nan, 7]

            # {method} = "cumul" ---------------------------------------------------------------------------------------

            # Case #15: | . T A 30xT . B | . |
            elif (i == 15) and (op == op_sum):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate * window)
                res_expected = [273, np.nan]

            # Case #16: | . T A 30xT T . B | . |
            elif (i == 16) and (op == op_sum):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate * window)
                assign(da_pr, [y1, 10, 1], [y1, 10, 1], pr_rate * window)
                res_expected = [274, np.nan]

            # Case #17: | . T A 30xT 7x. T . B | . |
            elif (i == 17) and (op == op_sum):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 9, 30], pr_rate * window)
                assign(da_pr, [y1, 10, 8], [y1, 10, 8], pr_rate * window)
                res_expected = [281, np.nan]

            # Case #18: | . T A 24xT 7x. | . B . |
            elif (i == 18) and (op == op_sum):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 12, 24], pr_rate * window)
                res_expected = [358, np.nan]

            # Case #19: | . T A 24xT/14 7x. | 6x. T . B . |
            elif (i == 19) and (op == op_sum):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = generate(var, y1, n_years, 0.0)
                assign(da_pr, [y1, 4, 1], [y1, 12, 24], pr_rate * window)
                assign(da_pr, [y2, 1, 7], [y2, 1, 7], pr_rate * window)
                res_expected = [np.nan, 7]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Convert from precipitation amount to rate.
            if is_synthetic:
                da_pr = da_pr / cfg.spd
                da_pr.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1
                if da_etp is not None:
                    da_etp = da_etp / cfg.spd
                    da_etp.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1
            else:
                op = op_max if op == op_max_data else op_sum if op == op_sum_data else op_etp

            # Calculate index.
            da_end = xr.DataArray(
                xindices.rain_season_end(da_pr, da_etp, da_start, da_start_next, op, str(thresh) + " mm", window,
                                         (str(etp_rate_i) if op == op_etp else "0") + " mm", start_date, end_date)
            )

            # Reorder dimensions.
            if len(list(da_end.dims)) > 1:
                da_end = utils.reorder_dims(da_end, da_pr)

            # Verify results.
            res = np.array(da_end.squeeze())
            if not res_is_valid(res, res_expected):
                error = True

    return error


def rain_season_length_prcptot() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Test indices.rain_season_length and indices.rain_season_prcptot.

    Returns
    -------
    bool
        True if all tests are successful.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable.
    var = cfg.var_cordex_pr

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Operators.
    op_synthetic = "synthetic"
    op_data      = "data"

    # Loop through cases.
    error = False
    n_cases = 5
    for i in range(0, n_cases + 1):

        # Parameters.
        da_pr = generate(var, y1, n_years, 1.0)

        for op in [op_synthetic, op_data]:
            is_synthetic = (op == op_synthetic)

            # xclim ----------------------------------------------------------------------------------------------------

            if i == 0:

                # Case #0: | . A1 . 3xTw/3 10xTd 9x. 11xTd . T A2 T 4T/5 3T/5 2T/5 1T/5 . B1=B2 | . |
                if is_synthetic:
                    da_pr = generate(var, y1, n_years, 0.0)
                    assign(da_pr, [y1, 1, 1], [y1, 3, 31], 1.01)
                    assign(da_pr, [y1, 4, 1], [y1, 4, 3], 5.01)
                    assign(da_pr, [y1, 4, 4], [y1, 4, 13], 1.01)
                    assign(da_pr, [y1, 4, 14], [y1, 4, 22], 0.99)
                    assign(da_pr, [y1, 4, 23], [y1, 5, 3], 1.01)
                    assign(da_pr, [y1, 5, 4], [y1, 10, 15], 5.0)
                    assign(da_pr, [y1, 10, 16], [y1, 10, 16], 4.0)
                    assign(da_pr, [y1, 10, 17], [y1, 10, 17], 3.0)
                    assign(da_pr, [y1, 10, 18], [y1, 10, 18], 2.0)
                    assign(da_pr, [y1, 10, 19], [y1, 10, 19], 1.0)
                    da_start = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                    assign(da_start, y1, y1, 91)
                    da_end = generate(cfg.idx_rain_season_end, y1, n_years, np.nan, "YS")
                    assign(da_end, y1, y1, 288)
                    res_expected_length = [198, np.nan]
                    res_expected_prcptot = [870, np.nan]

                # Case #0: real dataset.
                else:
                    da_pr = get_sample_data(cfg.var_cordex_pr)
                    locations = list(da_pr.location.values)
                    da_start = generate(cfg.idx_rain_season_start, 1990, 4, np.nan, "YS", locations)
                    assign(da_start, [], [], [89, 61, 66, 63], locations[0])
                    assign(da_start, [], [], [92, 97, 70, 90], locations[1])
                    assign(da_start, [], [], [np.nan, 115, 130, np.nan], locations[3])
                    assign(da_start, [], [], [np.nan, 60, 106, 62], locations[4])
                    da_end = generate(cfg.idx_rain_season_end, 1990, 4, np.nan, "YS", locations)
                    assign(da_end, [], [], [190, 152, 256, 217], locations[0])
                    assign(da_end, [], [], [204, 152, 202, 179], locations[1])
                    assign(da_end, [], [], [176, 152, 152, 152], locations[2])
                    assign(da_end, [], [], [162, 196, 156, 152], locations[3])
                    assign(da_end, [], [], [161, 152, 152, 165], locations[4])
                    res_expected_length = [[102, 92, 191, 155],
                                           [113, 56, 133, 90],
                                           [np.nan, np.nan, np.nan, np.nan],
                                           [np.nan, 82, 27, np.nan],
                                           [np.nan, 93, 47, 104]]
                    res_expected_prcptot = [[1402, 1143, 1951, 2067],
                                            [1471, 660, 1256, 1121],
                                            [np.nan, np.nan, np.nan, np.nan],
                                            [np.nan, 834, 242, np.nan],
                                            [np.nan, 1149, 402, 1278]]

            # workflow -------------------------------------------------------------------------------------------------

            # Case #1: | . A1 . B1 . | . A2 . B2 . |
            elif (i == 1) and (op == op_synthetic):
                da_start = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                for j in range(n_years):
                    assign(da_start, y1 + j, y1 + j, 91 + j)
                da_end = generate(cfg.idx_rain_season_end, y1, n_years, np.nan, "YS")
                for j in range(n_years):
                    assign(da_end, y1 + j, y1 + j, 273 - j)
                res_expected_length = res_expected_prcptot = [273 - 91 + 1, 272 - 92 + 1]

            # Case #2: | . A1 . | . B1 . |
            elif (i == 2) and (op == op_synthetic):
                da_start = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                assign(da_start, y1, y1, 273)
                da_end = generate(cfg.idx_rain_season_end, y1, n_years, np.nan, "YS")
                assign(da_end, y2, y2, 91)
                res_expected_length = res_expected_prcptot = [365 - 273 + 1 + 91, np.nan]

            # Case #3: | . B0 . A1 . | . B1 . A2 . |
            elif (i == 3) and (op == op_synthetic):
                da_start = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                for y in [y1, y2]:
                    assign(da_start, y, y, 273)
                da_end = generate(cfg.idx_rain_season_end, y1, n_years, np.nan, "YS")
                for y in [y1, y2]:
                    assign(da_end, y, y, 91)
                res_expected_length = res_expected_prcptot = [365 - 273 + 1 + 91, np.nan]

            # Case #4: | . B1 . | . A2 . |
            elif (i == 4) and (op == op_synthetic):
                da_start = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                assign(da_start, y2, y2, 91)
                da_end = generate(cfg.idx_rain_season_end, y1, n_years, np.nan, "YS")
                assign(da_end, y1, y1, 273)
                res_expected_length = res_expected_prcptot = [np.nan, np.nan]

            # Case #5: | . B1 A1 . | . |
            elif (i == 5) and (op == op_synthetic):
                da_start = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                assign(da_start, y1, y1, 91)
                da_end = generate(cfg.idx_rain_season_end, y1, n_years, np.nan, "YS")
                assign(da_end, y1, y1, 90)
                res_expected_length = res_expected_prcptot = [np.nan, np.nan]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Convert from precipitation amount to rate.
            if is_synthetic:
                da_pr = da_pr / cfg.spd
                da_pr.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1

            # Calculate indices.
            da_length = xr.DataArray(xindices.rain_season_length(da_start, da_end))
            da_prcptot = xr.DataArray(xindices.rain_season_prcptot(da_pr, da_start, da_end))

            # Reorder dimensions.
            if len(list(da_length.dims)) > 1:
                da_length = utils.reorder_dims(da_length, da_pr)
            if len(list(da_prcptot.dims)) > 1:
                da_prcptot = utils.reorder_dims(da_prcptot, da_pr)

            # Verify results.
            res_length = np.array(da_length.squeeze())
            res_prcptot = np.array(da_prcptot.squeeze())
            if (not res_is_valid(res_length, res_expected_length)) or\
               (not res_is_valid(res_prcptot, res_expected_prcptot)):
                error = True

    return error


def rain_season() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Test indices.rain_season.

    Returns
    -------
    bool
        True if all tests are successful.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable.
    var = cfg.var_cordex_pr

    # Methods.
    e_op_max = "max"
    e_op_sum = "sum"
    e_op_etp = "etp"
    e_op_max_data = "max_data"

    # Years.
    n_years = 2
    y1 = 1981

    # Loop through cases.
    error = False
    n_cases = 2
    for i in range(0, n_cases + 1):

        for e_op in [e_op_max, e_op_sum, e_op_max_data]:
            is_synthetic = e_op in [e_op_max, e_op_sum, e_op_etp]

            # Default parameters.
            da_etp        = None
            da_start_next = None
            s_thresh_wet  = 15  # Tw
            s_window_wet  = 3
            s_thresh_dry  = 1  # Td
            s_dry_days    = 10
            s_window_dry  = 30
            e_etp_rate    = 5
            e_pr_rate     = s_thresh_dry
            e_window      = 14
            if e_op == e_op_max:
                e_thresh = e_pr_rate  # T
            elif e_op == e_op_sum:
                e_thresh = e_pr_rate * e_window  # T
            else:
                e_thresh = e_etp_rate * e_window  # T

            # xclim ----------------------------------------------------------------------------------------------------

            # Case #0: start = | . A . 3xTw/3 10xTd 9x. 11xTd .                           B | . |
            #          end   = | .                            T C T 4T/5 3T/5 2T/5 1T/5 . D | . |
            if (i == 0) and (e_op in [e_op_max, e_op_max_data]):

                if e_op == e_op_max:
                    s_thresh_wet = 15
                    s_window_wet = 3
                    s_thresh_dry = 1
                    s_dry_days   = 10
                    s_window_dry = 30
                    s_start_date = "03-01"
                    s_end_date   = "12-31"
                    e_thresh     = 5
                    e_window     = 14
                    e_etp_rate   = 5
                    e_start_date = "09-01"
                    e_end_date   = "12-31"
                    da_pr = generate(var, y1, n_years, 0.0)
                    assign(da_pr, [y1, 1, 1], [y1, 3, 31], 1.01)
                    assign(da_pr, [y1, 4, 1], [y1, 4, 3], 5.01)
                    assign(da_pr, [y1, 4, 4], [y1, 4, 13], 1.01)
                    assign(da_pr, [y1, 4, 14], [y1, 4, 22], 0.99)
                    assign(da_pr, [y1, 4, 23], [y1, 5, 3], 1.01)
                    assign(da_pr, [y1, 5, 4], [y1, 10, 15], 5.0)
                    assign(da_pr, [y1, 10, 16], [y1, 10, 16], 4.0)
                    assign(da_pr, [y1, 10, 17], [y1, 10, 17], 3.0)
                    assign(da_pr, [y1, 10, 18], [y1, 10, 18], 2.0)
                    assign(da_pr, [y1, 10, 19], [y1, 10, 19], 1.0)
                    res_expected_start   = [91, np.nan]
                    res_expected_end     = [288, np.nan]
                    res_expected_length  = [198, np.nan]
                    res_expected_prcptot = [870, np.nan]

                else:
                    s_thresh_wet = 25
                    s_window_wet = 3
                    s_thresh_dry = 1
                    s_dry_days   = 10
                    s_window_dry = 30
                    s_start_date = "03-01"
                    s_end_date   = "12-31"
                    e_thresh     = 5
                    e_window     = 10
                    e_etp_rate   = 5
                    e_start_date = "06-01"
                    e_end_date   = "09-30"
                    da_pr = get_sample_data(cfg.var_cordex_pr)
                    res_expected_start = [[89, 61, 66, 63],
                                          [92, 97, 70, 90],
                                          [213, 230, 190, 187],
                                          [182, 115, 130, 162],
                                          [275, 60, 106, 62]]
                    res_expected_end = [[190, 152, 273, 219],
                                        [205, 152, 224, 179],
                                        [np.nan, np.nan, np.nan, 195],
                                        [213, 196, 157, np.nan],
                                        [np.nan, 172, 152, 165]]
                    res_expected_length = [[102, 92, 208, 157],
                                           [114, 56, 155, 90],
                                           [np.nan, np.nan, np.nan, 9],
                                           [32, 82, 28, np.nan],
                                           [np.nan, 113, 47, 104]]
                    res_expected_prcptot = [[1402, 1143, 2183, 2073],
                                            [1486, 660, 1494, 1121],
                                            [np.nan, np.nan, 4214, np.nan],
                                            [448, 834, 246, np.nan],
                                            [np.nan, 1335, 402, 1278]]

            # workflow -------------------------------------------------------------------------------------------------

            else:

                # Case #1-2: start      = | . A . 3xTw/3 30xTd .             B | . |
                #            end        = | .                  T/14 C T/14 . D | . |
                #            start_next = | .                       .      E . | . | (i == 2)
                if (i in [1, 2]) and (e_op == e_op_sum):
                    s_start_date, s_end_date = "03-01", "12-31"  # A, B
                    e_start_date, e_end_date = "09-01", "12-31"  # C, D
                    da_pr = generate(var, y1, n_years, 0.0)
                    assign(da_pr, [y1, 4, 1], [y1, 4, 3], s_thresh_wet / s_window_wet)
                    assign(da_pr, [y1, 4, 4], [y1, 9, 30], s_thresh_dry)
                    if i == 2:
                        da_start_next = generate(cfg.idx_rain_season_start, y1, n_years, np.nan, "YS")
                        assign(da_start_next, y1, y1, utils.doy_str_to_doy("09-05"))
                    res_expected_start = [91, np.nan]
                    if i == 1:
                        res_expected_end = [260, np.nan]
                    else:
                        res_expected_end = [da_start_next[0] - 1, np.nan]
                    res_expected_length = [res_expected_end[0] - res_expected_start[0] + 1, np.nan]
                    res_expected_prcptot = copy.deepcopy(res_expected_length)
                    res_expected_prcptot[0] += s_thresh_wet - s_window_wet

                else:
                    continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Convert from precipitation amount to rate.
            if is_synthetic:
                da_pr = da_pr / cfg.spd
                da_pr.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1
                if da_etp is not None:
                    da_etp = da_etp / cfg.spd
                    da_etp.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1

            # Calculate indices.
            da_start, da_end, da_length, da_prcptot =\
                xindices.rain_season(da_pr, da_etp, da_start_next, str(s_thresh_wet) + " mm", s_window_wet,
                                     str(s_thresh_dry) + " mm", s_dry_days, s_window_dry, s_start_date, s_end_date,
                                     e_op, str(e_thresh) + " mm", e_window,
                                     (str(e_etp_rate) if e_op == e_op_etp else "0") + " mm", e_start_date, e_end_date)

            # Reorder dimensions.
            if len(list(da_start.dims)) > 1:
                da_start = utils.reorder_dims(da_start, da_pr)
            if len(list(da_end.dims)) > 1:
                da_end = utils.reorder_dims(da_end, da_pr)
            if len(list(da_length.dims)) > 1:
                da_length = utils.reorder_dims(da_length, da_pr)
            if len(list(da_prcptot.dims)) > 1:
                da_prcptot = utils.reorder_dims(da_prcptot, da_pr)

            #  Verify results.
            res_start = np.array(da_start.squeeze())
            res_end = np.array(da_end.squeeze())
            res_length = np.array(da_length.squeeze())
            res_prcptot = np.array(da_prcptot.squeeze())
            if (not res_is_valid(res_start, res_expected_start)) or \
               (not res_is_valid(res_end, res_expected_end)) or \
               (not res_is_valid(res_length, res_expected_length)) or \
               (not res_is_valid(res_prcptot, res_expected_prcptot)):
                error = True

    return error


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    utils.log("=")
    utils.log("Step #0   Testing indices")

    utils.log("Step #0a  dry_spell_total_length")
    dry_spell_total_length()

    utils.log("Step #0b  rain_season_start")
    rain_season_start()

    utils.log("Step #0c  rain_season_end")
    rain_season_end()

    utils.log("Step #0d  rain_season_length/prcptot")
    rain_season_length_prcptot()

    utils.log("Step #0f  rain_season")
    rain_season()