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
from xclim.core.units import convert_units_to, rate2amount, to_agg_units


def gen_scen(
    var: str,
    start_year: int = 1981,
    n_years: int = 1,
    val: float = 0
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a scenario.

    Parameters
    ----------
    var : str
        Variable.
    start_year : int, optional
        First year.
    n_years : int, optional
        Number of years
    val : float, optional
        Value to assign to all cells.
    --------------------------------------------------------------------------------------------------------------------
    """

    arr = [[[0] * 365 * n_years]]
    longitude = [0]
    latitude = [0]
    time = pd.date_range(str(start_year) + "-01-01", periods=365 * n_years)

    # Variable.
    description = units = ""
    if var == cfg.var_cordex_pr:
        description = "Precipitation"
        units = "mm"

    # Create data array.
    da = xr.DataArray(
        data=arr,
        dims=[cfg.dim_longitude, cfg.dim_latitude, cfg.dim_time],
        coords=dict(
            longitude=([cfg.dim_longitude], longitude),
            latitude=([cfg.dim_latitude], latitude),
            time=time
        ),
        attrs=dict(
            description=description,
            units=units,
        )
    )

    # Reorder dimensions.
    da = da.transpose(cfg.dim_time, cfg.dim_latitude, cfg.dim_longitude)

    # Assign values.
    da = xr.ones_like(da).astype(bool) * val
    da.attrs[cfg.attrs_units] = units

    return da


def gen_idx(
        idx_name: str,
        start_year: int = 1981,
        n_years: int = 1,
        val: float = 0
):
    """
    --------------------------------------------------------------------------------------------------------------------
    Generate an index.

    Parameters
    ----------
    idx_name : str
        Index.
    start_year : int, optional
        First year.
    n_years : int, optional
        Number of years
    val : float, optional
        Value to assign to all cells.
    --------------------------------------------------------------------------------------------------------------------
    """

    arr = [[[0] * n_years]]
    longitude = [0]
    latitude = [0]
    time = pd.date_range(str(start_year) + "-01-01", periods=n_years, freq=cfg.freq_YS)

    # Variable.
    description = idx_name
    units = cfg.get_unit(idx_name)

    # Create data array.
    da = xr.DataArray(
        data=arr,
        dims=[cfg.dim_longitude, cfg.dim_latitude, cfg.dim_time],
        coords=dict(
            longitude=([cfg.dim_longitude], longitude),
            latitude=([cfg.dim_latitude], latitude),
            time=time
        ),
        attrs=dict(
            description=description,
            units=units,
        )
    )

    # Reorder dimensions.
    da = da.transpose(cfg.dim_time, cfg.dim_latitude, cfg.dim_longitude)

    # Assign values.
    da = xr.ones_like(da).astype(bool) * val
    da.attrs[cfg.attrs_units] = units

    return da


def assign(
    da: xr.DataArray,
    date_start: str,
    date_end: str,
    val: float
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Assign values.

    Parameters
    ----------
    da: xr.DataArray
        Data array.
    date_start: str
        Start date, as a string (ex: "1981-04-01" represents April 1st 1981).
    date_end: str
        End date, as a string.
    val: float
        Value to assign.
    --------------------------------------------------------------------------------------------------------------------
    """

    def extract_year_doy(date_str: str):
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").timetuple()
        year = date.tm_year
        doy = date.tm_yday
        return year, doy

    if len(date_start) == 4:
        n = 1
        date_start += "-01-01"
        date_end += "-01-01"
    else:
        n = 365

    year_1, doy_1 = extract_year_doy(date_start)
    year_n, doy_n = extract_year_doy(date_end)

    t1 = (year_1 - int(da["time"].dt.year.min())) * n + (doy_1 - 1)
    tn = (year_n - int(da["time"].dt.year.min())) * n + (doy_n - 1)
    da.loc[slice(da[cfg.dim_time][t1], da[cfg.dim_time][tn])] = val


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

    valid = True

    # Arrays of different lengths.
    if len(res) != len(res_expected):
        valid = False

    # Loop through results.
    else:
        for i in range(len(res)):
            res_i = float(res[i])
            res_expected_i = float(res_expected[i])
            if not ((np.isnan(res_i) and np.isnan(res_expected_i)) or
                    ((np.isnan(res_i) == False) and (np.isnan(res_expected_i) == False) and (res_i == res_expected_i))):
                valid = False
                break

    return valid


def dstr(
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
    date_str = str(y_str + ("-" if y_str != "" else "") + m_str + "-" + d_str)

    return date_str


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

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Loop through cases.
    error = False
    n_cases = 30
    for i in range(1, n_cases + 1):

        for op in [op_max, op_sum]:

            # Parameters.
            fill_value = (op == op_sum)
            freq = "YS"
            start_date = ""
            end_date = ""

            # {op} = "max" ---------------------------------------------------------------------------------------------

            # Case #1: | T A 14x. T B T | T |
            if (i == 1) and (op == op_max):
                start_date, end_date = dstr(-1, 3, 1), dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #2: | T A 13x. T B T | T |
            elif (i == 2) and (op == op_max):
                start_date, end_date = dstr(-1, 3, 1), dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 13)), 0)
                res_expected = [0, 0]

            # Case #3: | T A 7x. T 7x. T B T | T |
            elif (i == 3) and (op == op_max):
                start_date, end_date = dstr(-1, 3, 1), dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 7)), 0)
                assign(da_pr, str(dstr(y1, 3, 9)), str(dstr(y1, 3, 15)), 0)
                res_expected = [0, 0]

            # Case #4: | T . A 13x. T B T | T |
            elif (i == 4) and (op == op_max):
                start_date, end_date = dstr(-1, 3, 1), dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 2, 28)), str(dstr(y1, 3, 13)), 0)
                res_expected = [13, 0]

            # Case #5: | T A T 13x. B . T | T |
            elif (i == 5) and (op == op_max):
                start_date, end_date = dstr(-1, 3, 1), dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 11, 18)), str(dstr(y1, 12, 1)), 0)
                res_expected = [13, 0]

            # Case #6: | T A 14x. T 14x. B T | T |
            elif (i == 6) and (op == op_max):
                start_date, end_date = dstr(-1, 3, 1), dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                assign(da_pr, str(dstr(y1, 8, 1)), str(dstr(y1, 8, 14)), 0)
                res_expected = [28, 0]

            # Case #7: | T B 14x. T A T | T |
            elif (i == 7) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 2, 28)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [0, 0]

            # Case #8: | T B 13x. T A T | T |
            elif (i == 8) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 2, 28)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 13)), 0)
                res_expected = [0, 0]

            # Case #9: | T B T 7x. T 7x. T A T | T |
            elif (i == 9) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 2, 28)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 7)), 0)
                assign(da_pr, str(dstr(y1, 3, 9)), str(dstr(y1, 3, 15)), 0)
                res_expected = [0, 0]

            # Case #10: | T B 14x. T A T | T |
            elif (i == 10) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 2, 28)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 2, 28)), str(dstr(y1, 3, 13)), 0)
                res_expected = [1, 0]

            # Case 11: | T B T 14x. A T | T |
            elif (i == 11) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 2, 28)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 11, 18)), str(dstr(y1, 12, 1)), 0)
                res_expected = [1, 0]

            # Case #12: | T B T 14x. T 14x. T A T | T |
            elif (i == 12) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 2, 28)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                assign(da_pr, str(dstr(y1, 8, 1)), str(dstr(y1, 8, 14)), 0)
                res_expected = [0, 0]

            # Case #13: | T 14x. T | T |
            elif (i == 13) and (op == op_max):
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #14: | T A 14x. T | T |
            elif (i == 14) and (op == op_max):
                start_date = dstr(-1, 3, 1)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #15: | T 14x. T B T | T |
            elif (i == 15) and (op == op_max):
                end_date = dstr(-1, 11, 30)
                thresh, window   = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #16: | T 7x. A 7x. T | T |
            elif (i == 16) and (op == op_max):
                start_date = dstr(-1, 3, 1)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 2, 22)), str(dstr(y1, 3, 7)), 0)
                res_expected = [7, 0]

            # Case #17: | T 7x. T B 7x. T | T |
            elif (i == 17) and (op == op_max):
                end_date = dstr(-1, 11, 30)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 11, 24)), str(dstr(y1, 12, 7)), 0)
                res_expected = [7, 0]

            # Case #18: | T B T 7x. | 7x. T B T |
            elif (i == 18) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 1, 31)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 12, 25)), str(dstr(y1 + 1, 1, 7)), 0)
                res_expected = [7, 7]

            # Case #19: | 14x. T B T A T | T |
            elif (i == 19) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 1, 31)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1, 1, 1)), str(dstr(y1, 1, 14)), 0)
                res_expected = [14, 0]

            # Case #20: | T | T B T A T 14x. |
            elif (i == 20) and (op == op_max):
                start_date, end_date = dstr(-1, 12, 1), dstr(-1, 1, 31)
                thresh, window = 1, 14
                da_pr = gen_scen(var, y1, n_years, thresh)
                assign(da_pr, str(dstr(y1 + 1, 12, 18)), str(dstr(y1 + 1, 12, 31)), 0)
                res_expected = [0, 14]

            # Case #21: | T 3x<T T 2x<T T 3x<T T | T 3x<T T 2x<T T 3x<T T |
            elif (i == 21) and (op == op_max):
                thresh, window = 3, 3
                da_pr = gen_scen(var, y1, n_years, thresh)
                pr = thresh / window
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 3)), pr)
                assign(da_pr, str(dstr(y1, 6, 1)), str(dstr(y1, 6, 2)), pr)
                assign(da_pr, str(dstr(y1, 9, 1)), str(dstr(y1, 9, 3)), pr)
                assign(da_pr, str(dstr(y2, 3, 1)), str(dstr(y2, 3, 3)), pr)
                assign(da_pr, str(dstr(y2, 6, 1)), str(dstr(y2, 6, 2)), pr)
                assign(da_pr, str(dstr(y2, 9, 1)), str(dstr(y2, 9, 3)), pr)
                res_expected = [6, 6]

            # {op} = "sum" ---------------------------------------------------------------------------------------------

            # Case #22: | . | . |
            elif (i == 22) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                res_expected = [365, 365]

            # Case #23: | . T*13/14 . | . |
            elif (i == 23) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), thresh - 1.0)
                res_expected = [365, 365]

            # Case #24: | . T . | . |
            elif (i == 24) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), thresh)
                res_expected = [364, 365]

            # Case #25: | . T/2x3 . | . |
            elif (i == 25) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 3)), thresh / 2.0)
                res_expected = [364, 365]

            # Case #26: | . 14xT/14 . | . |
            elif (i == 26) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), thresh / window)
                res_expected = [365, 365]

            # Case #27: | . 7xT/14 | 7xT/14 . |
            elif (i == 27) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 12, 25)), str(dstr(y2, 1, 7)), thresh / window)
                res_expected = [365, 365]

            # Case #28: | . 7xT | 7xT . |
            elif (i == 28) and (op == op_sum):
                thresh, window = 14, 14
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 12, 25)), str(dstr(y2, 1, 7)), thresh)
                res_expected = [358, 358]

            # Case #29: | . 15xT/14 . | . |
            elif (i == 29) and (op == op_sum):
                thresh, window = 15, 15
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 15)), thresh / window * 3)
                res_expected = [358, 365]

            # Case #30: | . 9xT/3 . | . |
            elif (i == 30) and (op == op_sum):
                thresh, window = 3, 3
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 9)), thresh / window)
                res_expected = [360, 365]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Convert from precipitation amount to rate.
            da_pr = da_pr / cfg.spd
            da_pr.attrs[cfg.attrs_units] = cfg.unit_kg_m2s1

            # Exit if case does not apply.
            if (algo == 1) and ((start_date != "") or (end_date != "")) and\
               (i not in [1, 2, 3, 6, 13, 14, 15, 18, 19, 20, 21]):
                continue

            # Calculate indices using the old algorithm.
            if algo == 1:
                pram = rate2amount(da_pr, out_units="mm")
                thresh = convert_units_to(str(thresh) + " mm", pram)
                if op == op_max:
                    mask = xr.DataArray(da_pr.rolling(time=window, center=True).max() < thresh)
                else:
                    mask = xr.DataArray(da_pr.rolling(time=window, center=True).sum() < thresh)
                out = (mask.rolling(time=window, center=True).sum() >= 1).resample(time=freq).sum()
                da_idx = to_agg_units(out, pram, "count")
            # Calculate indices using the new algorithm.
            else:
                da_idx = xindices.dry_spell_total_length(da_pr, str(thresh) + " mm", window, op, fill_value, freq,
                                                         start_date, end_date)

            # Extract results.
            res = [int(da_idx[0])]
            if len(da_idx) > 1:
                res.append(int(da_idx[1]))

            #  Raise error flag.
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

    # Parameters.
    thresh_wet = 15  # Tw
    window_wet = 3
    thresh_dry = 1   # Td
    window_dry = 10
    window_tot = 30

    # Loop through cases.
    error = False
    n_cases = 9
    for i in range(1, n_cases + 1):

        # Cases --------------------------------------------------------------------------------------------------------

        # Case #1: | . A . 3xTw/3 30xTd . B | . |
        if i == 1:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 5, 3)), thresh_dry)
            res_expected = [91, np.nan]

        # Case #2: | . A . 3xTw/3 10xTd 9x. 11xTd . B | . |
        elif i == 2:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 13)), thresh_dry)
            assign(da_pr, str(dstr(y1, 4, 23)), str(dstr(y1, 5, 3)), thresh_dry)
            res_expected = [91, np.nan]

        # Case #3: | . A . 3xTw/3 10xTd 10x. 10xTd . B | . |
        elif i == 3:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 13)), thresh_dry)
            assign(da_pr, str(dstr(y1, 4, 24)), str(dstr(y1, 5, 3)), thresh_dry)
            res_expected = [np.nan, np.nan]

        # Case #4: | . A . 2xTw*2/3 . 10xTd 9x. 11xTd . B | . |
        elif i == 4:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 2)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 13)), thresh_dry)
            assign(da_pr, str(dstr(y1, 4, 23)), str(dstr(y1, 5, 3)), thresh_dry)
            res_expected = [np.nan, np.nan]

        # Case #5: | . A . 3xTw/3 25xTd . B | . |
        elif i == 5:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 28)), thresh_dry)
            res_expected = [91, np.nan]

        # Case #6: | . A . 3xTw/3 5xTd 5x. 10xTd 5x. 5xTd . B | . |
        elif i == 6:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 8)), thresh_dry)
            assign(da_pr, str(dstr(y1, 4, 14)), str(dstr(y1, 4, 23)), thresh_dry)
            assign(da_pr, str(dstr(y1, 4, 29)), str(dstr(y1, 5, 3)), thresh_dry)
            res_expected = [91, np.nan]

        # Case #7: | . A . 2xTw/3 . 30xTd . 3xTw/3 30xTd . B | . |
        elif i == 7:
            start_date, end_date = "03-01", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 2)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 5, 3)), thresh_dry)
            assign(da_pr, str(dstr(y1, 6, 1)), str(dstr(y1, 6, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 6, 4)), str(dstr(y1, 7, 3)), thresh_dry)
            res_expected = [152, np.nan]

        # Case #8: | . 3xTw/3 A 30xTd . B | . |
        elif i == 8:
            start_date, end_date = "04-04", "12-31"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 5, 3)), thresh_dry)
            res_expected = [np.nan, np.nan]

        # Case #9: | . A . 3xTw/3 30xTd . | . B . |
        elif i == 9:
            start_date, end_date = "09-01", "06-30"  # A, B
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 10, 3)), thresh_wet / window_wet)
            assign(da_pr, str(dstr(y1, 10, 4)), str(dstr(y1, 11, 2)), thresh_dry)
            res_expected = [274, np.nan]

        else:
            continue

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Calculate index.
        da_start = indices.rain_season_start(da_pr, thresh_wet, window_wet, thresh_dry, window_dry, window_tot,
                                             start_date, end_date)
        # Verify results.
        res = [float(da_start[0])]
        if len(da_start) > 1:
            res.append(float(da_start[1]))
        if not res_is_valid(res, res_expected):
            error = True

    return error


def rain_season_end() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    TODO: Test indices.rain_season_end.

    Returns
    -------
    bool
        True if all tests are successful.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable.
    var = cfg.var_cordex_pr

    # Methods.
    method_depletion = "depletion"
    method_event     = "event"
    method_cumul     = "cumul"

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Loop through cases.
    error = False
    n_cases = 19
    for i in range(1, n_cases + 1):

        for method in [method_depletion, method_event, method_cumul]:

            # Parameters.
            etp        = 5
            etp_i      = etp
            pr         = etp
            window     = 14
            if method == method_event:
                thresh = pr               # T
            elif method == method_cumul:
                thresh = pr * window      # T
            else:
                thresh = etp * window     # T

            # {method} = any -------------------------------------------------------------------------------------------

            # Case #1: | . A . | . B . |
            if i == 1:
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                res_expected = [np.nan, np.nan]

            # Case #2: | T/14 A T/14 | T/14 B T/14 |
            elif i == 2:
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, pr)
                res_expected = [np.nan, np.nan]

            # {method} = "depletion" -----------------------------------------------------------------------------------

            # Case #3: | . A 30xT/14 . B | . |
            elif (i == 3) and (method == method_depletion):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr)
                res_expected = [287, np.nan]

            # Case #4: | . A 30xT/14 T/28 B | . |
            elif (i == 4) and (method == method_depletion):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr)
                assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 31)), pr / 2)
                res_expected = [301, np.nan]

            # Case #5: | . A 30xT/14 . B | . | (no etp)
            elif (i == 5) and (method == method_depletion):
                etp_i = 0.0
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr)
                res_expected = [np.nan, np.nan]

            # Case #6: # | . A 30xT/14 . B | . | (2 x etp)
            elif (i == 6) and (method == method_depletion):
                etp_i = 2 * etp
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr)
                res_expected = [280, np.nan]

            # Case #7: | . A 30xT/14 2x. 2xT/14 . B | . | (2 x etp)
            elif (i == 7) and (method == method_depletion):
                etp_i = 2 * etp
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr)
                assign(da_pr, str(dstr(y1, 10, 3)), str(dstr(y1, 10, 4)), 2 * pr)
                res_expected = [284, np.nan]

            # Case #8: | . A 15xT/14 | 15xT/14 . B . |
            elif (i == 8) and (method == method_depletion):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 16)), pr)
                assign(da_pr, str(dstr(y2, 1, 1)), str(dstr(y2, 1, 15)), pr)
                res_expected = [np.nan, 29]

            # {method} = "event" ---------------------------------------------------------------------------------------

            # Case #9: | . T A 30xT . B | . |
            elif (i == 9) and (method == method_event):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr)
                res_expected = [273, np.nan]

            # Case #10: | . T A . B | . |
            elif (i == 10) and (method == method_event):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 8, 31)), pr)
                res_expected = [np.nan, np.nan]

            # Case #11: | . T/2 A 30xT/2 . B | . |
            elif (i == 11) and (method == method_event):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr / 2)
                res_expected = [np.nan, np.nan]

            # Case #12: | . T A 15xT . B | . |
            elif (i == 12) and (method == method_event):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 10, 15)), pr)
                res_expected = [288, np.nan]

            # Case #13: | . T A . B 15xT | . |
            elif (i == 13) and (method == method_event):
                etp_i = etp
                start_date, end_date = "09-01", "12-16"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 8, 31)), pr)
                assign(da_pr, str(dstr(y1, 12, 17)), str(dstr(y1, 12, 31)), pr)
                res_expected = [np.nan, np.nan]

            # Case #14: | . T A 24xT/14 7x. | 6x. T . B . |
            elif (i == 14) and (method == method_event):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 24)), pr)
                assign(da_pr, str(dstr(y2, 1, 7)), str(dstr(y2, 1, 7)), pr)
                res_expected = [np.nan, 7]

            # {method} = "cumul" ---------------------------------------------------------------------------------------

            # Case #15: | . T A 30xT . B | . |
            elif (i == 15) and (method == method_cumul):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr * window)
                res_expected = [273, np.nan]

            # Case #16: | . T A 30xT T . B | . |
            elif (i == 16) and (method == method_cumul):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr * window)
                assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 10, 1)), pr * window)
                res_expected = [274, np.nan]

            # Case #17: | . T A 30xT 7x. T . B | . |
            elif (i == 17) and (method == method_cumul):
                start_date, end_date = "09-01", "12-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), pr * window)
                assign(da_pr, str(dstr(y1, 10, 8)), str(dstr(y1, 10, 8)), pr * window)
                res_expected = [281, np.nan]

            # Case #18: | . T A 24xT 7x. | . B . |
            elif (i == 18) and (method == method_cumul):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 24)), pr * window)
                res_expected = [358, np.nan]

            # Case #19: | . T A 24xT/14 7x. | 6x. T . B . |
            elif (i == 19) and (method == method_cumul):
                start_date, end_date = "06-01", "03-31"  # A, B
                da_pr = gen_scen(var, y1, n_years, 0.0)
                assign(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 24)), pr * window)
                assign(da_pr, str(dstr(y2, 1, 7)), str(dstr(y2, 1, 7)), pr * window)
                res_expected = [np.nan, 7]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Calculate index.
            da_end =\
                indices.rain_season_end(da_pr, None, None, None, method, thresh, etp_i, window, start_date, end_date)

            # Verify results.
            res = [float(da_end[0])]
            if len(da_end) > 1:
                res.append(float(da_end[1]))
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

    # Precipitation dataset.
    da_pr = gen_scen(var, y1, n_years, 1.0)

    # Loop through cases.
    error = False
    n_cases = 5
    for i in range(1, n_cases + 1):

        # Initialization.
        da_start = None
        da_end = None
        res_expected = [0] * n_years

        # Cases --------------------------------------------------------------------------------------------------------

        # Case #1: | . A1 . B1 . | . A2 . B2 . |
        if i == 1:
            da_start = gen_idx(cfg.idx_rain_season_start, y1, n_years, np.nan)
            for j in range(n_years):
                assign(da_start, str(y1 + j), str(y1 + j), 91 + j)
            da_end = gen_idx(cfg.idx_rain_season_end, y1, n_years, np.nan)
            for j in range(n_years):
                assign(da_end, str(y1 + j), str(y1 + j), 273 - j)
            res_expected = [273 - 91 + 1, 272 - 92 + 1]

        # Case #2: | . A1 . | . B1 . |
        elif i == 2:
            da_start = gen_idx(cfg.idx_rain_season_start, y1, n_years, np.nan)
            assign(da_start, str(y1), str(y1), 273)
            da_end = gen_idx(cfg.idx_rain_season_end, y1, n_years, np.nan)
            assign(da_end, str(y2), str(y2), 91)
            res_expected = [365 - 273 + 1 + 91, np.nan]

        # Case #3: | . B0 . A1 . | . B1 . A2 . |
        elif i == 3:
            da_start = gen_idx(cfg.idx_rain_season_start, y1, n_years, np.nan)
            for y in [y1, y2]:
                assign(da_start, str(y), str(y), 273)
            da_end = gen_idx(cfg.idx_rain_season_end, y1, n_years, np.nan)
            for y in [y1, y2]:
                assign(da_end, str(y), str(y), 91)
            res_expected = [365 - 273 + 1 + 91, np.nan]

        # Case #4: | . B1 . | . A2 . |
        elif i == 4:
            da_start = gen_idx(cfg.idx_rain_season_start, y1, n_years, np.nan)
            assign(da_start, str(y2), str(y2), 91)
            da_end = gen_idx(cfg.idx_rain_season_end, y1, n_years, np.nan)
            assign(da_end, str(y1), str(y1), 273)
            res_expected = [np.nan, np.nan]

        # Case #5: | . B1 A1 . | . |
        elif i == 5:
            da_start = gen_idx(cfg.idx_rain_season_start, y1, n_years, np.nan)
            assign(da_start, str(y1), str(y1), 91)
            da_end = gen_idx(cfg.idx_rain_season_end, y1, n_years, np.nan)
            assign(da_end, str(y1), str(y1), 90)
            res_expected = [np.nan, np.nan]

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Calculate indices.
        da_length = indices.rain_season_length(da_start, da_end)
        da_prcptot = indices.rain_season_prcptot(da_pr, da_start, da_end)

        # Verify results.
        res_length = [float(da_length[0])]
        res_prcptot = [float(da_prcptot[0])]
        if len(da_length) > 1:
            res_length.append(float(da_length[1]))
            res_prcptot.append(float(da_prcptot[1]))
        if (not res_is_valid(res_length, res_expected)) or (not res_is_valid(res_prcptot, res_expected)):
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
    method_depletion = "depletion"
    method_event     = "event"
    method_cumul     = "cumul"

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Loop through cases.
    error = False
    n_cases = 2
    for i in range(1, n_cases + 1):

        # Initialization.
        da_etp = None
        da_start_next = None

        # Parameters: rain season start.
        s_thresh_wet = 15  # Tw
        s_window_wet = 3
        s_thresh_dry = 1   # Td
        s_window_dry = 10
        s_window_tot = 30

        # Parameters: rain season end.
        e_method = "cumul"
        e_etp    = 5
        e_pr     = s_thresh_dry
        e_window = 14
        if e_method == method_event:
            e_thresh = e_pr              # T
        elif e_method == method_cumul:
            e_thresh = e_pr * e_window   # T
        else:
            e_thresh = e_etp * e_window  # T

        # Cases --------------------------------------------------------------------------------------------------------

        # Case #1-2: start      = | . A . 3xTw/3 30xTd .             B | . |
        #            end        = | .                  T/14 C T/14 . D | . |
        #            start_next = | .                       .      E . | . | (i == 2)
        if i in [1, 2]:
            # Parameters and data.
            s_start_date, s_end_date = "03-01", "12-31"  # A, B
            e_start_date, e_end_date = "09-01", "12-31"  # C, D
            da_pr = gen_scen(var, y1, n_years, 0.0)
            assign(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)), s_thresh_wet / s_window_wet)
            assign(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 9, 30)), s_thresh_dry)
            if i == 2:
                da_start_next = gen_idx(cfg.idx_rain_season_start, y1, n_years, np.nan)
                assign(da_start_next, str(y1), str(y1), utils.doy_str_to_doy("09-05"))
            # Expected results.
            res_start_expected = [91, np.nan]
            if i == 1:
                res_end_expected = [260, np.nan]
            else:
                res_end_expected = [da_start_next[0] - 1, np.nan]
            res_length_expected = [res_end_expected[0] - res_start_expected[0] + 1, np.nan]
            res_prcptot_expected = copy.deepcopy(res_length_expected)
            res_prcptot_expected[0] += s_thresh_wet - s_window_wet

        else:
            continue

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Calculate indices.
        da_start, da_end, da_length, da_prcptot =\
            indices.rain_season(da_pr, da_etp, da_start_next, s_thresh_wet, s_window_wet, s_thresh_dry, s_window_dry,
                                s_window_tot, s_start_date, s_end_date, e_method, e_thresh, e_etp, e_window,
                                e_start_date, e_end_date)

        #  Verify results.
        res_start = [float(da_start[0])]
        res_end = [float(da_end[0])]
        res_length = [float(da_length[0])]
        res_prcptot = [float(da_prcptot[0])]
        if len(da_start) > 1:
            res_start.append(float(da_start[1]))
            res_end.append(float(da_end[1]))
            res_length.append(float(da_length[1]))
            res_prcptot.append(float(da_prcptot[1]))
        if (not res_is_valid(res_start, res_start_expected)) or \
           (not res_is_valid(res_end, res_end_expected)) or \
           (not res_is_valid(res_length, res_length_expected)) or \
           (not res_is_valid(res_prcptot, res_prcptot_expected)):
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
    # rain_season_start()

    utils.log("Step #0c  rain_season_end")
    # rain_season_end()

    utils.log("Step #0d  rain_season_length/prcptot")
    # rain_season_length_prcptot()

    utils.log("Step #0f  rain_season")
    # rain_season()