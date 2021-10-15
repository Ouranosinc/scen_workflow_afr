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
import datetime
import indices
import numpy as np
import pandas as pd
import utils
import xarray as xr
import xclim.indices as xindices


def create_da(
    var: str,
    start_year: int = 1981,
    n_years: int = 1,
    val: float = 0
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Create an empty data array.

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

    pr = [[[0] * 365 * n_years]]
    longitude = [0]
    latitude = [0]
    time = pd.date_range(str(start_year) + "-01-01", periods=365 * n_years)
    reference_time = pd.Timestamp("2021-09-28")

    # Variable.
    description = units = ""
    if var == cfg.var_cordex_pr:
        description = "Precipitation"
        units = cfg.unit_mm_d

    # Create data array.
    da = xr.DataArray(
        data=pr,
        dims=[cfg.dim_longitude, cfg.dim_latitude, cfg.dim_time],
        coords=dict(
            longitude=([cfg.dim_longitude], longitude),
            latitude=([cfg.dim_latitude], latitude),
            time=time,
            reference_time=reference_time,
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


def assign_values(
    da: xr.DataArray,
    date_start: str,
    date_end: str,
    val: float
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate rain.

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

    year_1, doy_1 = extract_year_doy(date_start)
    year_n, doy_n = extract_year_doy(date_end)

    da_t = da[cfg.dim_time]
    t1 = (year_1 - int(da["time"].dt.year.min())) * 365 + (doy_1 - 1)
    tn = (year_n - int(da["time"].dt.year.min())) * 365 + (doy_n - 1)
    da.loc[slice(da_t[t1], da_t[tn])] = val

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

    # Methods:
    method_1d = "1d"
    method_cumul = "cumul"

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Loop through cases.
    error = False
    n_cases = 30
    for i in range(1, n_cases + 1):

        for method in [method_1d, method_cumul]:

            # {method} = "1d" ------------------------------------------------------------------------------------------

            # Case #1: | T A 14x. T B T | T |
            if (i == 1) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #2: | T A 13x. T B T | T |
            elif (i == 2) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 13)), 0)
                res_expected = [0, 0]

            # Case #3: | T A 7x. T 7x. T B T | T |
            elif (i == 3) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 7)), 0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 9)), str(dstr(y1, 3, 15)), 0)
                res_expected = [0, 0]

            # Case #4: | T . A 13x. T B T | T |
            elif (i == 4) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 2, 28)), str(dstr(y1, 3, 13)), 0)
                res_expected = [13, 0]

            # Case #5: | T A T 13x. B . T | T |
            elif (i == 5) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 11, 18)), str(dstr(y1, 12, 1)), 0)
                res_expected = [13, 0]

            # Case #6: | T A 14x. T 14x. B T | T |
            elif (i == 6) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                da_pr = assign_values(da_pr, str(dstr(y1, 8, 1)), str(dstr(y1, 8, 14)), 0)
                res_expected = [28, 0]

            # Case #7: | T B 14x. T A T | T |
            elif (i == 7) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [0, 0]

            # Case #8: | T B 13x. T A T | T |
            elif (i == 8) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 13)), 0)
                res_expected = [0, 0]

            # Case #9: | T B T 7x. T 7x. T A T | T |
            elif (i == 9) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 7)), 0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 9)), str(dstr(y1, 3, 15)), 0)
                res_expected = [0, 0]

            # Case #10: | T B 14x. T A T | T |
            elif (i == 10) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 2, 28)), str(dstr(y1, 3, 13)), 0)
                res_expected = [1, 0]

            # Case 11: | T B T 14x. A T | T |
            elif (i == 11) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 11, 18)), str(dstr(y1, 12, 1)), 0)
                res_expected = [1, 0]

            # Case #12: | T B T 14x. T 14x. T A T | T |
            elif (i == 12) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                da_pr = assign_values(da_pr, str(dstr(y1, 8, 1)), str(dstr(y1, 8, 14)), 0)
                res_expected = [0, 0]

            # Case #13: | T 14x. T | T |
            elif (i == 13) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #14: | T A 14x. T | T |
            elif (i == 14) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": ""}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #15: | T 14x. T B T | T |
            elif (i == 15) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": "", "end_date": dstr(-1, 12, 1)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
                res_expected = [14, 0]

            # Case #16: | T 7x. A 7x. T | T |
            elif (i == 16) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": ""}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 2, 22)), str(dstr(y1, 3, 7)), 0)
                res_expected = [7, 0]

            # Case #17: | T 7x. T B 7x. T | T |
            elif (i == 17) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": "", "end_date": dstr(-1, 12, 1)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 11, 24)), str(dstr(y1, 12, 7)), 0)
                res_expected = [8, 0]

            # Case #18: | T B T 7x. | 7x. T B T |
            elif (i == 18) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 1, 31)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 12, 25)), str(dstr(y1 + 1, 1, 7)), 0)
                res_expected = [7, 7]

            # Case #19: | 14x. T B T A T | T |
            elif (i == 19) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 1, 31)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1, 1, 1)), str(dstr(y1, 1, 14)), 0)
                res_expected = [14, 0]

            # Case #20: | T | T B T A T 14x. |
            elif (i == 20) and (method == method_1d):
                params = {"method": method, "thresh": 1.0, "window": 14,
                          "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 1, 31)}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                da_pr = assign_values(da_pr, str(dstr(y1 + 1, 12, 18)), str(dstr(y1 + 1, 12, 31)), 0)
                res_expected = [0, 14]

            # Case #21: | T 3x<T T 2x<T T 3x<T T | T 3x<T T 2x<T T 3x<T T |
            elif (i == 21) and (method == method_1d):
                params = {"method": method, "thresh": 3.0, "window": 3,
                          "dry_fill": True, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, params["thresh"])
                pr = params["thresh"] / params["window"]
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 3)), pr)
                da_pr = assign_values(da_pr, str(dstr(y1, 6, 1)), str(dstr(y1, 6, 2)), pr)
                da_pr = assign_values(da_pr, str(dstr(y1, 9, 1)), str(dstr(y1, 9, 3)), pr)
                da_pr = assign_values(da_pr, str(dstr(y2, 3, 1)), str(dstr(y2, 3, 3)), pr)
                da_pr = assign_values(da_pr, str(dstr(y2, 6, 1)), str(dstr(y2, 6, 2)), pr)
                da_pr = assign_values(da_pr, str(dstr(y2, 9, 1)), str(dstr(y2, 9, 3)), pr)
                res_expected = [6, 6]

            # {method} = "cumul" ---------------------------------------------------------------------------------------

            # Case #22: | . | . |
            elif (i == 22) and (method == method_cumul):
                params = {"method": method, "thresh": 14.0, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                res_expected = [365, 365]

            # Case #23: | . T*13/14 . | . |
            elif (i == 23) and (method == method_cumul):
                params = {"method": method, "thresh": 14.0, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), params["thresh"] - 1.0)
                res_expected = [365, 365]

            # Case #24: | . T . | . |
            elif (i == 24) and (method == method_cumul):
                params = {"method": method, "thresh": 14.0, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), params["thresh"])
                res_expected = [364, 365]

            # Case #25: | . T/2 T/2 . | . |
            elif (i == 25) and (method == method_cumul):
                params = {"method": method, "thresh": 14.0, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 2)), params["thresh"] / 2.0)
                res_expected = [363, 365]

            # Case #26: | . 14xT/14 . | . |
            elif (i == 26) and (method == method_cumul):
                params = {"method": method, "thresh": 14, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)),
                                      params["thresh"] / params["window"])
                res_expected = [351, 365]

            # Case #27: | . 7xT/14 | 7xT/14 . |
            elif (i == 27) and (method == method_cumul):
                params = {"method": method, "thresh": 14, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 12, 25)), str(dstr(y2, 1, 7)),
                                      params["thresh"] / params["window"])
                res_expected = [358, 358]

            # Case #28: | . 7xT | 7xT . |
            elif (i == 28) and (method == method_cumul):
                params = {"method": method, "thresh": 14.0, "window": 14,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 12, 25)), str(dstr(y2, 1, 7)), params["thresh"])
                res_expected = [358, 358]

            # Case #29: | . 15xT/14 . | . |
            elif (i == 29) and (method == method_cumul):
                params = {"method": method, "thresh": 15.0, "window": 15,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 15)),
                                      params["thresh"] / params["window"])
                res_expected = [350, 365]

            # Case #30: | . 9xT/14 . | . |
            elif (i == 30) and (method == method_cumul):
                params = {"method": method, "thresh": 3.0, "window": 3,
                          "dry_fill": False, "start_date": "", "end_date": ""}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 9)),
                                      params["thresh"] / params["window"])
                res_expected = [356, 365]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            # Exit if case does not apply.
            if (algo == 1) and ((params["start_date"] != "") or (params["end_date"] != "")):
                continue

            # Calculate indices using old algorithm.
            # This algorithm is not working properly:
            # - the 'rolling' function creates 'nan' values near boundaries; the <window size-1>/2 days from the
            #   beginning and end of the dataset are indirectly assumed to be wet (because they are not dry); it would
            #   be better to let the user specify if cells are wet or dry near boundaries;
            # - results are wrong if an even windows size is specified;
            # - dry days are not affected to the right year when a period overlaps two years.
            if algo == 1:
                da_idx = xindices.dry_spell_total_length(da_pr, str(params["thresh"]) + " mm", params["window"],
                                                         cfg.freq_YS)

            # Calculate indices using new algorithm.
            else:
                da_idx = indices.dry_spell_total_length(da_pr, params["method"], params["thresh"], params["window"],
                                                        params["dry_fill"], params["start_date"], params["end_date"])

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

    # Loop through cases.
    error = False
    n_cases = 9
    for i in range(1, n_cases + 1):

        # Cases --------------------------------------------------------------------------------------------------------

        # Case #1: | . A . 3xTw/3 30xTd . B | . |
        if i == 1:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            res_expected = [91, np.nan]

        # Case #2: | . A . 3xTw/3 10xTd 9x. 11xTd . B | . |
        elif i == 2:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 13)), params["thresh_dry"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 23)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            res_expected = [91, np.nan]

        # Case #3: | . A . 3xTw/3 10xTd 10x. 10xTd . B | . |
        elif i == 3:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 13)), params["thresh_dry"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 24)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            res_expected = [np.nan, np.nan]

        # Case #4: | . A . 2xTw*2/3 . 10xTd 9x. 11xTd . B | . |
        elif i == 4:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 2)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 13)), params["thresh_dry"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 23)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            res_expected = [np.nan, np.nan]

        # Case #5: | . A . 3xTw/3 25xTd . B | . |
        elif i == 5:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 28)), params["thresh_dry"])
            res_expected = [91, np.nan]

        # Case #6: | . A . 3xTw/3 5xTd 5x. 10xTd 5x. 5xTd . B | . |
        elif i == 6:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 4, 8)), params["thresh_dry"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 14)), str(dstr(y1, 4, 23)), params["thresh_dry"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 29)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            res_expected = [91, np.nan]

        # Case #7: | . A . 2xTw/3 . 30xTd . 3xTw/3 30xTd . B | . |
        elif i == 7:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "03-01", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 2)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            da_pr = assign_values(da_pr, str(dstr(y1, 6, 1)), str(dstr(y1, 6, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 6, 4)), str(dstr(y1, 7, 3)), params["thresh_dry"])
            res_expected = [152, np.nan]

        # Case #8: | . 3xTw/3 A 30xTd . B | . |
        elif i == 8:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "04-04", "end_date": "12-31", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 4, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 4, 4)), str(dstr(y1, 5, 3)), params["thresh_dry"])
            res_expected = [np.nan, np.nan]

        # Case #9: | . A . 3xTw/3 30xTd . | . B . |
        elif i == 9:
            params = {"thresh_wet": 20, "window_wet": 3, "thresh_dry": 1, "window_dry": 10, "window_tot": 30,
                      "start_date": "09-01", "end_date": "06-30", }
            da_pr = create_da(var, y1, n_years, 0.0)
            da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 10, 3)),
                                  params["thresh_wet"] / params["window_wet"])
            da_pr = assign_values(da_pr, str(dstr(y1, 10, 4)), str(dstr(y1, 11, 2)), params["thresh_dry"])
            res_expected = [274, np.nan]

        else:
            continue

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Calculate indices.
        da_idx = indices.rain_season_start(da_pr, params["thresh_wet"], params["window_wet"],
                                           params["thresh_dry"], params["window_dry"], params["window_tot"],
                                           params["start_date"], params["end_date"])

        # Extract results.
        res = [float(da_idx[0])]
        if len(da_idx) > 1:
            res.append(float(da_idx[1]))

        #  Raise error flag.
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
    method_event = "event"
    method_cumul = "cumul"

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
            etp = 5.0
            window = 14
            thresh = etp * (1 if method == method_event else window)

            # {method} = any -------------------------------------------------------------------------------------------

            # Case #1: | . A . | . B . |
            if i == 1:
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                res_expected = [np.nan, np.nan]

            # Case #2: | T/14 A T/14 | T/14 B T/14 |
            elif i == 2:
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, etp)
                res_expected = [np.nan, np.nan]

            # {method} = "depletion" -----------------------------------------------------------------------------------

            # Case #3: | . A 30xT/14 . B | . |
            elif (i == 3) and (method == method_depletion):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp)
                res_expected = [287, np.nan]

            # Case #4: | . A 30xT/14 T/28 B | . |
            elif (i == 4) and (method == method_depletion):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 31)), etp / 2)
                res_expected = [301, np.nan]

            # Case #5: | . A 30xT/14 . B | . | (no etp)
            elif (i == 5) and (method == method_depletion):
                params = {"method": method, "thresh": thresh, "etp": 0.0, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp)
                res_expected = [np.nan, np.nan]

            # Case #6: # | . A 30xT/14 . B | . | (2 x etp)
            elif (i == 6) and (method == method_depletion):
                params = {"method": method, "thresh": thresh, "etp": 2 * etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp)
                res_expected = [280, np.nan]

            # Case #7: | . A 30xT/14 2x. 2xT/14 . B | . | (2 x etp)
            elif (i == 7) and (method == method_depletion):
                params = {"method": method, "thresh": thresh, "etp": 2 * etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 3)), str(dstr(y1, 10, 4)), 2 * etp)
                res_expected = [284, np.nan]

            # Case #8: | . A 15xT/14 | 15xT/14 . B . |
            elif (i == 8) and (method == method_depletion):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "06-01", "end_date": "03-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 16)), etp)
                da_pr = assign_values(da_pr, str(dstr(y2, 1, 1)), str(dstr(y2, 1, 15)), etp)
                res_expected = [np.nan, 29]

            # {method} = "event" ---------------------------------------------------------------------------------------

            # Case #9: | . T A 30xT . B | . |
            elif (i == 9) and (method == method_event):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp)
                res_expected = [273, np.nan]

            # Case #10: | . T A . B | . |
            elif (i == 10) and (method == method_event):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 8, 31)), etp)
                res_expected = [np.nan, np.nan]

            # Case #11: | . T/2 A 30xT/2 . B | . |
            elif (i == 11) and (method == method_event):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), etp / 2)
                res_expected = [np.nan, np.nan]

            # Case #12: | . T A 15xT . B | . |
            elif (i == 12) and (method == method_event):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 10, 15)), etp)
                res_expected = [288, np.nan]

            # Case #13: | . T A . B 15xT | . |
            elif (i == 13) and (method == method_event):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-16"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 8, 31)), etp)
                da_pr = assign_values(da_pr, str(dstr(y1, 12, 17)), str(dstr(y1, 12, 31)), etp)
                res_expected = [np.nan, np.nan]

            # Case #14: | . T A 24xT/14 7x. | 6x. T . B . |
            elif (i == 14) and (method == method_event):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "06-01", "end_date": "03-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 24)), thresh)
                da_pr = assign_values(da_pr, str(dstr(y2, 1, 7)), str(dstr(y2, 1, 7)), thresh)
                res_expected = [np.nan, 7]

            # {method} = "cumul" ---------------------------------------------------------------------------------------

            # Case #15: | . T A 30xT . B | . |
            elif (i == 15) and (method == method_cumul):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), thresh)
                res_expected = [273, np.nan]

            # Case #16: | . T A 30xT T . B | . |
            elif (i == 16) and (method == method_cumul):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), thresh)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 10, 1)), thresh)
                res_expected = [274, np.nan]

            # Case #17: | . T A 30xT 7x. T . B | . |
            elif (i == 17) and (method == method_cumul):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "09-01", "end_date": "12-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 4, 1)), str(dstr(y1, 9, 30)), thresh)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 8)), str(dstr(y1, 10, 8)), thresh)
                res_expected = [281, np.nan]

            # Case #18: | . T A 24xT 7x. | . B . |
            elif (i == 18) and (method == method_cumul):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "06-01", "end_date": "03-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 24)), thresh)
                res_expected = [358, np.nan]

            # Case #19: | . T A 24xT/14 7x. | 6x. T . B . |
            elif (i == 19) and (method == method_cumul):
                params = {"method": method, "thresh": thresh, "etp": etp, "window": window,
                          "start_date": "06-01", "end_date": "03-31"}
                da_pr = create_da(var, y1, n_years, 0.0)
                da_pr = assign_values(da_pr, str(dstr(y1, 10, 1)), str(dstr(y1, 12, 24)), thresh)
                da_pr = assign_values(da_pr, str(dstr(y2, 1, 7)), str(dstr(y2, 1, 7)), thresh)
                res_expected = [np.nan, 7]

            else:
                continue

            # Calculation and interpretation ---------------------------------------------------------------------------

            da_idx = indices.rain_season_end(da_pr, None, None, None, params["method"], params["thresh"],
                                             params["etp"], params["window"], params["start_date"], params["end_date"])

            # Extract results.
            res = [float(da_idx[0])]
            if len(da_idx) > 1:
                res.append(float(da_idx[1]))

            #  Raise error flag.
            if not res_is_valid(res, res_expected):
                error = True

    return error


def rain_season_length() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    TODO: Test indices.rain_season_length.

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

    # Loop through cases.
    error = False
    n_cases = 0
    for i in range(1, n_cases + 1):

        # Initialization.
        da_pr = None
        params = None
        res_expected = [0] * n_years

        da_idx = [0, 0]

        # Cases --------------------------------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Extract results.
        res = [float(da_idx[0])]
        if len(da_idx) > 1:
            res.append(float(da_idx[1]))

        #  Raise error flag.
        if not res_is_valid(res, res_expected):
            error = True

    return error


def rain_season_prcptot() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    TODO: Test indices.rain_season_prcptot.

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

    # Loop through cases.
    error = False
    n_cases = 0
    for i in range(1, n_cases + 1):

        # Initialization.
        da_pr = None
        params = None
        res_expected = [0] * n_years

        da_idx = [0, 0]

        # Cases --------------------------------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Extract results.
        res = [float(da_idx[0])]
        if len(da_idx) > 1:
            res.append(float(da_idx[1]))

        #  Raise error flag.
        if not res_is_valid(res, res_expected):
            error = True

    return error


def rain_season() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    TODO: Test indices.rain_season.

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

    # Loop through cases.
    error = False
    n_cases = 0
    for i in range(1, n_cases + 1):

        # Initialization.
        da_pr = None
        params = None
        res_expected = [0] * n_years

        da_idx = [0, 0]

        # Cases --------------------------------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Extract results.
        res = [float(da_idx[0])]
        if len(da_idx) > 1:
            res.append(float(da_idx[1]))

        #  Raise error flag.
        if not res_is_valid(res, res_expected):
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
    # dry_spell_total_length()

    utils.log("Step #0b  rain_season_start")
    # rain_season_start()

    utils.log("Step #0c  rain_season_end")
    rain_season_end()

    utils.log("Step #0d  rain_season_length")
    rain_season_length()

    utils.log("Step #0e  rain_season_prcptot")
    rain_season_prcptot()
