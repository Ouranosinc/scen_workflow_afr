# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Unit tester.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2021 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import datetime
import indices
import pandas as pd
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

    # Years.
    n_years = 2
    y1 = 1981
    y2 = y1 + 1

    # Loop through cases.
    error = False
    n_cases = 30
    for i in range(1, n_cases + 1):

        # Initialization.
        da_pr = None
        params = None
        res_expected = [0] * n_years

        # Method "1d" --------------------------------------------------------------------------------------------------

        # Case #1: sequence of 14/14 days, completely between start/end dates.
        # | . A+14x . B . | . |
        if i == 1:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            res_expected = [14, 0]

        # Case #2: sequence of 13/14 days, completely between start/end dates.
        # | . A+13x . B . | . |
        elif i == 2:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 13)), 0)
            res_expected = [0, 0]

        # Case #3: 2 sequences of 7/14 days, completely between start/end dates.
        # | . A+7x . 7x . B . | . |
        elif i == 3:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 7)), 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 9)), str(dstr(y1, 3, 15)), 0)
            res_expected = [0, 0]

        # Case #4: sequence of 14/14 days, starting one day before start date, partially between start/end dates.
        # | . x A+13x . B . | . |
        elif i == 4:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 2, 28)), str(dstr(y1, 3, 13)), 0)
            res_expected = [13, 0]

        # Case #5: sequence of 14/14 days, ending one day after end date, partially between start/end dates.
        # | . A . 13x B x . | . |
        elif i == 5:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 11, 18)), str(dstr(y1, 12, 1)), 0)
            res_expected = [13, 0]

        # Case #6: 2 sequences of 14/14 days, between start/end dates, completely between start/end dates.
        # | . A+14x . 14x B . | . |
        elif i == 6:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": dstr(-1, 11, 30)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 8, 1)), str(dstr(y1, 8, 14)), 0)
            res_expected = [28, 0]

        # Case #7: sequence of 14/14 days, not between start/end dates.
        # | . B 14x . A . | . |
        elif i == 7:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            res_expected = [0, 0]

        # Case #8: sequence of 13/14 days, not between start/end dates.
        # | . B 13x . A . | . |
        elif i == 8:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 13)), 0)
            res_expected = [0, 0]

        # Case #9: 2 sequences of 7/14 days, not between start/end dates.
        # | . B . 7x . 7x . A . | . |
        elif i == 9:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 7)), 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 9)), str(dstr(y1, 3, 15)), 0)
            res_expected = [0, 0]

        # Case #10: sequence of 14/14 days, starting on end date, partially between start/end dates.
        # | . B+14x . A . | . |
        elif i == 10:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 2, 28)), str(dstr(y1, 3, 13)), 0)
            res_expected = [1, 0]

        # Case 11: sequence of 14/14 days, ending on start date, partially between start/end dates.
        # | . B . 14x+A . | . |
        elif i == 11:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 11, 18)), str(dstr(y1, 12, 1)), 0)
            res_expected = [1, 0]

        # Case #12: 2 sequences of 14/14 days, not between start/end dates.
        # | . B . 14x . 14x . A . | . |
        elif i == 12:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 2, 28)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 8, 1)), str(dstr(y1, 8, 14)), 0)
            res_expected = [0, 0]

        # Case #13: sequence of 14/14 days, unspecified start/end dates.
        # | . 14x . | . |
        elif i == 13:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            res_expected = [14, 0]

        # Case #14: sequence of 14/14 days, unspecified end date.
        # | . A+14x . | . |
        elif i == 14:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": ""}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            res_expected = [14, 0]

        # Case #15: sequence of 14/14 days, unspecified start date.
        # | . 14x . B . | . |
        elif i == 15:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": "", "end_date": dstr(-1, 12, 1)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), 0)
            res_expected = [14, 0]

        # Case #16: sequence of 14/14 days, unspecified end date, partially between start/end dates.
        # | . 7x A+x 6x . | . |
        elif i == 16:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 3, 1), "end_date": ""}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 2, 22)), str(dstr(y1, 3, 7)), 0)
            res_expected = [7, 0]

        # Case #17: sequence of 14/14 days, unspecified start date, partially between start/end dates.
        # | . 7x . B+x 6x . | . |
        elif i == 17:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": "", "end_date": dstr(-1, 12, 1)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 11, 24)), str(dstr(y1, 12, 7)), 0)
            res_expected = [8, 0]

        # Case #18: sequence of 14/14 days, overlapping between 2 years, between start/end dates.
        # | . B . 7x | A+x 6x . B . |
        elif i == 18:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 1, 31)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 12, 25)), str(dstr(y1 + 1, 1, 7)), 0)
            res_expected = [7, 7]

        # Case #19: sequence of 14/14 days, at the beginning of year 1, between start/end dates.
        # | 14x . B . A . | . |
        elif i == 19:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 1, 31)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 1, 1)), str(dstr(y1, 1, 14)), 0)
            res_expected = [14, 0]

        # Case #20: sequence of 14/14 days, at the end of year 2, between start/end dates.
        # | . | . B . A . 14x |
        elif i == 20:
            params = {"method": "1d", "thresh": 1.0, "window": 14,
                      "dry_fill": True, "start_date": dstr(-1, 12, 1), "end_date": dstr(-1, 1, 31)}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1 + 1, 12, 18)), str(dstr(y1 + 1, 12, 31)), 0)
            res_expected = [0, 14]

        # Case #21: 2 sequences of 3/3 days, 1 sequence of 2/3 days, per year, unspecified start/end dates.
        # | . 3x . 2x . 3x . | . 3x . 2x . 3x . |
        elif i == 21:
            params = {"method": "1d", "thresh": 3.0, "window": 3,
                      "dry_fill": True, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, params["thresh"])
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 3)), params["thresh"] / params["window"])
            da_pr = assign_values(da_pr, str(dstr(y1, 6, 1)), str(dstr(y1, 6, 2)), params["thresh"] / params["window"])
            da_pr = assign_values(da_pr, str(dstr(y1, 9, 1)), str(dstr(y1, 9, 3)), params["thresh"] / params["window"])
            da_pr = assign_values(da_pr, str(dstr(y2, 3, 1)), str(dstr(y2, 3, 3)), params["thresh"] / params["window"])
            da_pr = assign_values(da_pr, str(dstr(y2, 6, 1)), str(dstr(y2, 6, 2)), params["thresh"] / params["window"])
            da_pr = assign_values(da_pr, str(dstr(y2, 9, 1)), str(dstr(y2, 9, 3)), params["thresh"] / params["window"])
            res_expected = [6, 6]

        # Method "cumul" -----------------------------------------------------------------------------------------------

        # Case #22: no rain, unspecified start/end dates.
        # | . | . |
        elif i == 22:
            params = {"method": "cumul", "thresh": 14.0, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), 0.0)
            res_expected = [365, 365]

        # Case #23: 1 rainy day, below threshold, unspecified start/end dates.
        # | . 13 . | . |
        elif i == 23:
            params = {"method": "cumul", "thresh": 14.0, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), params["thresh"] - 1.0)
            res_expected = [365, 365]

        # Case #24: 1 rainy day, reaching threshold, unspecified start/end dates.
        # | . 14 . | . |
        elif i == 24:
            params = {"method": "cumul", "thresh": 14.0, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 1)), params["thresh"])
            res_expected = [364, 365]

        # Case #25: 2 rainy days, reaching threshold, unspecified start/end dates.
        # | . 7x2 . | . |
        elif i == 25:
            params = {"method": "cumul", "thresh": 14.0, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 2)), params["thresh"] / 2.0)
            res_expected = [363, 365]

        # Case #26: sequence of 14 rainy days, reaching threshold, unspecified start/end dates.
        # | . 14x1 . | . |
        elif i == 26:
            params = {"method": "cumul", "thresh": 14, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 14)), params["thresh"] / params["window"])
            res_expected = [351, 365]

        # Case #27: sequence of 14 rainy days, reaching threshold, overlapping 2 years, unspecified start/end dates.
        # | . 1x7 | 1x7 . |
        elif i == 27:
            params = {"method": "cumul", "thresh": 14, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 12, 25)), str(dstr(y2, 1, 7)),
                                  params["thresh"] / params["window"])
            res_expected = [358, 358]

        # Case #28: sequence of 14 rainy days, reaching threshold, overlapping 2 years, unspecified start/end dates.
        # dates.
        # | . 14x7 | 14x7 . |
        elif i == 28:
            params = {"method": "cumul", "thresh": 14.0, "window": 14,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 12, 25)), str(dstr(y2, 1, 7)), params["thresh"])
            res_expected = [358, 358]

        # Case #29: sequence of 15 rainy days, reaching threshold, unspecified start/end dates.
        # | . 1x15 . | . |
        elif i == 29:
            params = {"method": "cumul", "thresh": 15.0, "window": 15,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 15)), params["thresh"] / params["window"])
            res_expected = [350, 365]

        # Case #30: sequence of 9/3 days, unspecified start/end dates.
        # | . 9x . | . |
        elif i == 30:
            params = {"method": "cumul", "thresh": 3.0, "window": 3,
                      "dry_fill": False, "start_date": "", "end_date": ""}
            da_pr = create_da(var, y1, n_years, 0)
            da_pr = assign_values(da_pr, str(dstr(y1, 3, 1)), str(dstr(y1, 3, 9)), params["thresh"] / params["window"])
            res_expected = [356, 365]

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Exit if case does not apply.
        if (algo == 1) and ((params["start_date"] != "") or (params["end_date"] != "")):
            continue

        # Calculate using old algorithm.
        # This algorithm is not working properly:
        # - the 'rolling' function creates 'nan' values near boundaries; the <window size-1>/2 days from the beginning
        #   and end of the dataset are indirectly assumed to be wet (because they are not dry); it would be better to
        #   let the user specify if cells are wet or dry near boundaries;
        # - results are wrong if an even windows size is specified;
        # - dry days are not affected to the right year when a period overlaps two years.
        if algo == 1:
            da_idx = xindices.dry_spell_total_length(da_pr, str(params["thresh"]) + " mm", params["window"],
                                                     cfg.freq_YS)

        # Calculate using new algorithm.
        else:
            da_idx = indices.dry_spell_total_length(da_pr, params["method"], params["thresh"], params["window"],
                                                    params["dry_fill"], params["start_date"], params["end_date"])

        # Extract results.
        res = [int(da_idx[0])]
        if len(da_idx) > 1:
            res.append(int(da_idx[1]))

        #  Raise error flag.
        if res != res_expected:
            error = True

    return error


def rain_season_start() -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    TODO: Test indices.rain_season_start.

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

        # Cases (method = "depletion") ---------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Extract results.
        res = [int(da_idx[0])]
        if len(da_idx) > 1:
            res.append(int(da_idx[1]))

        #  Raise error flag.
        if res != res_expected:
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

        # Cases (method = "depletion") ---------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Cases (method = "event") -------------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Cases (method = "total") -------------------------------------------------------------------------------------

        # TODO: Insert cases.

        # Calculation and interpretation -------------------------------------------------------------------------------

        # Extract results.
        res = [int(da_idx[0])]
        if len(da_idx) > 1:
            res.append(int(da_idx[1]))

        #  Raise error flag.
        if res != res_expected:
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
        res = [int(da_idx[0])]
        if len(da_idx) > 1:
            res.append(int(da_idx[1]))

        #  Raise error flag.
        if res != res_expected:
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
        res = [int(da_idx[0])]
        if len(da_idx) > 1:
            res.append(int(da_idx[1]))

        #  Raise error flag.
        if res != res_expected:
            error = True

    return error


def run():

    dry_spell_total_length()
    rain_season_start()
    rain_season_end()
    rain_season_length()
    rain_season_prcptot()
