# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions for the evaluation of datasets.
#
# Contact information:
# 1. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020-2022 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import calendar
import datetime
import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, List, Optional, Tuple

# xclim libraries.
import xclim.core.missing as missing

# Workflow libraries.
import wf_file_utils as wfu
import wf_plot
from cl_constant import const as c


def load_multipart_dataset(
    p_l: List[str]
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine datasets.

    The data for a station can be in several datasets, which explains why stations are added one by one. There must be a
    way to imrove the code, but the 'xr.merge' function does not seem to have an option to do this.

    Parameters
    ----------
    p_l: List[str]
        Paths of NetCDF file containing data.
        Must have the following dimensions: [station, time] or [station_id, time].

    Returns
    -------
    xr.Dataset
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds = None

    # Station identifiers, names, longitudes and latitudes.
    stn_id_l, stn_name_l, lon_l, lat_l = [], [], [], []

    # Variable holding station identifier.
    stn_id_var = ""

    # Loop through files.
    for p in p_l:

        # Load dataset and rename/rearrange dimensions.
        ds_i = wfu.open_dataset(p)
        if "station_id" in list(ds_i.dims):
            ds_i = ds_i.rename({"station_id": c.DIM_STATION})
        if stn_id_var == "":
            stn_id_var = "station_id" if "station_id" in list(ds_i.variables) else c.DIM_STATION

        ds_i = ds_i.transpose(c.DIM_STATION, c.DIM_TIME)

        if ds is None:
            ds = ds_i
            stn_id_l = list(ds[stn_id_var].values)
            stn_name_l = list(ds["station_name"].values)
            lon_l = list(ds[c.DIM_LON].values)
            lat_l = list(ds[c.DIM_LAT].values)

        else:
            for j in range(len(ds_i[c.DIM_STATION])):
                ds_j = ds_i.isel(station=slice(j, j + 1))
                n_before = len(ds[c.DIM_STATION])
                ds = xr.Dataset.merge(ds, ds_j, compat="override")
                n_after = len(ds[c.DIM_STATION])
                if n_before != n_after:
                    stn_id_l = stn_id_l + list(ds_j[stn_id_var].values)
                    stn_name_l = stn_name_l + list(ds_j["station_name"].values)
                    lon_l = lon_l + list(ds_j[c.DIM_LON].values)
                    lat_l = lat_l + list(ds_j[c.DIM_LAT].values)

    # Put station information back into the dataset.
    ds[stn_id_var] = stn_id_l
    ds["station_name"][:] = stn_name_l
    ds[c.DIM_LON][:] = lon_l
    ds[c.DIM_LAT][:] = lat_l

    return ds


def evaluate_single_analysis(
    method_id: str,
    ds: xr.Dataset,
    var_name: str,
    per: List[int] = None,
    month_l: Union[List[int], None] = None,
    n_years_req: int = 1,
    nm: Optional[int] = 11,
    nc: Optional[int] = 5,
    pct: Optional[float] = 0.35
) -> pd.DataFrame:

    """
    --------------------------------------------------------------------------------------------------------------------
    Evaluate a dataset of observations according to WMO criteria.

    The WMO criterion is the following: a result is missing if 11 days are missing, or if 5 consecutive values are
    missing in a month.

    The algorithm is compatible at least with CanSWE and MELCC datasets.

    Parameters
    ----------
    method_id: str
        Evaluation method = {"wmo", "pct"}:
        "wmo": At least X year(s) in which less than Y missing values/month are missing (with less than Z consecutive
               values/month missing), considering specific months (WMO criteria).
               Must provide values for 'nm' and 'nc'.
        "pct": At least X year(s) in which less than Y% of values/month are missing, in a given period.
               Must provide a value for 'nm' and 'nc'.
    ds: xr.Dataset
        Dataset with the following dimensions: [station, time] or [station_id, time].
    var_name: str
        Variable name.
    per: List[int]
        Period of interest (first and last year).
    month_l: Union[List[int], None]
        List of months for which data must be availble.
    n_years_req: int
        Number of years for which data must be available at a station.
    nm: Optional[int]
        Number of days without data in a period that results in the exlucsion of this period.
    nc: Optional[int]
        Number of consecutive days without data in a period that results in the exclusion of this period.
    pct: Optional[float]
        Fraction of missing values that are tolerated [0,1].

    Returns
    -------
    pd.DataFrame
        Dataframe with the following columns:
        - station identifier (stn_id)
        - station name (stn_name)
        - longitude
        - latitude
        - number of years with data (n_years_w_data)
        - year 1 (yyyy) : 0 = not enough data available ; 1 = enough data available
        - year 2 (yyyy)
        - ...
        - year n (yyyy)
    --------------------------------------------------------------------------------------------------------------------
    """

    msg = "Processing : var_name=" + var_name
    if (per is not None) and (len(per) >= 2):
        msg += ", per=" + str(per)
    if (month_l is not None) and (len(month_l) >= 1):
        msg += ", month_l=" + str(month_l)
    msg += ", n_years_req=" + str(n_years_req)
    if method_id == "wmo":
        msg += ", nm=" + str(nm) + ", nc=" + str(nc)
    else:
        msg += ", pct=" + str(pct*100.0) + "%"
    wfu.log(msg, True)

    # Consider all months if they were not specified.
    if month_l is None:
        month_l = list(range(1, 13))

    # Structure that will hold station information.
    stn_name_l, lon_l, lat_l, n_years_w_data_l = [], [], [], []

    # Extract DataArray.
    da = ds[var_name]

    # Variable holding station identifier.
    stn_id_var = "station_id" if "station_id" in list(ds.variables) else c.DIM_STATION

    # Extract station identifiers, names, longitudes and latitudes.
    stn_id_all_l = list(ds[stn_id_var].values)
    stn_name_all_l = list(ds["station_name"].values)
    lon_all_l = list(ds[c.DIM_LON].values)
    lat_all_l = list(ds[c.DIM_LAT].values)

    # Tells whether data for the months of interest are missing/available.
    # Patch: months that are of no interest are assumed to have missing values so that only the month of interest can be
    #        marked as having available data. This seems to be essential for the 'pct' method.
    if method_id == "wmo":
        da_missing = missing.missing_wmo(da, freq=c.FREQ_MS, nm=nm, nc=nc, src_timestep="D", month=month_l)
    else:
        da_missing = missing.missing_pct(da, freq=c.FREQ_MS, tolerance=pct, src_timestep="D", month=month_l)
    for m in range(1, 13):
        if m not in month_l:
            da_missing = xr.where(da_missing.time.dt.month == m, True, da_missing)
    da_available = da_missing.astype(int) == 0

    # Select years.
    if (per is not None) and (len(per) >= 2):
        da_available = da_available.sel(time=np.logical_and(da_available.time.dt.year >= per[0],
                                                            da_available.time.dt.year <= per[1]))

    # Calculate the number of months with data for each station-year.
    da_stn_yr = da_available.astype(int).resample(time=c.FREQ_YS).sum(dim=c.DIM_TIME)

    # Calculate the number of years with data at each station.
    da_stn = xr.where(da_stn_yr < len(month_l), 0, 1)

    # List years.
    year_l = list(set(list(da_stn[c.DIM_TIME].dt.year.values)))
    year_l.sort()

    # Collect relevant information about stations.
    dict_pd = {"year": year_l}
    df = pd.DataFrame(dict_pd)
    for i in list(range(len(stn_id_all_l))):

        # Select station.
        stn_id = stn_id_all_l[i]
        da_stn_i = da_stn.isel(station=slice(i, i + 1)).squeeze()

        # Calculate the number of years with data.
        n_years_w_data = int(da_stn_i.sum(dim=c.DIM_TIME).values)

        # Add current result as a column in the DataFrame.
        if stn_id not in df.columns:
            df[stn_id] = da_stn_i.values
            stn_name_l.append(stn_name_all_l[i])
            lon_l.append(lon_all_l[i])
            lat_l.append(lat_all_l[i])
            n_years_w_data_l.append(n_years_w_data)
        else:
            df[stn_id] += da_stn_i.values
            n_years_w_data_l[stn_id_all_l.index(stn_id)] += n_years_w_data

    # Transpose.
    df_t = df.set_index("year")
    df_t = df_t.transpose().rename_axis("stn_id").reset_index()

    # Add station names and coordinates.
    df_t.insert(loc=1, column="stn_name", value=stn_name_l)
    df_t.insert(loc=2, column=c.DIM_LONGITUDE, value=lon_l)
    df_t.insert(loc=3, column=c.DIM_LATITUDE, value=lat_l)
    df_t.insert(loc=4, column="n_years_w_data", value=n_years_w_data_l)
    df_t = df_t.sort_values(by="stn_id")

    # Select records with enough years.
    df_t = df_t[df_t["n_years_w_data"] >= n_years_req]

    return df_t


def evaluate_sensitivity_analysis(
    method_id: str,
    ds: xr.Dataset,
    df_poi: Union[pd.DataFrame, None],
    ens: str,
    per: Union[List[int], None] = None,
    stn_key_l: Union[List[str], None] = None,
    p_bounds: Optional[str] = "",
    p_csv_template: Optional[str] = "",
    opt_overwrite: Optional[bool] = False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Evaluate the availability of data using a sensitivity analysis.

    Parameters
    ----------
    method_id: str
        Evaluation method = {"wmo", "pct"}:
        "wmo": At least X year(s) in which less than Y missing values/month are missing (with less than Z consecutive
               values/month missing), considering specific months (WMO criteria).
        "pct": At least X year(s) in which less than Y% of values/month are missing, in a given period.
    ds: xr.Dataset
        Dataset with the following dimensions: [station, time] or [station_id, time].
    df_poi: Union[pd.DataFrame, None]
        DataFrame containing the points of interest.
    ens: str
        Ensemble = {c.ENS_CANSWE, c.ENS_MELCC}.
    per: Union[List[int], None]
        Period of interest (first and last year).
    stn_key_l: Union[List[str], None]
        List of keywords used to identifying stations for which statistics are required. At least one keyword must be
        present in station name.
    p_bounds: Optional[str]
        Path of a GEOJSON file containing polygons.
    p_csv_template: Optional[str]
        Template of output CSV (only missing a suffix).
    opt_overwrite: Optional[bool]
        Option that tells whether the algorithm should overwrite existing files.
    --------------------------------------------------------------------------------------------------------------------
    """

    wfu.log("=")
    wfu.log("Evaluating " + ens + " using '" + method_id + "' method (sensitivity analysis).", True)

    # Structure to save into.
    content = []

    # Initialize search key if it was not provided.
    if stn_key_l is None:
        stn_key_l = ["total"]

    # Initialize period if it was not provided.
    if per is None:
        year_l = list(set(list(ds[c.DIM_TIME].dt.year.values)))
        if per is None:
            per = [min(year_l), max(year_l)]
        else:
            per = [min(year_l + per), max(year_l + per)]

    # Convert month identifiers into a string (for display).
    def months_to_str(
        month_l: List[int]
    ) -> str:
        return "".join(("" if i == 0 else "-") + str(month_l[i]) for i in range(len(month_l)))

    # Convert a period into a string (for display).
    def per_to_str(
        per: List[int]
    ) -> str:
        return str(per[0]) + "-" + str(per[1]) if (per is not None) and (len(per) >= 2) else ""

    # Determine path of output CSV and PNG.
    def p_csv_png(
        var_name: str,
        suffix: str
    ) -> Tuple[str, str]:
        p_csv = p_csv_template.\
            replace("analysis/", "analysis/" + var_name + "_" + method_id + "/").replace("<suffix>", suffix)
        p_png = p_csv.replace(c.F_EXT_CSV, c.F_EXT_PNG)
        return p_csv, p_png

    # Calculate de the number corresponding to each key.
    def stn_count(
        df_count: pd.DataFrame
    ) -> List[int]:
        n_stn_l = []
        for stn_key in stn_key_l:
            n_stn = 0
            if stn_key == "total":
                n_stn = len(df_count)
            else:
                for stn_id in df_count["stn_id"]:
                    if stn_key in stn_id:
                        n_stn += 1
            n_stn_l.append(n_stn)
        return n_stn_l

    # Variables.
    var_name_l = c.V_CANSWE if ens == c.ENS_CANSWE else c.V_MELCC
    for var_name in var_name_l:

        if method_id == "wmo":

            # Parameters.
            per_l         = [per]
            n_years_req_l = [1]
            month_l_l     = [[1, 2, 3, 12]]
            nm_l          = [11]
            nc_l          = [5, 11]

            # Combinations.
            for n_years_req in n_years_req_l:
                for nm in nm_l:
                    for nc in nc_l:
                        for per in per_l:
                            for month_l in month_l_l:

                                # Paths.
                                suffix = "_" + ("nan" if n_years_req < 0 else str(n_years_req)) + \
                                         "_" + ("nan" if nm < 0 else str(round(nm, 2))) + \
                                         "-" + ("nan" if nc < 0 else str(round(nc, 2))) + \
                                         "_" + ("nan" if len(month_l) == 0 else months_to_str(month_l)) + \
                                         "_" + ("nan" if per is None else per_to_str(per))
                                p_csv, p_png = p_csv_png(var_name, suffix)

                                # Evaluate dataset.
                                if (not os.path.exists(p_csv)) or opt_overwrite:
                                    df_stn = evaluate_single_analysis(method_id=method_id, ds=ds, var_name=var_name,
                                                                      per=per, month_l=month_l, n_years_req=n_years_req,
                                                                      nm=nm, nc=nc)
                                    wfu.save_csv(df_stn, p_csv)
                                else:
                                    df_stn = pd.read_csv(p_csv)
                                content.append([n_years_req, [nm, nc], month_l, per] + list(stn_count(df_stn)))

                                # Generate map.
                                if (not os.path.exists(p_png)) or opt_overwrite:
                                    wf_plot.gen_map_stations(df_stn, df_poi, var_name, n_years_req=n_years_req,
                                                             param_per_l=[nm, nc], per=per, month_l=month_l,
                                                             method_id=method_id, p_bounds=p_bounds, p_fig=p_png)

        if method_id == "pct":

            # Parameters.
            per_l         = [per]
            n_years_req_l = [1, 5, 10, 15, 20]
            month_l_l     = [[1, 2], [1, 2, 12], [1, 2, 3], [1, 2, 3, 12], [1, 2, 3, 11, 12], [1, 2, 3, 4, 12],
                             [1, 2, 3, 4, 11, 12]]
            pct_l         = [0.99, 0.95, 0.90, 0.75, 0.50, 0.25]

            # Combinations.
            for n_years_req in n_years_req_l:
                for pct in pct_l:
                    for per in per_l:
                        for month_l in month_l_l:

                            # Paths.
                            suffix = "_" + ("nan" if n_years_req < 0 else str(n_years_req)) + \
                                     "_" + ("nan" if pct < 0 else str(round(pct, 2))) + \
                                     "_" + ("nan" if len(month_l) == 0 else months_to_str(month_l)) + \
                                     "_" + ("nan" if per is None else per_to_str(per))
                            p_csv, p_png = p_csv_png(var_name, suffix)

                            # Evaluate dataset.
                            if (not os.path.exists(p_csv)) or opt_overwrite:
                                df_stn = evaluate_single_analysis(method_id=method_id, ds=ds, var_name=var_name,
                                                                  per=per, month_l=month_l, n_years_req=n_years_req,
                                                                  pct=pct)
                                wfu.save_csv(df_stn, p_csv)
                            else:
                                df_stn = pd.read_csv(p_csv)
                            content.append([n_years_req, pct, month_l, per] + list(stn_count(df_stn)))

                            # Generate map.
                            if (not os.path.exists(p_png)) or opt_overwrite:
                                wf_plot.gen_map_stations(df_stn, df_poi, var_name, n_years_req=n_years_req,
                                                         param_per_l=[pct], per=per, month_l=month_l,
                                                         method_id=method_id, p_bounds=p_bounds, p_fig=p_png)

        # Export overall result to CSV.
        if len(content) > 0:

            # Transpose array.
            content_t = [[row[i] for row in content] for i in range(len(content[0]))]

            # Create DataFrame.
            df = pd.DataFrame()
            df["n_years_w_data"] = content_t[0]
            df["pct_vals"] = content_t[1]
            df["months"] = content_t[2]
            df["period"] = content_t[3]
            for i_stn_key in range(len(stn_key_l)):
                df["n_stations_" + stn_key_l[i_stn_key]] = content_t[4 + i_stn_key]

            # Path.
            p_csv = p_csv_png(var_name, "")[0]

            # Save CSV.
            wfu.save_csv(df, p_csv)
