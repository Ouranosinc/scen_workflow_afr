# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions for file manipulation.
#
# Contact information:
# 1. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020-2022 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import altair as alt
import glob
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import xarray as xr
import xarray.core.variable as xcv
import warnings
from distutils import dir_util
from itertools import compress
from typing import Union, List, Optional, Any

# Workflow libraries.
import wf_utils
from cl_constant import const as c
from cl_context import cntx


def list_cordex(
    p_ds: str,
    rcps: [str],
    freq: Optional[str] = "day"
):

    """
    --------------------------------------------------------------------------------------------------------------------
     Lists CORDEX simulations.

    Parameters
    ----------
    p_ds: str
        Path of data_source.
    rcps: [str]
        List of RCP scenarios.
    freq: Optional[str]
        Frequency = {"day", "fx"}
    --------------------------------------------------------------------------------------------------------------------
    """

    list_f = {}

    # Find all the available simulations for a given RCP.
    for r in range(len(rcps)):

        d_format = p_ds + "*" + cntx.sep + "*" + cntx.sep + "AFR-*{r}".format(r=rcps[r]) + cntx.sep + "*" +\
                   cntx.sep + "atmos" + cntx.sep + "*" + cntx.sep
        d_l = glob.glob(d_format)
        d_l = [i for i in d_l if freq in i]
        d_l.sort()

        # Remove timestep information.
        for i in range(0, len(d_l)):
            tokens = d_l[i].split(cntx.sep)
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
    d_ds: str
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
    d_format = d_ds + "*" + cntx.sep + "*" + cntx.sep + "AFR-*" + cntx.sep + "day" + cntx.sep + "atmos" + cntx.sep +\
        "*" + cntx.sep
    n_token = len(d_ds.split(cntx.sep)) - 2

    # Loop through simulations sets.
    for i in glob.glob(d_format):
        tokens_i = i.split(cntx.sep)

        # Extract institute, RGM, CGM and emission scenario.
        inst = tokens_i[n_token + 1]
        rgm  = tokens_i[n_token + 2]
        cgm  = tokens_i[n_token + 3].split("_")[1]
        scen = tokens_i[n_token + 3].split("_")[2]

        # Extract variables and ensure that there is at least one NetCDF file available for each  one.
        vars_i = []
        for j in glob.glob(i + "*" + cntx.sep):
            n_netcdf = len(glob.glob(j + "*" + c.F_EXT_NC))
            if n_netcdf > 0:
                tokens_j = j.split(cntx.sep)
                var      = tokens_j[len(tokens_j) - 2]
                vars_i.append(var)

        sets.append([inst, rgm, cgm, scen, vars_i])

    return sets


def list_files(
    p: str
) -> [str]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Lists files in a directory.

    Parameters
    ----------
    p: str
        Path of directory.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List.
    p_l = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for p in f:
            if c.F_EXT_NC in p:
                p_l.append(os.path.join(r, p))

    # Sort.
    p_l.sort()

    return p_l


def create_mask(
    template: Optional[Union[str, xr.Dataset]] = ""
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Create a mask based on an existing layer for the current simulation, reanalysis data, or simulation data.

    The behaviour of the current function depends on the ensemble and climate variable/index.
    - A mask is not required if 'nan' values are present over the sea (ex: c.ENS_ENACTS, c.ENS_CHIRPS).
    - Sometimes, 'nan' values are given for a variable, but unavailable for another variable of the same ensemble.
    - The mask can be used if available (ex: ERA5 or CORDEX), or it can be derived from another variable
      (ex: ERA5-Land).
    - The order of attemps is related to resolution that can be achieved (ascending for resolution), hoping to get the
      finest details near the coasts.

    Parameters
    ----------
    template: Optional[Union[str, xr.Dataset]]
        Option #1: Path of NetCDF file used to build the mask.
        Option #2: Dataset used to build the mask.
        For both options, if the minimum and maximum values are identical (considering all time stepts), we can assume
        that values in these cells should be 'nan'.

    Notes
    -----
    The layer corresponding to c.V_SFTLF (CORDEX) contains values between 0 and 100, whereas
    the layer correspondong to c.V_ECMWF_LSM (ERA5) contains values between 0 and 1.

    Table. Values over sea in recognized ensembles:
    Ensemble         Variable       Value            Action
    -------------------------------------------------------------------------------------------------------------
    CORDEX           Temperature    value            Nothing
                     Precipitation  value            Nothing
    ECMWF.ERA5       Temperature    value            Apply mask using c.V_ECMWF_LSM or c.V_SFTLF.
                     Precipitation  value            Apply mask using c.V_ECMWF_LSM or c.V_SFTLF.
    ECMWF.ERA5-Land  Temperature    np.nan           Nothing
                     Precipitation  value            Apply mask based on the variable itself (min=max).
    ENACTS           Temperature    np.nan or value  Apply mask using c.V_SFTLF.
                     Precipitation  np.nan or value  Apply mask using c.V_SFTLF.
    CHIRPS           Precipitation  np.nan           Nothing
    -------------------------------------------------------------------------------------------------------------
    Any              Index          value            Apply mask based on the first variable used to compute this
                                                     index.

    Returns
    -------
    xr.DataArray
        Mask containing either 1 (over land) or np.nan (elsewhere).
    --------------------------------------------------------------------------------------------------------------------
    """

    da_mask = None

    # Apply a mask based on a climate scenario or index.
    if template != "":

        # Open NetCDF file.
        if isinstance(template, str):
            ds = open_netcdf(template)
        else:
            ds = template

        # Create mask.
        var_name = list(ds.data_vars)[0]
        da_min = ds[var_name].min(dim={c.DIM_TIME})
        da_max = ds[var_name].max(dim={c.DIM_TIME})
        da_nan = xr.DataArray(da_min == da_max).astype(int)
        da_mask = xr.where((da_nan == 1) | (da_min.isnull()), np.nan, 1)

    # Apply a mask based on ECMWF data (c.V_ECMWF_LSM or c.V_ECMWF_T2M).
    if (cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND]) and (da_mask is None):

        # Sometimes, the temperature variable sometimes include 'nan' values over the sea, whereas values for another
        # variable are provided and incorrect. This is why there is an attempt to consider any temperature variable.
        for var_name in [c.V_SFTLF, c.V_TAS]:

            # List NetCDF files.
            f_l = glob.glob(cntx.d_stn() + "*" + cntx.sep + var_name + "*" + c.F_EXT_NC)
            f_l.sort()
            if len(f_l) > 0:

                # Open NetCDF file.
                ds = open_netcdf(f_l[0])
                var_name = list(ds.data_vars)[0]

                # Create mask.
                if var_name == c.V_SFTLF:
                    da_mask = ds[var_name][0]
                else:
                    da_mask = ds[var_name][0] * 0 + 1
                da_mask = xr.where(da_mask == 0, np.nan, da_mask)

            if (da_mask is not None) or (cntx.obs_src == c.ENS_ERA5):
                break

    # Apply a mask based on CORDEX data (c.V_SFTLF).
    if ((cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND]) and (da_mask is None)) or\
       (cntx.obs_src == c.ENS_ENACTS):

        # List NetCDF files related to land area fraction.
        var_name = c.V_SFTLF
        f_l = glob.glob(cntx.d_scen(c.CAT_REGRID, var_name) + "*" + c.F_EXT_NC)
        f_l.sort()
        if len(f_l) > 0:

            # Open NetCDF file.
            ds = open_netcdf(f_l[0])

            # Create mask.
            da_mask = xr.DataArray(ds[var_name] > 0).astype(float)
            da_mask = xr.where(da_mask == 0, np.nan, da_mask)

    return da_mask


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
    p: Union[str, [str]]
        Path of file to be created.
    drop_variables: [str]
        Drop-variables parameter.
    chunks: Union[int,dict]
        Chunks parameter
    combine: str
        Combine parameter.
    concat_dim: str
        Concatenate dimension.
    desc: str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = (os.path.basename(p) if isinstance(p, str) else os.path.basename(p[0]))

    if cntx.opt_trace:
        log("Opening NetCDF file: " + desc, True)

    if isinstance(p, str):

        # Open file normally.
        ds = xr.open_dataset(p, drop_variables=drop_variables).load()
        close_netcdf(ds)

        # Determine the number of chunks.
        if cntx.use_chunks and (cntx.n_proc == 1) and (chunks is None) and ("scen" in p) and (c.DIM_TIME in ds.dims):
            chunks = {c.DIM_TIME: len(ds[c.DIM_TIME])}

        # Reopen file using chunks.
        if chunks is not None:
            ds = xr.open_dataset(p, drop_variables=drop_variables, chunks=chunks).copy(deep=True).load()
            close_netcdf(ds)
    else:
        ds = xr.open_mfdataset(p, drop_variables=drop_variables, chunks=chunks, combine=combine, concat_dim=concat_dim,
                               lock=False)

    if cntx.opt_trace:
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
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset.
    p: str
        Path of file to be created.
    desc: str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cntx.opt_trace:
        log("Saving NetCDF file: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.exists(d)):
        os.makedirs(d)

    # Create a temporary file to indicate that writing is in progress.
    p_inc = p.replace(c.F_EXT_NC, ".incomplete")
    if not os.path.exists(p_inc):
        open(p_inc, "a").close()

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save NetCDF file.
    ds.to_netcdf(p, mode="w", engine="netcdf4")

    # Discard the temporary file.
    if os.path.exists(p_inc):
        os.remove(p_inc)

    if cntx.opt_trace:
        log("Saved NetCDF file", True)


def close_netcdf(
    ds: Union[xr.Dataset, xr.DataArray]
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Close a NetCDF file.

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    if ds is not None:
        try:
            ds.close()
        finally:
            pass


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
    d: str
        Base directory to search from.
    --------------------------------------------------------------------------------------------------------------------
    """

    msg = "Cleaning NetCDF files: "
    if cntx.d_stn() in d:
        msg += "~" + cntx.sep + "stn" + cntx.sep + "..." + cntx.sep + d.replace(cntx.d_stn(), "")
    else:
        msg += "~" + cntx.sep + "res" + cntx.sep + "..." + cntx.sep + d.replace(cntx.d_res, "")
    msg += "*"
    log(msg, True)

    # List temporary files.
    if d[len(d) - 1] != cntx.sep:
        d = d + cntx.sep
    p_inc_l = glob.glob(d + "**" + cntx.sep + "*.incomplete", recursive=True)

    # Loop through temporary files.
    for p_inc in p_inc_l:

        # Attempt removing an associated NetCDF file.
        p_nc = p_inc.replace(".incomplete", c.F_EXT_NC)
        if os.path.exists(p_nc):
            log("Removing: " + p_nc)
            os.remove(p_nc)

        # Remove the temporary file.
        if os.path.exists(p_inc):
            log("Removing: " + p_inc)
            os.remove(p_inc)


def crop_netcdf(
    p_in: str,
    keyword_l: List[str],
    lon_l: List[float],
    lat_l: List[float],
    p_out: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Subset datasets based on a list of keywords and a box describing the range of longitudes and latitudes of interest.

    Parameters
    ----------
    p_in: str
        Path of input file.
    keyword_l: List[str]
        List of keywords.
    lon_l: List[float]
        List of longitudes.
    lat_l: List[float]
        List of latitudes.
    p_out: str
        Path of output file.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Try to find one of the keywords in the path.
    keyword_found = False
    for keyword in keyword_l:
        if keyword in p_in:
            keyword_found = True
            break

    # Skip iteration if the file to create already exists or if no keyword is comprised in the path.
    if ((os.path.exists(p_out)) and not cntx.opt_force_overwrite) or not keyword_found:
        return

    # Load NetCDF.
    # The except clause is necessary in the case there are two time dimensions (in CORDEX).
    try:
        ds = open_netcdf(p_in).load()
    except xcv.MissingDimensionsError:
        ds = open_netcdf(p_in, drop_variables=["time_bnds"]).load()

    # Get variable name.
    vi_name = ""
    for item in list(ds.variables):
        if (item in c.V_CORDEX) or (item in c.V_ECMWF) or (item in c.V_ENACTS) or (item in c.V_CHIRPS):
            vi_name = item
            break

    # Crop.
    ds_i_out = wf_utils.subset_lon_lat_time(ds, vi_name, lon_l, lat_l)

    # Save NetCDF.
    save_netcdf(ds_i_out, p_out)


def save_plot(
    plot: Union[alt.Chart, any, plt.Figure],
    p: str,
    desc: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a plot to a file.

    Parameters
    ----------
    plot: Union[alt.Chart, any, plt.Figure]
        Plot.
    p: str
        Path of file to be created.
    desc: str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cntx.opt_trace:
        log("Saving plot: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save file.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Library: matplot.
        if isinstance(plot, plt.Figure):
            plot.savefig(p)

        # Library: altair (not tested).
        elif isinstance(plot, alt.Chart):
            plot.save(p)

        # Library: hvplot (not working).
        # This requires:
        # - conda install selenium
        # - conda install -c conda-forge firefox geckodriver
        # else:
        #     from bokeh.io import export_png
        #     export_png(plot, p)

    if cntx.opt_trace:
        log("Saving plot", True)


def save_csv(
    df: pd.DataFrame,
    p: str,
    desc: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a dataset or a data array to a CSV file.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    p: str
        Path of file to be created.
    desc: Optional[str]
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cntx.opt_trace:
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

    if cntx.opt_trace:
        log("Saved CSV file", True)


def log(
    msg: str,
    indent=False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Log message to console and into a file.

    Parameters
    ----------
    msg: str
        Message.
    indent: bool
        If True, indent text.
    --------------------------------------------------------------------------------------------------------------------
    """

    ln = ""

    # Start line with a timestamp, unless this is a divide.
    if (msg != "-") and (msg != "="):
        ln = wf_utils.datetime_str()

    if indent:
        ln += " " * c.LOG_N_BLANK
    if (msg == "-") or (msg == "="):
        if indent:
            ln += msg * c.LOG_SEP_LEN
        else:
            ln += msg * (c.LOG_SEP_LEN + c.LOG_N_BLANK)
    else:
        ln += " " + msg

    # Print to console.
    pid_current = os.getpid()
    if pid_current == cntx.pid:
        print(ln)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(cntx.p_log)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Print to file.
    p_log = cntx.p_log
    if pid_current != cntx.pid:
        p_log = p_log.replace(c.F_EXT_LOG, "_" + str(pid_current) + c.F_EXT_LOG)
    if p_log != "":
        f = open(p_log, "a")
        f.writelines(ln + "\n")
        f.close()


def rename(
    p: str,
    text_to_modify: str,
    text_to_replace_with: str,
    rename_files: Optional[bool] = True,
    rename_directories: Optional[bool] = True,
    update_content: Optional[bool] = False,
    recursive: Optional[bool] = False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Rename files, directories and content under a path.

    This function is not used by the workflow. It is useful to batch rename files from a project if a new version of the
    code has a different structure.

    Parameters
    ----------
    p: str
        Path.
    text_to_modify: str
        Text to modify.
    text_to_replace_with: str
        Text to replace with.
    rename_files: Optional[bool]
        If True, rename files.
    rename_directories: Optional[bool]
        If True, rename directories.
    update_content: Optional[bool]
        If True, update file content.
    recursive: Optional[bool]
        If True, rename recursively (if 'p' is a directory).
    --------------------------------------------------------------------------------------------------------------------
    """

    # List files and directories.
    if os.path.isdir(p):
        p += cntx.sep if p[len(p) - 1] != cntx.sep else ""
        item_l = os.listdir(p)
    else:
        item_l = [p]
    item_l.sort()

    # Loop through items.
    for item_i in item_l:

        # Rename directory or file name.
        if (text_to_modify in item_i) and\
           (((not os.path.isdir(p + item_i)) and rename_files) or ((os.path.isdir(p + item_i)) and rename_directories)):
            shutil.move(p + item_i, p + item_i.replace(text_to_modify, text_to_replace_with))
            item_i = item_i.replace(text_to_modify, text_to_replace_with)

        # Update the content of a CSV file (and remove the index).
        if ((text_to_modify in item_i) or (text_to_replace_with in item_i)) and\
           (c.F_EXT_CSV in item_i) and update_content:

            df = pd.read_csv(p + item_i)
            columns: List[str] = list(df.columns)

            # View 'map': rename the column holding the value to 'val'.
            if cntx.sep + c.VIEW_MAP + cntx.sep in p:
                df.columns = [columns[i].replace(text_to_modify, "val").replace(text_to_replace_with, "val")
                              for i in range(len(columns))]

            # View 'ts': rename column names for lower, middle and upper values.
            elif ((cntx.sep + c.VIEW_TS + cntx.sep in p) or (cntx.sep + c.VIEW_TS_BIAS + cntx.sep in p)) and\
                 ("_rcp" in p):
                df.columns = [columns[i].replace("min", "lower").replace("mean", "middle").replace("moy", "middle")
                                        .replace("max", "upper") for i in range(len(columns))]

            # View 'tbl': rename column 'q' to 'centile', ajust values in columns 'stat' and 'centile'.
            elif cntx.sep + c.VIEW_TBL + cntx.sep in p:
                df.columns = [columns[i].replace("q", c.STAT_CENTILE) for i in range(len(columns))]
                def update_stat(i): return c.STAT_CENTILE if i == c.STAT_QUANTILE else i
                df["stat"] = df["stat"].map(update_stat)
                def update_centile(i): return i * 100 if (i > 0) and (i < 1) else i
                df["centile"] = df["centile"].map(update_centile)

            # Remove the index.
            if ("index" in df.columns[0]) or ("unnamed" in str(df.columns[0]).lower()):
                df = df.iloc[:, 1:]

            save_csv(df, p + item_i)

        # Rename items in a children directory.
        if os.path.isdir(p + item_i) and recursive:
            rename(p + item_i, text_to_modify, text_to_replace_with, rename_files, rename_directories,
                   update_content, recursive)


def migrate(
    platform: str,
    country_region: str,
    version_pre_migration: float,
    version_post_migration: float
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Migrate a project.

    Parameters
    ----------
    platform: str
        Platform = {c.PLATFORM_SCRIPT, c.PLATFORM_STREAMLIT, c.PLATFORM_JUPYTER}
    country_region: str
        Country and region ("<country>-<region>").
    version_pre_migration: float
        Version of workflow, to migrate from.
    version_post_migration: float
        Version of workflow, to migrate to.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Base path of the project.
    if platform == c.PLATFORM_SCRIPT:
        p = cntx.d_res
    else:
        p = os.getcwd() + cntx.sep + "dashboard" + cntx.sep + "data" + cntx.sep + country_region + cntx.sep

    if (version_pre_migration == 1.2) and (version_post_migration == 1.4):

        # Determine the path of the directory holding maps and time series.
        p_map, p_ts = p, p
        if platform == c.PLATFORM_SCRIPT:
            p_map += c.CAT_FIG + cntx.sep
            p_ts += c.CAT_FIG + cntx.sep
        p_map += c.VIEW_MAP + cntx.sep
        p_ts += c.VIEW_TS + cntx.sep

        # Rename cycle plots.
        rename(p, "daily", c.VIEW_CYCLE_D, recursive=True)
        rename(p, "monthly", c.VIEW_CYCLE_MS, recursive=True)

        # Using 'centile' (instead of 'quantile'), and the range of values is from 0 to 100 (instead of 0 to 1).
        rename(p_map, "q10", "c010", update_content=False, recursive=True)
        rename(p_map, "q90", "c090", update_content=False, recursive=True)

        # Indices.
        idx_name_l =\
            [["cdd"] * 2,
             ["cwd"] * 2,
             ["dc", "drought_code"],
             ["drydays", "dry_days"],
             ["drydurtot", "dry_spell_total_length"],
             ["etr"] * 2,
             ["heatwavemaxlen", "heat_wave_max_length"],
             ["heatwavetotlen", "heat_wave_total_length"],
             ["hotspellfreq", "hot_spell_frequency"],
             ["hotspellmaxlen", "hot_spell_max_length"],
             ["prcptot"] * 2,
             ["pr"] * 2,
             ["rainstart", "rain_season_start"],
             ["rainend", "rain_season_end"],
             ["raindur", "rain_season_length"],
             ["rainqty", "rain_season_prcptot"],
             ["r10mm"] * 2,
             ["r20mm"] * 2,
             ["rnnmm", "wet_days"],
             ["rx1day"] * 2,
             ["rx5day"] * 2,
             ["sdii"] * 2,
             ["sfcWindmax"] * 2,
             ["tasmax"] * 2,
             ["tasmin"] * 2,
             ["tas"] * 2,
             ["tgg"] * 2,
             ["tndaysbelow", "tn_days_below"],
             ["tng"] * 2,
             ["tngmonthsbelow", "tng_months_below"],
             ["tnx"] * 2,
             ["tropicalnights", "tropical_nights"],
             ["tx90p"] * 2,
             ["txdaysabove", "tx_days_above"],
             ["txg"] * 2,
             ["txx"] * 2,
             ["uas"] * 2,
             ["vas"] * 2,
             ["wetdays", "wet_days"],
             ["wgdaysabove", "wg_days_above"],
             ["wsdi"] * 2,
             ["wxdaysabove", "wx_days_above"]]

        # Loop through the items to rename.
        for i in range(len(idx_name_l)):
            rename(p, idx_name_l[i][0], idx_name_l[i][1], rename_files=True, rename_directories=True,
                   update_content=True, recursive=True)
            rename(p_ts, "_era5_land", "_rcp", rename_files=True, rename_directories=False,
                   update_content=False, recursive=True)


def deploy():

    """
    --------------------------------------------------------------------------------------------------------------------
    Deploy a project.

    This copies CSV files from the workflow to the dashboard.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Base path of the dashboard project.
    p_dash = os.getcwd() + cntx.sep + "dashboard" + cntx.sep + "data" + cntx.sep + cntx.country + "-" +\
        cntx.region + "-" + cntx.obs_src + cntx.sep

    # Copy figures.
    for view in [c.VIEW_CYCLE_D, c.VIEW_CYCLE_MS, c.VIEW_MAP, c.VIEW_TS, c.VIEW_TS_BIAS, c.VIEW_TAYLOR]:
        for p_work_i in glob.glob(cntx.d_fig(view) + "*_csv"):
            p_dash_i = p_dash + view + cntx.sep + os.path.basename(p_work_i).replace("_csv", "")
            dir_util.copy_tree(p_work_i, p_dash_i)
        if view == c.VIEW_MAP:
            shutil.copy(cntx.p_bounds, p_dash + view + cntx.sep + c.F_BOUNDS)

    # Copy tables.
    dir_util.copy_tree(cntx.d_tbl(), p_dash + c.VIEW_TBL)
