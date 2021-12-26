# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions for file manipulation.
#
# Contact information:
# 1. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import constants as const
import glob
import matplotlib.pyplot
import os
import pandas as pd
import utils
import xarray as xr
import warnings
from config import cfg
from itertools import compress
from typing import Union, List

import sys
sys.path.append("dashboard")
from dashboard import def_varidx as vi

# Files.
f_csv     = "csv"         # CSV file type (comma-separated values).
f_png     = "png"         # PNG file type (image).
f_tif     = "tif"         # TIF file type (image, potentially georeferenced).
f_nc      = "nc"          # NetCDF file type.
f_nc4     = "nc4"         # NetCDF v4 file type.
f_ext_csv = "." + f_csv   # CSV file extension.
f_ext_png = "." + f_png   # PNG file extension.
f_ext_tif = "." + f_tif   # TIF file extension.
f_ext_nc  = "." + f_nc    # NetCDF file extension.
f_ext_nc4 = "." + f_nc4   # NetCDF v4 file extension.
f_ext_log = ".log"        # LOG file extension.


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
            n_netcdf = len(glob.glob(j + "*" + f_ext_nc))
            if n_netcdf > 0:
                tokens_j = j.split(cfg.sep)
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
    p : str
        Path of directory.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List.
    p_l = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for p in f:
            if f_ext_nc in p:
                p_l.append(os.path.join(r, p))

    # Sort.
    p_l.sort()

    return p_l


def create_mask(
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a mask, based on climate scenarios for the temperature or precipitation variable.
    All values with a value are attributed a value of 1. Other values are assigned 'nan'.
    --------------------------------------------------------------------------------------------------------------------
    """

    da_mask = None

    f_l = glob.glob(cfg.d_stn + "*/*" + f_ext_nc)
    for i in range(len(f_l)):

        # Open NetCDF file.
        ds = open_netcdf(f_l[i])
        var = list(ds.data_vars)[0]
        if var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:

            # Create mask.
            da_mask = ds[var][0] * 0 + 1

            break

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
        if cfg.use_chunks and (cfg.n_proc == 1) and (chunks is None) and ("scen" in p) and (const.dim_time in ds.dims):
            chunks = {const.dim_time: len(ds[const.dim_time])}

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
    p_inc = p.replace(f_ext_nc, ".incomplete")
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
        p_nc = p_inc.replace(".incomplete", f_ext_nc)
        if os.path.exists(p_nc):
            log("Removing: " + p_nc)
            os.remove(p_nc)

        # Remove the temporary file.
        if os.path.exists(p_inc):
            log("Removing: " + p_inc)
            os.remove(p_inc)


def save_plot(
    plt: matplotlib.pyplot,
    p: str,
    desc=""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a plot to a file.

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

    # Save file.
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


def log(
    msg: str,
    indent=False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Log message to console and into a file.

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
        ln = utils.get_datetime_str()

    if indent:
        ln += " " * const.log_n_blank
    if (msg == "-") or (msg == "="):
        if indent:
            ln += msg * const.log_sep_len
        else:
            ln += msg * (const.log_sep_len + const.log_n_blank)
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
        p_log = p_log.replace(f_ext_log, "_" + str(pid_current) + f_ext_log)
    if p_log != "":
        f = open(p_log, "a")
        f.writelines(ln + "\n")
        f.close()
