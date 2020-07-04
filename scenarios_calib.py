# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Quantile mapping functions.
# This requires installing package SciTools (when xr.DataArray.time.dtype=cfg.dtype_obj).
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# 2. bourgault.marcandre@ouranos.ca (original)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import numpy as np
import os
import plot
import utils
import xarray as xr
from qm import train, predict


def bias_correction_loop(stn, var):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias correction.

    Parameters
    ----------
    stn : str
        Station name.
    var : str
        Weather variable.
    -------------------------------------------------------------------------------------------------------------------
    """

    # Path of directory containing regrid files.
    path_regrid = cfg.get_path_sim(stn, cfg.cat_regrid, var)

    # List of files in 'path'.
    files = utils.list_files(path_regrid)

    # Loop through simulations sets (3 files per simulation set).
    n_set = int(len(files) / 3)
    if len(cfg.idx_sim) == 0:
        cfg.idx_sim = range(0, n_set)
    for i in cfg.idx_sim:
        list_i   = files[i * 3].split("/")
        sim_name = list_i[len(list_i) - 1].replace(var + "_", "").replace(".nc", "")

        # Best parameter set.
        error_best = -1

        # Loop through nq values.
        for nq in cfg.nq_calib:

            # Loop through up_qmf values.
            for up_qmf in cfg.up_qmf_calib:

                # Loop through time_int values.
                for time_int in cfg.time_int_calib:

                    print("Correcting i=" + str(i) + ", up_qmf=" + str(up_qmf) + ", time_int=" + str(time_int))

                    # NetCDF file.
                    fn_obs = cfg.get_path_obs(stn, var)
                    fn_ref = [i for i in files if "ref" in i][i]
                    fn_fut = fn_ref.replace("ref_", "")
                    fn_qqmap = cfg.get_path_sim(stn, cfg.cat_qqmap, var)

                    # Figures.
                    fn_fig = fn_fut.split("/")[-1].replace("4plot_calib_summaryqqmap.nc", "calib.png")
                    title = os.path.basename(fn_fig) + "_time_int_" + str(time_int) + "_up_qmf_" + str(up_qmf) + \
                        "_nq_" + str(nq)
                    path_fig = cfg.get_path_sim(stn, cfg.cat_fig + "/calib", var)
                    if not (os.path.isdir(path_fig)):
                        os.makedirs(path_fig)
                    fn_fig = path_fig + fn_fig

                    # Examine bias correction.
                    bias_correction(var, nq, up_qmf, time_int, fn_obs, fn_ref, fn_fut, fn_qqmap, title, fn_fig)

                    if not cfg.opt_calib_extra:
                        continue

                    # Extra --------------------------------------------------------------------------------------------

                    ds_obs = xr.open_dataset(fn_obs)
                    ds_fut = xr.open_dataset(fn_fut)

                    # Generate plot.
                    plot.plot_obs_fut(ds_obs, ds_fut, var, title, fn_fig)

                    # TODO: Calculate the error between observations and simulation for the reference period.
                    error_current = -1

                    # Set nq, up_qmf and time_int for the current simulation.
                    if cfg.opt_calib_auto and ((error_best < 0) or (error_current < error_best)):
                        cfg.nq[sim_name][stn][var]       = float(nq)
                        cfg.up_qmf[sim_name][stn][var]   = up_qmf
                        cfg.time_int[sim_name][stn][var] = float(time_int)


def bias_correction(var, nq, up_qmf, time_int, fn_obs, fn_ref, fn_fut, fn_qqmap, title, fn_fig):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias correction.

    Parameters
    ----------
    var : str
        Weather variable.
    nq : float
        Number of quantiles
    up_qmf : float
        Upper limit for quantile mapping function.
    time_int : float
        Windows size (i.e. number of days before + number of days after).
    fn_obs : str
        NetCDF file for observations.
    fn_ref : str
        NetCDF file for reference period.
    fn_fut : str
        NetCDF file for future period.
    fn_qqmap : str
        NetCDF file of qqmap.
    title : str
        Title of figure.
    fn_fig : str
        File of figure associated with the plot.
    -------------------------------------------------------------------------------------------------------------------
    """

    # Datasets.
    ds_obs = xr.open_dataset(fn_obs)[var].squeeze()
    ds_ref = xr.open_dataset(fn_ref)[var]
    ds_fut = xr.open_dataset(fn_fut)[var]

    # DELETE: Amount of precipitation in winter.
    # DELETE: if var == cfg.var_cordex_pr:
    # DELETE:     pos_ref = (ds_ref.time.dt.month >= 11) | (ds_ref.time.dt.month <= 3)
    # DELETE:     pos_fut = (ds_fut.time.dt.month >= 11) | (ds_fut.time.dt.month <= 3)
    # DELETE:     ds_ref[pos_ref] = 1e-8
    # DELETE:     ds_fut[pos_fut] = 1e-8
    # DELETE:     ds_ref.values[ds_ref<1e-7] = 0
    # DELETE:     ds_fut.values[ds_fut<1e-7] = 0

    # Information for post-processing ---------------------------------------------------------------------------------

    # Precipitation.
    if var == cfg.var_cordex_pr:
        kind = "*"
        ds_obs.interpolate_na(dim="time")
    # Temperature.
    elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        ds_obs  = ds_obs + 273.15
        ds_obs  = ds_obs.interpolate_na(dim="time")
        kind = "+"
    # Other variables.
    else:
        ds_obs = ds_obs.interpolate_na(dim="time")
        kind = "+"

    # Calculate/read quantiles -----------------------------------------------------------------------------------------

    # Calculate QMF.
    ds_qmf = train(ds_ref.squeeze(), ds_obs.squeeze(), int(nq), cfg.group, kind, time_int,
                   detrend_order=cfg.detrend_order)

    # Calculate QQMAP.
    if cfg.opt_calib_qqmap:
        if var == cfg.var_cordex_pr:
            ds_qmf.values[ds_qmf > up_qmf] = up_qmf
        ds_qqmap     = predict(ds_fut.squeeze(), ds_qmf, interp=True, detrend_order=cfg.detrend_order)

    # Read QQMAP.
    else:
        ds_qqmap = xr.open_dataset(fn_qqmap)[var]

    ds_qqmap_per = ds_qqmap.where((ds_qqmap.time.dt.year >= cfg.per_ref[0]) &
                                  (ds_qqmap.time.dt.year <= cfg.per_ref[1]), drop=True)

    plot.plot_calib_summary(ds_qmf, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, var, title, fn_fig)


def physical_coherence(stn, var):

    """
    # ------------------------------------------------------------------------------------------------------------------
    Verifies physical coherence.

    Parameters
    ----------
    stn: str
        Station name.
    var : [str]
        List of variables.
    --------------------------------------------------------------------------------------------------------------------
    """

    path_qqmap   = cfg.get_path_sim(stn, cfg.cat_qqmap, var[0])
    files        = utils.list_files(path_qqmap)
    file_tasmin  = files
    file_tasmax  = [i.replace(cfg.var_cordex_tasmin, cfg.var_cordex_tasmax) for i in files]

    for i in range(len(file_tasmin)):
        i = 10
        print(stn + "____________" + file_tasmax[i])
        out_tasmax = xr.open_dataset(file_tasmax[i])
        out_tasmin = xr.open_dataset(file_tasmin[i])

        pos = out_tasmax[var[1]] < out_tasmin[var[0]]

        val_max = out_tasmax[var[1]].values[pos]
        val_min = out_tasmin[var[0]].values[pos]

        out_tasmax[var[1]].values[pos] = val_min
        out_tasmin[var[0]].values[pos] = val_max

        os.remove(file_tasmax[i])
        os.remove(file_tasmin[i])

        out_tasmax.to_netcdf(file_tasmax[i])
        out_tasmin.to_netcdf(file_tasmin[i])


def adjust_date_format(ds):

    """
    --------------------------------------------------------------------------------------------------------------------
    Adjusts date format.

    Parameters
    ----------
    ds : ...
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    dates = ds.cftime_range(start="0001", periods=24, freq="MS", calendar=cfg.cal_noleap)
    da    = xr.DataArray(np.arange(24), coords=[dates], dims=["time"], name="ds")

    return da


def run(var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    TODO: Quantify error numerically to facilitate calibration. "bias_correction" could return the error.
    TODO: Determine why an exception is launched at i = 9. It has to do with datetime format.

    Parameters
    ----------
    var: str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Module calib launched.")

    # Loop through stations.
    for stn in cfg.stns:

        # Bias correction.
        if cfg.opt_calib_bias:

            bias_correction_loop(stn, var)

        # Physical coherence.
        if cfg.opt_calib_coherence:
            physical_coherence(stn, [cfg.var_cordex_tasmin, cfg.var_cordex_tasmax])

    print("Module calib completed successfully.")


if __name__ == "__main__":
    run()
