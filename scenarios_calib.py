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
import pandas as pd
import plot
import scenarios as scen
import utils
import xarray as xr
from qm import train, predict


def bias_correction(stn, var, sim_name=""):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias correction.

    Parameters
    ----------
    stn : str
        Station name.
    var : str
        Weather variable.
    sim_name : str
        Simulation name.
    -------------------------------------------------------------------------------------------------------------------
    """

    # List regrid files.
    d_regrid = cfg.get_d_sim(stn, cfg.cat_regrid, var)
    p_regrid_list = utils.list_files(d_regrid)
    if p_regrid_list == []:
        return
    p_regrid_list = [i for i in p_regrid_list if "_4qqmap" not in i]
    p_regrid_list.sort()

    # Loop through simulation sets.
    for i in range(len(p_regrid_list)):
        p_regrid_tokens = p_regrid_list[i].split("/")
        sim_name_i = p_regrid_tokens[len(p_regrid_tokens) - 1].replace(var + "_", "").replace(".nc", "")

        # Skip iteration if it does not correspond to the specified simulation name.
        if (sim_name != "") and (sim_name != sim_name_i):
            continue

        # Best parameter set.
        bias_err_best = -1

        # Loop through combinations of nq values, up_qmf and time_int values.
        for nq in cfg.nq_calib:
            for up_qmf in cfg.up_qmf_calib:
                for time_int in cfg.time_int_calib:

                    msg = "Assessing " + sim_name_i + ": nq=" + str(nq) + ", up_qmf=" + str(up_qmf) +\
                          ", time_int=" + str(time_int)
                    utils.log(msg, True)

                    # NetCDF files.
                    p_obs        = cfg.get_p_obs(stn, var)
                    p_regrid     = p_regrid_list[i]
                    p_regrid_ref = p_regrid.replace(".nc", "_ref_4qqmap.nc")
                    p_regrid_fut = p_regrid.replace(".nc", "_4qqmap.nc")
                    msg = "File missing: "
                    if not(os.path.exists(p_obs)) or not(os.path.exists(p_regrid_ref)) or\
                       not(os.path.exists(p_regrid_fut)):
                        if not(os.path.exists(p_obs)):
                            utils.log(msg + p_obs, True)
                        if not(os.path.exists(p_regrid_ref)):
                            utils.log(msg + p_regrid_ref, True)
                        if not(os.path.exists(p_regrid_fut)):
                            utils.log(msg + p_regrid_fut, True)
                        continue

                    # Calculate QQmap.
                    scen.postprocess(var, stn, int(nq), up_qmf, int(time_int), p_obs, p_regrid_ref, p_regrid_fut, "")

                    # Figures.
                    fn_fig = var + "_" + sim_name_i + "_calib.png"
                    title = sim_name_i + "_time_int_" + str(time_int) + "_up_qmf_" + str(up_qmf) + \
                        "_nq_" + str(nq)
                    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/calib", var) + fn_fig

                    # Examine bias correction.
                    if not(os.path.exists(p_fig)):
                        bias_correction_spec(var, nq, up_qmf, time_int, p_obs, p_regrid_ref, p_regrid_fut, "", title,
                                             p_fig)

                    # Time series --------------------------------------------------------------------------------------

                    ds_obs        = xr.open_dataset(p_obs)
                    ds_regrid_ref = xr.open_dataset(p_regrid_ref)
                    ds_regrid_fut = xr.open_dataset(p_regrid_fut)
                    if (var == cfg.var_cordex_tas) or (var == cfg.var_cordex_tasmin) or (var == cfg.var_cordex_tasmax):
                        ds_regrid_ref[var] = ds_regrid_ref[var] - 273.15
                        ds_regrid_fut[var] = ds_regrid_fut[var] - 273.15

                    # Generate plot.
                    p_fig = p_fig.replace(".png", "_ts.png")
                    if not(os.path.exists(p_fig)):
                        plot.plot_obs_fut(ds_obs, ds_regrid_fut, var, title, p_fig)

                    # Error --------------------------------------------------------------------------------------------

                    # Set calibration parameters (nq, up_qmf and time_int) and calculate error according to the
                    # selected method.
                    if cfg.opt_calib_auto:

                        # Calculate the error between observations and simulation for the reference period.
                        bias_err_current = utils.calc_error(ds_obs[var].values.ravel(), ds_regrid_ref[var].values.ravel())

                        if (bias_err_best < 0) or (bias_err_current < bias_err_best):
                            col_names = ["nq", "up_qmf", "time_int", "bias_err"]
                            col_values = [float(nq), up_qmf, float(time_int), bias_err_current]
                            cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                             (cfg.df_calib["stn"] == stn) &
                                             (cfg.df_calib["var"] == var), col_names] = col_values

        # Update calibration file.
        if os.path.exists(cfg.p_calib):
            cfg.df_calib.to_csv(cfg.p_calib)


def bias_correction_spec(var, nq, up_qmf, time_int, p_obs, p_ref, p_fut, p_qqmap, title, p_fig):

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
    p_obs : str
        NetCDF file for observations.
    p_ref : str
        NetCDF file for reference period.
    p_fut : str
        NetCDF file for future period.
    p_qqmap : str
        NetCDF file of qqmap.
    title : str
        Title of figure.
    p_fig : str
        Path of figure.
    -------------------------------------------------------------------------------------------------------------------
    """

    # Datasets.
    ds_obs = xr.open_dataset(p_obs)[var].squeeze()
    ds_ref = xr.open_dataset(p_ref)[var]
    ds_fut = xr.open_dataset(p_fut)[var]

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
    ds_qmf = train(ds_ref.squeeze(), ds_obs.squeeze(), int(nq), cfg.group, kind, int(time_int),
                   detrend_order=cfg.detrend_order)

    # Calculate QQMAP.
    if cfg.opt_calib_qqmap:
        if var == cfg.var_cordex_pr:
            ds_qmf.values[ds_qmf > up_qmf] = up_qmf
        ds_qqmap = predict(ds_fut.squeeze(), ds_qmf, interp=True, detrend_order=cfg.detrend_order)

    # Read QQMAP.
    else:
        ds_qqmap = xr.open_dataset(p_qqmap)[var]

    ds_qqmap_per = ds_qqmap.where((ds_qqmap.time.dt.year >= cfg.per_ref[0]) &
                                  (ds_qqmap.time.dt.year <= cfg.per_ref[1]), drop=True)

    plot.plot_calib_summary(ds_qmf, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, var, title, p_fig)


def init_calib_params():

    """
    -----------------------------------------------------------------------------------------------------------------
    Initialize calibration parameters.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Attempt loading a calibration file.
    if os.path.exists(cfg.p_calib):
        cfg.df_calib = pd.read_csv(cfg.p_calib)
        utils.log("Calibration file loaded.", True)
        return

    # List CORDEX files.
    list_cordex = utils.list_cordex(cfg.d_cordex, cfg.rcps)

    # List simulation names, stations and variables.
    sim_name_list = []
    stn_list = []
    var_list = []
    for idx_rcp in range(len(cfg.rcps)):
        rcp = cfg.rcps[idx_rcp]
        for idx_sim_i in range(0, len(list_cordex[rcp])):
            list_i = list_cordex[rcp][idx_sim_i].split("/")
            sim_name = list_i[cfg.get_idx_inst()] + "_" + list_i[cfg.get_idx_inst() + 1]
            for stn in cfg.stns:
                for var in cfg.variables_cordex:
                    sim_name_list.append(sim_name)
                    stn_list.append(stn)
                    var_list.append(var)

    # Build pandas dataframe.
    dict = {"sim_name": sim_name_list,
            "stn": stn_list,
            "var": var_list,
            "nq": cfg.nq_default,
            "up_qmf": cfg.up_qmf_default,
            "time_int": cfg.time_int_default,
            "bias_err": cfg.bias_err_default}
    cfg.df_calib = pd.DataFrame(dict)

    # Save calibration parameters to a CSV file.
    if cfg.p_calib != "":
        cfg.df_calib.to_csv(cfg.p_calib)
        if os.path.exists(cfg.p_calib):
            utils.log("Calibration file created.", True)


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


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through combinations of stations and variables.
    for stn in cfg.stns:
        for var in cfg.variables_cordex:

            utils.log("-", True)
            utils.log("Station  : " + stn, True)
            utils.log("Variable : " + var, True)
            utils.log("-", True)

            # Perform bias correction.
            bias_correction(stn, var)


if __name__ == "__main__":
    run()
