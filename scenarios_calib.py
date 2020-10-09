# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Quantile mapping functions.
# This requires installing package SciTools (when xr.DataArray.time.dtype=cfg.dtype_obj).
# TODO: Functions in this class should probably be included in scenarios.py to avoid a circular relationship.
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
import scenarios as scen
import utils
import xarray as xr


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
    if p_regrid_list is None:
        utils.log("The required files are not available.", True)
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

        # Loop through combinations of nq values, up_qmf and time_win values.
        for nq in cfg.nq_calib:
            for up_qmf in cfg.up_qmf_calib:
                for time_win in cfg.time_win_calib:

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

                    # Path and title of calibration figure.
                    fn_fig = var + "_" + sim_name_i + "_calibration.png"
                    comb = "nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                    title = sim_name_i + "_" + comb
                    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/calibration", var) + comb + "/" + fn_fig

                    # Calculate QQ and generate calibration plots.
                    msg = "Assessment of " + sim_name_i + ": nq=" + str(nq) + ", up_qmf=" + str(up_qmf) +\
                          ", time_win=" + str(time_win) + " is "
                    if not (os.path.exists(p_fig) and os.path.exists(p_fig.replace(".png", "_ts.png"))):
                        utils.log(msg + "running", True)
                        scen.postprocess(var, nq, up_qmf, time_win, p_obs, p_regrid_ref, p_regrid_fut, "", "", title,
                                         p_fig)
                    else:
                        utils.log(msg + "not required", True)

                    # Error --------------------------------------------------------------------------------------------

                    # Set calibration parameters (nq, up_qmf and time_win) and calculate error according to the
                    # selected method.
                    if cfg.opt_calib_auto:

                        # Calculate the error between observations and simulation for the reference period.
                        ds_obs        = utils.open_netcdf(p_obs)
                        ds_regrid_ref = utils.open_netcdf(p_regrid_ref)
                        bias_err_current = utils.calc_error(ds_obs[var].values.ravel(),
                                                            ds_regrid_ref[var].values.ravel())

                        if (bias_err_best < 0) or (bias_err_current < bias_err_best):
                            col_names = ["nq", "up_qmf", "time_win", "bias_err"]
                            col_values = [float(nq), up_qmf, float(time_win), bias_err_current]
                            cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                             (cfg.df_calib["stn"] == stn) &
                                             (cfg.df_calib["var"] == var), col_names] = col_values

        # Update calibration file.
        if cfg.opt_calib_auto and os.path.exists(cfg.p_calib):
            cfg.df_calib.to_csv(cfg.p_calib)


def init_calib_params():

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize calibration parameters.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Attempt loading a calibration file.
    if os.path.exists(cfg.p_calib):
        cfg.df_calib = pd.read_csv(cfg.p_calib)
        utils.log("Calibration file loaded.", True)
        return

    # List CORDEX files.
    list_cordex = utils.list_cordex(cfg.d_proj, cfg.rcps)

    # Stations.
    stns = cfg.stns
    if cfg.opt_ra:
        stns = [cfg.obs_src]

    # List simulation names, stations and variables.
    sim_name_list = []
    stn_list = []
    var_list = []
    for idx_rcp in range(len(cfg.rcps)):
        rcp = cfg.rcps[idx_rcp]
        for i_sim in range(0, len(list_cordex[rcp])):
            list_i = list_cordex[rcp][i_sim].split("/")
            sim_name = list_i[cfg.get_rank_inst()] + "_" + list_i[cfg.get_rank_inst() + 1]
            for stn in stns:
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
            "time_win": cfg.time_win_default,
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
    ds : xr.Dataset
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    dates = ds.cftime_range(start="0001", periods=24, freq="MS", calendar=cfg.cal_noleap)
    da    = xr.DataArray(np.arange(24), coords=[dates], dims=[cfg.dim_time], name="ds")

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
