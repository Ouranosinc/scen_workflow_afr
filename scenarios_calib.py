# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Quantile mapping functions.
# This requires installing package SciTools (when xr.DataArray.time.dtype=cfg.dtype_obj).
# TODO: Functions in this class should probably be included in scenarios.py to avoid a circular relationship.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca (current)
# 2. marc-andre.bourgault@ggr.ulaval.ca (second)
# 3. rondeau-genesse.gabriel@ouranos.ca (original)
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import numpy as np
import os
import pandas as pd
import scenarios as scen
import utils
import xarray as xr
import warnings


def bias_adj(stn: str, var: str, sim_name: str = "", calc_err: bool = False):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias adjustment.

    Parameters
    ----------
    stn : str
        Station name.
    var : str
        Weather variable.
    sim_name : str
        Simulation name.
    calc_err : bool
        If True, only calculate the error (will not work properly in parallel mode).
    -------------------------------------------------------------------------------------------------------------------
    """

    # List regrid files.
    d_regrid = cfg.get_d_scen(stn, cfg.cat_regrid, var)
    p_regrid_l = utils.list_files(d_regrid)
    if p_regrid_l is None:
        utils.log("The required files are not available.", True)
        return
    p_regrid_l = [i for i in p_regrid_l if "_4qqmap" not in i]
    p_regrid_l.sort()

    # Loop through simulation sets.
    for i in range(len(p_regrid_l)):
        p_regrid_tokens = p_regrid_l[i].split("/")
        sim_name_i = p_regrid_tokens[len(p_regrid_tokens) - 1].replace(var + "_", "").replace(cfg.f_ext_nc, "")

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
                    p_stn        = cfg.get_p_stn(var, stn)
                    p_regrid     = p_regrid_l[i]
                    p_regrid_ref = p_regrid.replace(cfg.f_ext_nc, "_ref_4qqmap" + cfg.f_ext_nc)
                    p_regrid_fut = p_regrid.replace(cfg.f_ext_nc, "_4qqmap" + cfg.f_ext_nc)

                    # If there is a single combination of calibration parameters, NetCDF files can be saved, so that
                    # they don't have to be created during post-process.
                    p_qqmap = ""
                    p_qmf   = ""
                    if (len(cfg.nq_calib) == 1) and (len(cfg.up_qmf_calib) == 1) and (len(cfg.time_win_calib) == 1):
                        p_qqmap = p_regrid.replace(cfg.cat_regrid, cfg.cat_qqmap)
                        p_qmf   = p_regrid.replace(cfg.cat_regrid, cfg.cat_qmf)

                    # Verify if required files exist.
                    msg = "File missing: "
                    if ((not os.path.exists(p_stn)) or (not os.path.exists(p_regrid_ref)) or
                        (not os.path.exists(p_regrid_fut))) and\
                       (not cfg.opt_force_overwrite):
                        if not(os.path.exists(p_stn)):
                            utils.log(msg + p_stn, True)
                        if not(os.path.exists(p_regrid_ref)):
                            utils.log(msg + p_regrid_ref, True)
                        if not(os.path.exists(p_regrid_fut)):
                            utils.log(msg + p_regrid_fut, True)
                        continue

                    # Load station data, drop February 29th, and select reference period.
                    ds_stn = utils.open_netcdf(p_stn)
                    ds_stn = utils.remove_feb29(ds_stn)
                    ds_stn = utils.sel_period(ds_stn, cfg.per_ref)
                    # Add small perturbation.
                    # if var in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
                    #     ds_stn = scen.perturbate(ds_stn, var)

                    # Path and title of calibration figure and csv file.
                    fn_fig = var + "_" + sim_name_i + "_" + cfg.cat_fig_calibration + cfg.f_ext_png
                    comb = "nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                    title = sim_name_i + "_" + comb
                    p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/" + cfg.cat_fig_calibration, var) + comb + "/" + fn_fig
                    p_fig_csv = p_fig.replace("/" + var + "/", "/" + var + "_" + cfg.f_csv + "/").\
                        replace(cfg.f_ext_png, "_" + cfg.stat_mean + cfg.f_ext_csv)
                    p_fig_ts = p_fig.replace(cfg.f_ext_png, "_ts" + cfg.f_ext_png)
                    p_fig_ts_csv = p_fig_ts.replace("/" + var + "/", "/" + var + "_" + cfg.f_csv + "/").\
                        replace(cfg.f_ext_png, cfg.f_ext_csv)

                    # Bias adjustment ----------------------------------------------------------------------------------

                    if not calc_err:

                        msg = "Assessment of " + sim_name_i + ": nq=" + str(nq) + ", up_qmf=" + str(up_qmf) +\
                              ", time_win=" + str(time_win) + " is "
                        if not(os.path.exists(p_fig)) or \
                           (not(os.path.exists(p_fig_csv)) and cfg.opt_save_csv[0]) or\
                           not(os.path.exists(p_fig_ts)) or \
                           (not (os.path.exists(p_fig_ts_csv)) and cfg.opt_save_csv[0]) or \
                           not(os.path.exists(p_qqmap)) or not(os.path.exists(p_qmf)) or cfg.opt_force_overwrite:
                            utils.log(msg + "running", True)

                            # Calculate QQ and generate calibration plots.
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                scen.postprocess(var, nq, up_qmf, time_win, ds_stn, p_regrid_ref, p_regrid_fut, p_qqmap,
                                                 p_qmf, title, p_fig)
                        else:
                            utils.log(msg + "not required", True)

                    # Bias error ---------------------------------------------------------------------------------------

                    elif os.path.exists(p_qqmap):

                        # Extract the reference period from the adjusted simulation.
                        ds_qqmap_ref = utils.open_netcdf(p_qqmap)
                        ds_qqmap_ref = utils.remove_feb29(ds_qqmap_ref)
                        ds_qqmap_ref = utils.sel_period(ds_qqmap_ref, cfg.per_ref)

                        # Calculate the error between observations and simulation for the reference period.
                        bias_err_current = float(round(utils.calc_error(ds_stn[var], ds_qqmap_ref[var]), 4))

                        # Set calibration parameters (nq, up_qmf and time_win) and calculate error according to the
                        # selected method.
                        if (not cfg.opt_calib_auto) or\
                           (cfg.opt_calib_auto and ((bias_err_best < 0) or (bias_err_current < bias_err_best))):
                            col_names = ["nq", "up_qmf", "time_win", "bias_err"]
                            col_values = [float(nq), up_qmf, float(time_win), bias_err_current]
                            calib_row = (cfg.df_calib["sim_name"] == sim_name) &\
                                        (cfg.df_calib["stn"] == stn) &\
                                        (cfg.df_calib["var"] == var)
                            cfg.df_calib.loc[calib_row, col_names] = col_values
                            utils.save_csv(cfg.df_calib, cfg.p_calib)


def init_calib_params():

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize calibration parameters.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Simulations, stations and variables.
    sim_name_l = []
    stn_l      = []
    var_l      = []
    nq_l       = []
    up_qmf_l   = []
    time_win_l = []
    bias_err_l = []

    # Function used to build the dataframe.
    def build_df(sim_name_l: str, stn_l: str, var_l: str, nq_l: int, up_qmf_l: float, time_win_l: int,
                 bias_err_l: float) -> pd.DataFrame:

        dict_pd = {
            "sim_name": sim_name_l,
            "stn": stn_l,
            "var": var_l,
            "nq": nq_l,
            "up_qmf": up_qmf_l,
            "time_win": time_win_l,
            "bias_err": bias_err_l}
        return pd.DataFrame(dict_pd)

    # Attempt loading a calibration file.
    if os.path.exists(cfg.p_calib):
        cfg.df_calib = pd.read_csv(cfg.p_calib)
        if len(cfg.df_calib) > 0:
            sim_name_l = list(cfg.df_calib["sim_name"])
            stn_l      = list(cfg.df_calib["stn"])
            var_l      = list(cfg.df_calib["var"])
            nq_l       = list(cfg.df_calib["nq"])
            up_qmf_l   = list(cfg.df_calib["up_qmf"])
            time_win_l = list(cfg.df_calib["time_win"])
            bias_err_l = list(cfg.df_calib["bias_err"])
        utils.log("Calibration file loaded.", True)

    # List CORDEX files.
    list_cordex = utils.list_cordex(cfg.d_proj, cfg.rcps)

    # Stations.
    stns = cfg.stns
    if cfg.opt_ra:
        stns = [cfg.obs_src]

    # List simulation names, stations and variables.
    for i_rcp in range(len(cfg.rcps)):
        rcp = cfg.rcps[i_rcp]
        sim_l = list_cordex[rcp]
        sim_l.sort()
        for i_sim in range(0, len(sim_l)):
            list_i = list_cordex[rcp][i_sim].split("/")
            sim_name = list_i[cfg.get_rank_inst()] + "_" + list_i[cfg.get_rank_inst() + 1]
            for stn in stns:
                for var in cfg.variables_cordex:

                    # Add the combination if it does not already exist.
                    if (cfg.df_calib is None) or\
                       (len(cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                             (cfg.df_calib["stn"] == stn) &
                                             (cfg.df_calib["var"] == var)]) == 0):
                        sim_name_l.append(sim_name)
                        stn_l.append(stn)
                        var_l.append(var)
                        nq_l.append(cfg.nq_default)
                        up_qmf_l.append(cfg.up_qmf_default)
                        time_win_l.append(cfg.time_win_default)
                        bias_err_l.append(cfg.bias_err_default)

                        # Update dataframe.
                        cfg.df_calib = build_df(sim_name_l, stn_l, var_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Build dataframe.
    cfg.df_calib = build_df(sim_name_l, stn_l, var_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Save calibration parameters to a CSV file.
    if cfg.p_calib != "":
        utils.save_csv(cfg.df_calib, cfg.p_calib)
        if os.path.exists(cfg.p_calib):
            utils.log("Calibration file created or updated.", True)


def adjust_date_format(ds: xr.Dataset):

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
            bias_adj(stn, var)
