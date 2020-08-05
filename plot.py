# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Plot functions.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# 2. marc-andre.bourgault@ggr.ulaval.ca (original)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import os.path
import seaborn as sns
import utils
import xarray as xr
from scipy import signal


def plot_year(ds_hour, ds_day, set_name, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Time series.

    Parameters
    ----------
    ds_hour : xr.Dataset
        Dataset with hourly frequency.
    ds_day : xr.Dataset
        Dataset with daily frequency.
    set_name : str
        Set name.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    var_desc_unit = cfg.get_var_desc(var, set_name)  + " [" + cfg.get_var_unit(var, set_name) + "]"

    fs = 10
    f = plt.figure(figsize=(10, 3))
    f.suptitle("Comparison entre les données horaires et agrégées", fontsize=fs)

    plt.subplots_adjust(top=0.9, bottom=0.20, left=0.10, right=0.98, hspace=0.30, wspace=0.416)
    ds_hour.plot(color="black")
    plt.title("")
    plt.xlabel("", fontsize=fs)
    plt.ylabel(var_desc_unit, fontsize=fs)

    ds_day.plot(color="orange")
    plt.title("")
    plt.xlabel("", fontsize=fs)
    plt.ylabel(var_desc_unit, fontsize=fs)

    plt.close()


def plot_dayofyear(ds_day, set_name, var, date):

    """
    --------------------------------------------------------------------------------------------------------------------
    Map.

    Parameters
    ----------
    ds_day : xr.Dataset
        Dataset with daily frequency.
    set_name : str
        Set name.
    var : str
        Variable.
    date : str
        Date.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable.
    var_desc = cfg.get_var_desc(var, set_name) + " (" + cfg.get_var_unit(var, set_name) + ")"

    # Data.
    ds_day_sel = ds_day.sel(time=date)

    # Plot.
    fs = 10
    ds_day_sel.plot.pcolormesh(add_colorbar=True, add_labels=True,
                               cbar_kwargs=dict(orientation='vertical', pad=0.05, shrink=1, label=var_desc))
    plt.title(date)
    plt.suptitle("", fontsize=fs)
    plt.xlabel("Longitude (º)", fontsize=fs)
    plt.ylabel("Latitude (º)", fontsize=fs)
    plt.tick_params(axis="x", labelsize=fs)
    plt.tick_params(axis="y", labelsize=fs)

    plt.close()


def plot_ts_single(stn, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (one simulation: one plot).

    Parameters:
    ----------
    stn: str
        Station name.
    var: str
        Weather variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    utils.log("Processing (single): variable = " + var + "; station = " + stn, True)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Paths and NetCDF files.
    d_regrid = cfg.get_d_sim(stn, cfg.cat_regrid, var)
    p_list   = utils.list_files(d_regrid)
    p_obs    = cfg.get_p_obs(stn, var)

    # Plot.
    fs_title  = 8
    fs_legend = 8
    fs_axes   = 8

    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.18, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

    # Loop through simulation sets.
    for i in range(int(len(p_list) / 3)):

        p_ref   = [i for i in p_list if "ref" in i][i]
        p_fut   = p_ref.replace("ref_", "")
        p_qqmap = p_fut.replace("_4qqmap", "").replace(cfg.cat_regrid, cfg.cat_qqmap)
        ds_fut   = xr.open_dataset(p_fut)
        ds_qqmap = xr.open_dataset(p_qqmap)
        ds_obs   = xr.open_dataset(p_obs)

        # Convert date format if the need is.
        if ds_fut.time.dtype == cfg.dtype_obj:
            ds_fut["time"] = utils.fix_calendar(ds_fut)
        if ds_qqmap.time.dtype == cfg.dtype_obj:
            ds_qqmap["time"] = utils.fix_calendar(ds_qqmap)

        # Curves.
        (ds_obs[var]).plot(alpha=0.5)
        (ds_fut[var]).plot()
        (ds_qqmap[var]).plot(alpha=0.5)

        # Format.
        plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
        title = os.path.basename(p_fut).replace("4qqmap.nc", "verif_ts_single")
        plt.suptitle(title, fontsize=fs_title)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.title("")
        plt.suptitle(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)

        # Save plot.
        p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/verif/ts_single", var) + title + ".png"
        utils.save_plot(plt, p_fig)

        # Close plot.
        plt.close()


def plot_ts_mosaic(stn, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (all in one plot).

    Parameters:
    ----------
    stn: str
        Station name.
    var: str
        Weather variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    utils.log("Processing (mosaic): variable = " + var + "; station = " + stn, True)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd

    # Paths and NetCDF files.
    d_regrid = cfg.get_d_sim(stn, cfg.cat_regrid, var)
    p_list   = utils.list_files(d_regrid)
    p_obs    = cfg.get_p_obs(stn, var)

    # Plot.
    fs_title  = 6
    fs_legend = 6
    fs_axes   = 6
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.04, right=0.99, hspace=0.40, wspace=0.30)

    # Loop through simulation sets.
    title = ""
    for i in range(int(len(p_list) / 3)):

        # NetCDF files.
        p_fut_i   = [i for i in p_list if "ref" in i][i].replace("ref_", "")
        p_qqmap_i = p_fut_i.replace("_4" + cfg.cat_qqmap, "").replace(cfg.cat_regrid, cfg.cat_qqmap)

        # Open datasets.
        ds_fut   = xr.open_dataset(p_fut_i)
        ds_qqmap = xr.open_dataset(p_qqmap_i)
        ds_obs   = xr.open_dataset(p_obs)

        # Convert date format if the need is.
        if ds_fut.time.dtype == cfg.dtype_obj:
            ds_fut["time"] = utils.fix_calendar(ds_fut)
        if ds_qqmap.time.dtype == cfg.dtype_obj:
            ds_qqmap["time"] = utils.fix_calendar(ds_qqmap)

        # Curves.
        plt.subplot(7, 7, i + 1)
        (ds_fut[var] * coef).plot()
        (ds_qqmap[var] * coef).plot(alpha=0.5)
        (ds_obs[var] * coef).plot(alpha=0.5)

        # Format.
        plt.xlabel("", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        title = os.path.basename(p_list[i]).replace(".nc", "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)
        if i == 0:
            plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
            sup_title = title + "_verif_ts_mosaic"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Save plot.
    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/verif/ts_mosaic", var) + title + ".png"
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_monthly(stn, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (all in one plot).

    Parameters:
    ----------
    stn: str
        Station name.
    var: str
        Weather variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    utils.log("Processing (monthly): variable = " + var + "; station = " + stn, True)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # NetCDF files.
    d_regrid = cfg.get_d_sim(stn, cfg.cat_regrid, var)
    p_list   = utils.list_files(d_regrid)
    p_obs    = cfg.get_p_obs(stn, var)
    ds_obs   = xr.open_dataset(p_obs)
    ds_plt   = ds_obs.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().groupby("time.month").\
               mean()[var]

    # Plot.
    fs_title  = 8
    fs_title  = 6
    fs_legend = 6
    fs_axes   = 6
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.04, right=0.99, hspace=0.40, wspace=0.30)

    # Loop through simulation sets.
    sup_title = ""
    for i in range(int(len(p_list) / 3)):

        # Plot.
        plt.subplot(7, 7, i + 1)

        # Curves.
        ds = xr.open_dataset(p_list[i])[var]
        if isinstance(ds.time[0].values, np.datetime64):
            ds.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().groupby("time.month").mean().\
                plot(color="blue")
        ds = xr.open_dataset(p_list[i])[var]
        if isinstance(ds.time[0].values, np.datetime64):
            ds.sel(time=slice("2050-01-01", "2070-12-31")).resample(time="M").mean().groupby("time.month").mean().\
                plot(color="green")
        ds_plt.plot(color="red")

        # Format.
        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        title = os.path.basename(p_list[i]).replace(".nc", "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)
        if i == 0:
            sup_title = title + "_verif_monthly"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Format.
    plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)

    # Save plot.
    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/verif/monthly", var) + sup_title + ".png"
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_postprocess(p_obs, p_fut, p_qqmap, var, p_fig, title):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of observed and future periods.

    Parameters
    ----------
    p_obs : str
        Dataset of observed period.
    p_fut : str
        Dataset of future period.
    p_qqmap : str
        Dataset of adjusted simulation.
    var : str
        Variable.
    title : str
        Title of figure.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    ds_obs = xr.open_dataset(p_obs)[var]
    ds_fut = xr.open_dataset(p_fut)[var]
    ds_qqmap = xr.open_dataset(p_qqmap)[var]

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd
    elif (var == cfg.var_cordex_tas) or (var == cfg.var_cordex_tasmin) or (var == cfg.var_cordex_tasmax):
        ds_fut   = ds_fut - 273.15
        ds_qqmap = ds_qqmap - 273.15

    # Plot.
    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.04, right=0.99, hspace=0.695, wspace=0.416)
    fs_sup_title = 8
    fs_legend = 8
    fs_axes = 8
    legend_items = ["Simulation", "Observation"]
    if ds_qqmap is not None:
        legend_items.insert(0, "Sim. ajustée")
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.year).mean().plot.line(color=cfg.col_sim_adj)
    (ds_fut * coef).groupby(ds_fut.time.dt.year).mean().plot.line(color=cfg.col_sim_fut)
    (ds_obs * coef).groupby(ds_obs.time.dt.year).mean().plot(color=cfg.col_obs)

    # Customize.
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_sup_title)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_workflow(var, nq, up_qmf, time_win, p_regrid_ref, p_regrid_fut, p_fig):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of reference and future periods.

    Parameters
    ----------
    var : str
        Weather var.
    nq : int
        ...
    up_qmf : float
        ...
    time_win : int
        ...
    p_regrid_ref : str
        Path of the NetCDF file containing data for the reference period.
    p_regrid_fut : str
        Path of the NetCDF file containing data for the future period.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Load datasets.
    ds_ref = xr.open_dataset(p_regrid_ref)[var]
    ds_fut = xr.open_dataset(p_regrid_fut)[var]
    if (var == cfg.var_cordex_tas) or (var == cfg.var_cordex_tasmin) or (var == cfg.var_cordex_tasmax):
        ds_ref = ds_ref - 273.15
        ds_fut = ds_fut - 273.15

    # Fit.
    x     = [*range(len(ds_ref.time))]
    y     = ds_ref.values
    coefs = poly.polyfit(x, y, 4)
    ffit  = poly.polyval(x, coefs)

    # Initialize plot.
    fs_sup_title = 8
    fs_title     = 8
    fs_legend    = 6
    fs_axes      = 8
    f = plt.figure(figsize=(15, 6))
    f.add_subplot(211)
    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.99, hspace=0.40, wspace=0.00)
    sup_title = os.path.basename(p_fig).replace(".png", "") +\
                "_nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
    plt.suptitle(sup_title, fontsize=fs_sup_title)

    # Convert date format if the need is.
    if ds_ref.time.dtype == cfg.dtype_obj:
        ds_ref["time"] = utils.fix_calendar(ds_ref)

    # Upper plot: Reference period.
    f.add_subplot(211)
    plt.plot(ds_ref.time, y, color=cfg.col_sim_ref)
    plt.plot(ds_ref.time, ffit, color="black")
    plt.legend(["Simulation (pér. référence)", "Tendance"], fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)
    plt.title("Tendance", fontsize=fs_title)

    # Lower plot: Complete simulation.
    f.add_subplot(212)
    arr_y_detrend = signal.detrend(ds_fut)
    arr_x_detrend = cfg.per_ref[0] + np.arange(0, len(arr_y_detrend), 1) / 365
    arr_y_error  = (y - ffit)
    arr_x_error = cfg.per_ref[0] + np.arange(0, len(arr_y_error), 1) / 365
    plt.plot(arr_x_detrend, arr_y_detrend, alpha=0.5, color=cfg.col_sim_fut)
    plt.plot(arr_x_error, arr_y_error, alpha=0.5, color=cfg.col_sim_ref)
    plt.legend(["Simulation", "Simulation (prédiction), pér. référence)"],
               fontsize=fs_legend, frameon=False)
    plt.xlabel("Jours", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)
    plt.title("Variation autour de la moyenne (prédiction basée sur une équation quartique)", fontsize=fs_title)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_calib(ds_qmf, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, var, sup_title, p_fig):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot containing a summary of calibration.

    Parameters
    ----------
    ds_qmf : xr.Dataset
        Dataset of quantile mapping function.
    ds_qqmap_ref : xr.Dataset
        Dataset of adjusted simulation for the referenc period.
    ds_obs : xr.Dataset
        Dataset of observations.
    ds_ref : xr.Dataset
        Dataset of simulation for the reference period.
    ds_fut : xr.Dataset
        Dataset of simulation for the future period.
    ds_qqmap : xr.Dataset
        Dataset of adjusted simulation.
    var : str
        Variable.
    sup_title : str
        Title of figure.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Quantile ---------------------------------------------------------------------------------------------------------

    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 4
    fs_axes      = 7

    f = plt.figure(figsize=(9, 9))
    f.add_subplot(431)
    plt.subplots_adjust(top=0.930, bottom=0.065, left=0.070, right=0.973, hspace=0.750, wspace=0.250)

    ds_qmf.plot()
    plt.xlabel("Quantile", fontsize=fs_axes)
    plt.ylabel("Jour de l'année", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    legend_items = ["Sim. ajustée (pér. réf.)", "Sim. (pér. réf.)", "Observations",
                    "Sim. ajustée", "Sim."]

    # Mean values ------------------------------------------------------------------------------------------------------

    # Plot.
    f.add_subplot(432)
    if var == cfg.var_cordex_pr:
        draw_curves(var, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, "sum")
    else:
        draw_curves(var, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, "mean")
    plt.title("Moyenne", fontsize=fs_title)
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlim([1, 12])
    plt.xticks(np.arange(1, 13, 1))
    plt.xlabel("Mois", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    # Mean, Q100, Q99, Q75, Q50, Q25, Q01 and Q00 monthly values -------------------------------------------------------

    for i in range(1, 8):

        plt.subplot(433 + i - 1)

        title    = ""
        quantile = -1
        if i == 1:
            title    = "Q100"
            stat     = "max"
        elif i == 2:
            title    = "Q99"
            stat     = "quantile"
            quantile = 0.99
        elif i == 3:
            title    = "Q75"
            stat     = "quantile"
            quantile = 0.75
        elif i == 4:
            title    = "Q50"
            stat     = "quantile"
            quantile = 0.50
        elif i == 5:
            title    = "Q25"
            stat     = "quantile"
            quantile = 0.25
        elif i == 6:
            title    = "Q1"
            stat     = "quantile"
            quantile = 0.01
        elif i == 7:
            title    = "Q0"
            stat     = "min"

        draw_curves(var, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, stat, quantile)

        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.legend(legend_items, fontsize=fs_legend, frameon=False)
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)

    # Time series ------------------------------------------------------------------------------------------------------

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd

    # Convert date format if the need is.
    if ds_qqmap.time.time.dtype == cfg.dtype_obj:
        ds_qqmap["time"] = utils.fix_calendar(ds_qqmap.time)
    if ds_ref.time.time.dtype == cfg.dtype_obj:
        ds_ref["time"] = utils.fix_calendar(ds_ref.time)

    plt.subplot(313)
    (ds_qqmap * coef).plot.line(alpha=0.5, color=cfg.col_sim_adj)
    (ds_ref * coef).plot.line(alpha=0.5, color=cfg.col_sim_ref)
    (ds_obs * coef).plot.line(alpha=0.5, color=cfg.col_obs)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.legend(["Sim. ajustée", "Sim. (pér. référence)", "Observations"], fontsize=fs_legend, frameon=False)
    plt.title("")

    f.suptitle(var + "_" + sup_title, fontsize=fs_sup_title)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    if "bias_corrected" in ds_qqmap.attrs:
        del ds_qqmap.attrs["bias_corrected"]

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_calib_ts(ds_obs, ds_fut, ds_qqmap, var, title, p_fig):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of observed and future periods.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Dataset of observations.
    ds_fut : xr.Dataset
        Dataset of simulation for the future period.
    ds_qqmap : xr.Dataset
        Dataset of adjusted simulation.
    var : str
        Variable.
    title : str
        Title of figure.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert date format if the need is.
    if ds_qqmap.time.dtype == cfg.dtype_obj:
        ds_qqmap["time"] = utils.fix_calendar(ds_qqmap)
    if ds_fut.time.dtype == cfg.dtype_obj:
        ds_fut["time"] = utils.fix_calendar(ds_fut)
    if ds_obs.time.dtype == cfg.dtype_obj:
        ds_obs["time"] = utils.fix_calendar(ds_obs)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd

    fs_sup_title = 8
    fs_legend = 8
    fs_axes = 8
    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.21, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

    # Add curves.
    (ds_qqmap * coef).plot.line(alpha=0.5, color=cfg.col_sim_adj)
    (ds_fut * coef).plot.line(alpha=0.5, color=cfg.col_sim_fut)
    (ds_obs * coef).plot.line(alpha=0.5, color=cfg.col_obs)

    # Customize.
    plt.legend(["Sim. ajustée", "Sim. (pér. référence)", "Observations"], fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_sup_title)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def draw_curves(var, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, stat, quantile=-1.0):

    """
    --------------------------------------------------------------------------------------------------------------------
    Draw curves.

    Parameters
    ----------
    var : str
        Weather variable.
    ds_qqmap_ref : xr.Dataset
        Dataset of the adjusted simulation for the reference period.
    ds_obs : xr.Dataset
        Dataset of observations.
    ds_ref : xr.Dataset
        Dataset of simulation for the reference period.
    ds_fut : xr.Dataset
        Dataset of simulation for the future period.
    ds_qqmap : xr.Dataset
        Dataset of the adjusted simulation.
    stat : {"max", "quantile", "mean", "sum"}
        Statistic.
    quantile : float, optional
        Quantile.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd

    # Draw curves.
    if stat == "min":
        (ds_qqmap_ref * coef).groupby(ds_qqmap_ref.time.dt.month).min().plot.line(color=cfg.col_sim_adj_ref)
        (ds_ref * coef).groupby(ds_ref.time.dt.month).min().plot.line(color=cfg.col_sim_ref)
        (ds_obs * coef).groupby(ds_obs.time.dt.month).min().plot.line(color=cfg.col_obs)
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).min().plot.line(color=cfg.col_sim_adj)
        (ds_fut * coef).groupby(ds_fut.time.dt.month).min().plot.line(color=cfg.col_sim_fut)
    elif stat == "max":
        (ds_qqmap_ref * coef).groupby(ds_qqmap_ref.time.dt.month).max().plot.line(color=cfg.col_sim_adj_ref)
        (ds_ref * coef).groupby(ds_ref.time.dt.month).max().plot.line(color=cfg.col_sim_ref)
        (ds_obs * coef).groupby(ds_obs.time.dt.month).max().plot.line(color=cfg.col_obs)
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).max().plot.line(color=cfg.col_sim_adj)
        (ds_fut * coef).groupby(ds_fut.time.dt.month).max().plot.line(color=cfg.col_sim_fut)
    elif stat == "quantile":
        (ds_qqmap_ref * coef).groupby(ds_qqmap_ref.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_adj_ref)
        (ds_ref * coef).groupby(ds_ref.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_ref)
        (ds_obs * coef).groupby(ds_obs.time.dt.month).quantile(quantile).plot.line(color=cfg.col_obs)
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_adj)
        (ds_fut * coef).groupby(ds_fut.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_fut)
    elif stat == "mean":
        (ds_qqmap_ref * coef).groupby(ds_qqmap_ref.time.dt.month).mean().plot.line(color=cfg.col_sim_adj_ref)
        (ds_ref * coef).groupby(ds_ref.time.dt.month).mean().plot.line(color=cfg.col_sim_ref)
        (ds_obs * coef).groupby(ds_obs.time.dt.month).mean().plot.line(color=cfg.col_obs)
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).mean().plot.line(color=cfg.col_sim_adj)
        (ds_fut * coef).groupby(ds_fut.time.dt.month).mean().plot.line(color=cfg.col_sim_fut)
    elif stat == "sum":
        (ds_qqmap_ref * coef).groupby(ds_qqmap_ref.time.dt.month).sum().plot.line(color=cfg.col_sim_adj_ref)
        (ds_ref * coef).groupby(ds_ref.time.dt.month).sum().plot.line(color=cfg.col_sim_ref)
        (ds_obs * coef).groupby(ds_obs.time.dt.month).sum().plot.line(color=cfg.col_obs)
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).sum().plot.line(color=cfg.col_sim_adj)
        (ds_fut * coef).groupby(ds_fut.time.dt.month).sum().plot.line(color=cfg.col_sim_fut)


def plot_idx_ts(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn, idx_name, idx_threshs, rcps, xlim, p_fig):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a time series of a climate index for the reference period and for emission scenarios.

    Parameters
    ----------
    ds_ref : xr.Dataset
        Dataset for the reference period.
    ds_rcp_26 : xr.Dataset
        Dataset for RCP 2.6.
    ds_rcp_45 : xr.Dataset
        Dataset for RCP 4.5.
    ds_rcp_85 : xr.Dataset
        Dataset for RCP 8.5.
    stn : str
        Station name.
    idx_name : str
        Index name.
    idx_threshs : float[]
        Threshold value associated with 'var'.
    rcps : [str]
        Emission scenarios.
    xlim : [int]
        Minimum and maximum values along the x-axis.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Variable-specific treatment.
    title = ""
    if idx_name == cfg.idx_tx_days_above:
        title = "Nombre de jours avec " + cfg.get_var_desc(cfg.var_cordex_tasmax).lower() + " > " +\
                str(idx_threshs[0]) + " " + cfg.get_var_unit(cfg.var_cordex_tasmax) + " (" + stn + ")"

    # Initialize plot.
    f, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Année')
    ax.secondary_yaxis('right')
    ax.get_yaxis().tick_right()
    ax.axes.get_yaxis().set_visible(False)
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel("Nombre de jours")
    plt.subplots_adjust(top=0.925, bottom=0.10, left=0.03, right=0.90, hspace=0.30, wspace=0.416)

    # Update plot.
    ds_ref_curve = None
    ds_ref_min = None
    ds_ref_max = None
    ds_fut_curve = None
    ds_fut_min = None
    ds_fut_max = None
    for rcp in rcps:

        color = "black"
        if rcp == "ref":
            ds_ref_curve = ds_ref[0]
            ds_ref_min   = ds_ref[1]
            ds_ref_max   = ds_ref[2]
        elif rcp == cfg.rcp_26:
            ds_fut_curve = ds_rcp_26[0]
            ds_fut_min   = ds_rcp_26[1]
            ds_fut_max   = ds_rcp_26[2]
            color = "blue"
        elif rcp == cfg.rcp_45:
            ds_fut_curve = ds_rcp_45[0]
            ds_fut_min   = ds_rcp_45[1]
            ds_fut_max   = ds_rcp_45[2]
            color = "green"
        elif rcp == cfg.rcp_85:
            ds_fut_curve = ds_rcp_85[0]
            ds_fut_min   = ds_rcp_85[1]
            ds_fut_max   = ds_rcp_85[2]
            color = "red"

        if rcp == "ref":
            ax.plot(ds_ref_max["time"], ds_ref_curve, color=color, alpha=1.0)
        else:
            ax.fill_between(ds_ref_max["time"], ds_ref_min, ds_ref_max, color="grey", alpha=0.25)
            ax.plot(ds_fut_max["time"], ds_fut_curve, color=color, alpha=1.0)
            ax.fill_between(ds_fut_max["time"], ds_fut_min, ds_fut_max, color=color, alpha=0.25)

    # Finalize plot.
    legend_list = ["Référence"]
    if cfg.rcp_26 in rcps:
        legend_list.append("RCP 2,6")
    if cfg.rcp_45 in rcps:
        legend_list.append("RCP 4,5")
    if cfg.rcp_85 in rcps:
        legend_list.append("RCP 8,5")
    ax.legend(legend_list, loc="upper left", frameon=False)
    plt.xlim(xlim[0] * 365, xlim[1] * 365)
    plt.ylim(bottom=0)

    # Save figure.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    plt.close()


def plot_idx_heatmap(ds, idx_name, idx_threshs, grid_x, grid_y, per, p_fig, map_package):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a heat map of a climate index for the reference period and for emission scenarios.
    TODO: Add a color scale that is common to all horizons.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset (with 2 dimensions: longitude and latitude).
    idx_name : str
        Index name.
    idx_threshs : float[]
        Threshold value associated with 'var'.
    grid_x: [float]
        X-coordinates.
    grid_y: [float]
        Y-coordinates.
    per: [int, int]
        Period of interest, for instance, [1981, 2010].
    p_fig : str
        Path of output figure.
    map_package: str
        Map package: {"seaborn", "matplotlib"}
    --------------------------------------------------------------------------------------------------------------------
    """

    title = ""
    label = ""
    if idx_name == cfg.idx_tx_days_above:
        title = "Nombre de jours avec " + cfg.get_var_desc(cfg.var_cordex_tasmax).lower() + " > " +\
                str(idx_threshs[0]) + " " + cfg.get_var_unit(cfg.var_cordex_tasmax) +\
                " (" + cfg.country.capitalize() + ", " + str(per[0]) + "-" + str(per[1]) + ")"
        label = "Nombre de jours"
    plt.subplots_adjust(top=0.9, bottom=0.11, left=0.12, right=0.995, hspace=0.695, wspace=0.416)

    # Using seaborn.
    if map_package == "seaborn":
        sns.set()
        fig, ax = plt.subplots(figsize=(8, 5))
        g = sns.heatmap(ax=ax, data=ds, xticklabels=grid_x, yticklabels=grid_y)
        x_labels = ['{:,.2f}'.format(i) for i in grid_x]
        y_labels = ['{:,.2f}'.format(i) for i in grid_y]
        g.set_xticklabels(x_labels)
        g.set_yticklabels(y_labels)

    # Using matplotlib.
    elif map_package == "matplotlib":
        fs = 10
        ds.plot.pcolormesh(add_colorbar=True, add_labels=True,
                           cbar_kwargs=dict(orientation='vertical', pad=0.05, shrink=1, label=label))
        plt.title(title)
        plt.suptitle("", fontsize=fs)
        plt.xlabel("Longitude (º)", fontsize=fs)
        plt.ylabel("Latitude (º)", fontsize=fs)
        plt.tick_params(axis="x", labelsize=fs)
        plt.tick_params(axis="y", labelsize=fs)

    # Save figure.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    plt.close()


def plot_360_vs_365(ds_360, ds_365, var=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Compare a 360- vs. 365-day calendars.

    Parameters
    ----------
    ds_360 : xr.Dataset
        Dataset.
    ds_365 : xr.Dataset
        Dataset.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    if var != "":
        plt.plot((np.arange(1, 361) / 360) * 365, ds_360[var][:360].values)
        plt.plot(np.arange(1, 366), ds_365[var].values[:365], alpha=0.5)
    else:
        plt.plot((np.arange(1, 361) / 360) * 365, ds_360[:360].values)
        plt.plot(np.arange(1, 366), ds_365[:365].values, alpha=0.5)
    plt.close()

def plot_rsq(rsq, n_sim):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a plot of root mean square error.

    Parameters
    ----------
    rsq : np
        Numpy array.
    n_sim : int
        Number of simulations
    --------------------------------------------------------------------------------------------------------------------
    """

    plt.figure()
    plt.plot(range(1, n_sim + 1), rsq, "k", label="R²")
    plt.plot(np.arange(1.5, n_sim + 0.5), np.diff(rsq), "r", label="ΔR²")
    axes = plt.gca()
    axes.set_xlim([0, n_sim])
    axes.set_ylim([0, 1])
    plt.xlabel("Number of groups")
    plt.ylabel("R² / ΔR²")
    plt.legend(loc="center right")
    plt.title("R² of groups vs. full ensemble")
    plt.close()
