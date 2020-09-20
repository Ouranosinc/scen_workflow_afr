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
import glob
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.polynomial.polynomial as poly
import os.path
import pandas as pd
import seaborn as sns
import statistics
import utils
import warnings
import xarray as xr
import xclim.subset as subset
from matplotlib.lines import Line2D
from scipy import signal
from scipy.interpolate import griddata


# ======================================================================================================================
# Aggregation
# ======================================================================================================================

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


# ======================================================================================================================
# Scenarios
# ======================================================================================================================


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
    delta = 0
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
        coef = cfg.spd * 365
    elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        delta = -273.15

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
        (ds_qqmap * coef + delta).groupby(ds_qqmap.time.dt.year).mean().plot.line(color=cfg.col_sim_adj)
    (ds_fut * coef + delta).groupby(ds_fut.time.dt.year).mean().plot.line(color=cfg.col_sim_fut)
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

    # Conversion coefficients.
    coef = 1
    delta = 0
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
        coef = cfg.spd
    if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        delta = -273.15

    # Fit.
    x     = [*range(len(ds_ref.time))]
    y     = (ds_ref * coef + delta).values
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
        ds_ref["time"] = utils.reset_calendar(ds_ref)

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
    arr_y_detrend = signal.detrend(ds_fut * coef + delta)
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


# ======================================================================================================================
# Calibration
# ======================================================================================================================


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
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
        draw_curves(var, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, cfg.stat_sum)
    else:
        draw_curves(var, ds_qqmap_ref, ds_obs, ds_ref, ds_fut, ds_qqmap, cfg.stat_mean)
    plt.title("Moyenne", fontsize=fs_title)
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlim([1, 12])
    plt.xticks(np.arange(1, 13, 1))
    plt.xlabel("Mois", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    # Mean, Q100, Q99, Q75, Q50, Q25, Q01 and Q00 monthly values -------------------------------------------------------

    # Conversion coefficient.
    coef = 1
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
        coef = 365

    for i in range(1, len(cfg.stat_quantiles) + 1):

        plt.subplot(433 + i - 1)

        stat     = "quantile"
        quantile = cfg.stat_quantiles[i-1]
        title    = "Q_" + str(quantile)
        if quantile == 0:
            stat = cfg.stat_min
        elif quantile == 1:
            stat = cfg.stat_max

        draw_curves(var, ds_qqmap_ref * coef, ds_obs * coef, ds_ref * coef, ds_fut * coef, ds_qqmap * coef, stat,
                    quantile)

        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.legend(legend_items, fontsize=fs_legend, frameon=False)
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)

    # Time series ------------------------------------------------------------------------------------------------------

    # Convert date format if the need is.
    if ds_qqmap.time.time.dtype == cfg.dtype_obj:
        ds_qqmap["time"] = utils.reset_calendar(ds_qqmap.time)
    if ds_ref.time.time.dtype == cfg.dtype_obj:
        ds_ref["time"] = utils.reset_calendar(ds_ref.time)

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
        ds_qqmap["time"] = utils.reset_calendar(ds_qqmap)
    if ds_fut.time.dtype == cfg.dtype_obj:
        ds_fut["time"] = utils.reset_calendar(ds_fut)
    if ds_obs.time.dtype == cfg.dtype_obj:
        ds_obs["time"] = utils.reset_calendar(ds_obs)

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
    stat : {cfg.stat_max, cfg.stat_quantile, cfg.stat_mean, cfg.stat_sum}
        Statistic.
    quantile : float, optional
        Quantile.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Draw curves.
    if stat == cfg.stat_min:
        ds_qqmap_ref.groupby(ds_qqmap_ref.time.dt.month).min().plot.line(color=cfg.col_sim_adj_ref)
        ds_ref.groupby(ds_ref.time.dt.month).min().plot.line(color=cfg.col_sim_ref)
        ds_obs.groupby(ds_obs.time.dt.month).min().plot.line(color=cfg.col_obs)
        ds_qqmap.groupby(ds_qqmap.time.dt.month).min().plot.line(color=cfg.col_sim_adj)
        ds_fut.groupby(ds_fut.time.dt.month).min().plot.line(color=cfg.col_sim_fut)
    elif stat == cfg.stat_max:
        ds_qqmap_ref.groupby(ds_qqmap_ref.time.dt.month).max().plot.line(color=cfg.col_sim_adj_ref)
        ds_ref.groupby(ds_ref.time.dt.month).max().plot.line(color=cfg.col_sim_ref)
        ds_obs.groupby(ds_obs.time.dt.month).max().plot.line(color=cfg.col_obs)
        ds_qqmap.groupby(ds_qqmap.time.dt.month).max().plot.line(color=cfg.col_sim_adj)
        ds_fut.groupby(ds_fut.time.dt.month).max().plot.line(color=cfg.col_sim_fut)
    elif stat == cfg.stat_quantile:
        ds_qqmap_ref.groupby(ds_qqmap_ref.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_adj_ref)
        ds_ref.groupby(ds_ref.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_ref)
        ds_obs.groupby(ds_obs.time.dt.month).quantile(quantile).plot.line(color=cfg.col_obs)
        ds_qqmap.groupby(ds_qqmap.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_adj)
        ds_fut.groupby(ds_fut.time.dt.month).quantile(quantile).plot.line(color=cfg.col_sim_fut)
    elif stat == cfg.stat_mean:
        ds_qqmap_ref.groupby(ds_qqmap_ref.time.dt.month).mean().plot.line(color=cfg.col_sim_adj_ref)
        ds_ref.groupby(ds_ref.time.dt.month).mean().plot.line(color=cfg.col_sim_ref)
        ds_obs.groupby(ds_obs.time.dt.month).mean().plot.line(color=cfg.col_obs)
        ds_qqmap.groupby(ds_qqmap.time.dt.month).mean().plot.line(color=cfg.col_sim_adj)
        ds_fut.groupby(ds_fut.time.dt.month).mean().plot.line(color=cfg.col_sim_fut)
    elif stat == cfg.stat_sum:
        ds_qqmap_ref.groupby(ds_qqmap_ref.time.dt.month).sum().plot.line(color=cfg.col_sim_adj_ref)
        ds_ref.groupby(ds_ref.time.dt.month).sum().plot.line(color=cfg.col_sim_ref)
        ds_obs.groupby(ds_obs.time.dt.month).sum().plot.line(color=cfg.col_obs)
        ds_qqmap.groupby(ds_qqmap.time.dt.month).sum().plot.line(color=cfg.col_sim_adj)
        ds_fut.groupby(ds_fut.time.dt.month).sum().plot.line(color=cfg.col_sim_fut)


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


# ======================================================================================================================
# Scenarios and indices.
# ======================================================================================================================


def plot_heatmap(var_or_idx, threshs, rcp, per_hors):

    """
    --------------------------------------------------------------------------------------------------------------------
    Interpolate data within a zone.

    Parameters
    ----------
    var_or_idx: str
        Climate variable (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).
    threshs: [float]
        Climate index thresholds.
    rcp: str
        Emission scenario.
    per_hors: [[int]]
        Horizons.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine category.
    if var_or_idx in cfg.variables_cordex:
        cat = cfg.cat_scen
    else:
        cat = cfg.cat_idx

    utils.log("-", True)
    if cat == cfg.cat_scen:
        utils.log("Scenario          : " + var_or_idx, True)
    else:
        utils.log("Index             : " + var_or_idx, True)
    utils.log("Emission scenario : " + cfg.get_rcp_desc(rcp), True)
    utils.log("-", True)

    # Number of years and stations.
    if rcp == cfg.rcp_ref:
        n_year = cfg.per_ref[1] - cfg.per_ref[0] + 1
    else:
        n_year = cfg.per_fut[1] - cfg.per_ref[1] + 1

    # Get information on stations.
    # TODO.YR: Coordinates should be embedded into ds_stat below.
    p_stn = glob.glob(cfg.get_d_stn(cfg.var_cordex_tas) + "../*.csv")[0]
    df = pd.read_csv(p_stn, sep=cfg.file_sep)

    # Collect values for each station and determine overall boundaries.
    utils.log("Collecting emissions scenarios at each station.", True)
    x_bnds = []
    y_bnds = []
    data_stn = []
    for stn in cfg.stns:

        # Get coordinates.
        lon = df[df["station"] == stn]["lon"].values[0]
        lat = df[df["station"] == stn]["lat"].values[0]

        # Calculate statistics.
        if rcp == cfg.rcp_ref:
            ds_stat = statistics.calc_stat(cfg.cat_obs, cfg.freq_YS, cfg.freq_YS, stn, var_or_idx, rcp, None,
                                           cfg.stat_mean)
        else:
            ds_stat = statistics.calc_stat(cfg.cat_sim, cfg.freq_YS, cfg.freq_YS, stn, var_or_idx, rcp, None,
                                           cfg.stat_mean)
        if ds_stat is None:
            continue

        # Extract data from stations.
        data = [[], [], []]
        n = ds_stat.dims["time"]
        for year in range(0, n):

            # Collect data.
            x = float(lon)
            y = float(lat)
            z = float(ds_stat[var_or_idx][0][0][year])
            if math.isnan(z):
                z = float(0)
            data[0].append(x)
            data[1].append(y)
            data[2].append(z)

            # Update overall boundaries (round according to the variable 'step').
            if not x_bnds:
                x_bnds = [x, x]
                y_bnds = [y, y]
            else:
                x_bnds = [min(x_bnds[0], x), max(x_bnds[1], x)]
                y_bnds = [min(y_bnds[0], y), max(y_bnds[1], y)]

        # Add data from station to complete dataset.
        data_stn.append(data)

    # Build the list of x and y locations for which interpolation is needed.
    utils.log("Collecting the coordinates of stations.", True)
    grid_time = range(0, n_year)

    def round_to_nearest_decimal(val, step):
        if val < 0:
            val_rnd = math.floor(val/step) * step
        else:
            val_rnd = math.ceil(val/step) * step
        return val_rnd

    for i in range(0, 2):
        x_bnds[i] = round_to_nearest_decimal(x_bnds[i], cfg.idx_resol)
        y_bnds[i] = round_to_nearest_decimal(y_bnds[i], cfg.idx_resol)
    grid_x = np.arange(x_bnds[0], x_bnds[1] + cfg.idx_resol, cfg.idx_resol)
    grid_y = np.arange(y_bnds[0], y_bnds[1] + cfg.idx_resol, cfg.idx_resol)

    # Perform interpolation.
    # There is a certain flexibility regarding the number of years in a dataset. Ideally, the station should not have
    # been considered in the analysis, unless there is no better option.
    utils.log("Performing interpolation.", True)
    new_grid = np.meshgrid(grid_x, grid_y)
    new_grid_data = np.empty((n_year, len(grid_y), len(grid_x)))
    for i_year in range(0, n_year):
        arr_x = []
        arr_y = []
        arr_z = []
        for i_stn in range(len(data_stn)):
            if i_year < len(data_stn[i_stn][0]):
                arr_x.append(data_stn[i_stn][0][i_year])
                arr_y.append(data_stn[i_stn][1][i_year])
                arr_z.append(data_stn[i_stn][2][i_year])
        new_grid_data[i_year, :, :] = griddata((arr_x, arr_y), arr_z, (new_grid[0], new_grid[1]), fill_value=np.nan,
                                               method="linear")
    da_idx = xr.DataArray(new_grid_data,
                          coords={"time": grid_time, "lat": grid_y, "lon": grid_x}, dims=["time", "lat", "lon"])
    ds_idx = da_idx.to_dataset(name=var_or_idx)

    # Clip to country boundaries.
    # TODO: Clipping is no longer working when launching the script from a terminal.
    if cfg.d_bounds != "":
        try:
            ds_idx = subset.subset_shape(ds_idx, cfg.d_bounds)
        except TypeError:
            utils.log("Unable to use a mask.", True)

    # Loop through horizons.
    if cfg.opt_plot:
        utils.log("Generating maps.", True)
        for per_hor in per_hors:

            # Select years.
            if rcp == cfg.rcp_ref:
                year_1 = 0
                year_n = cfg.per_ref[1] - cfg.per_ref[0]
            else:
                year_1 = per_hor[0] - cfg.per_ref[1]
                year_n = per_hor[1] - cfg.per_ref[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ds_hor = ds_idx[var_or_idx][year_1:(year_n+1)][:][:].mean("time", skipna=True)

            # Conversion coefficient.
            coef = 1
            if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
                coef = cfg.spd

            # Plot.
            p_fig = cfg.get_d_sim("", cfg.cat_fig + "/" + cat, "") +\
                var_or_idx + "_" + rcp + "_" + str(per_hor[0]) + "_" + str(per_hor[1]) + ".png"
            plot_heatmap_spec(ds_hor * coef, var_or_idx, threshs, grid_x, grid_y, per_hor, p_fig, "matplotlib")


def plot_heatmap_spec(ds, var_or_idx, threshs, grid_x, grid_y, per, p_fig, map_package):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a heat map of a climate index for the reference period and for emission scenarios.
    TODO: Add a color scale that is common to all horizons.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset (with 2 dimensions: longitude and latitude).
    var_or_idx : str
        Climate variable (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).
    threshs : float[]
        Threshold value associated a climate index.
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

    # ==========================================================
    # TODO.CUSTOMIZATION.BEGIN
    # When adding a new climate index, calculate the index by
    # copying the following code block.
    # ==========================================================

    if var_or_idx == cfg.idx_tx_days_above:
        title = cfg.get_idx_desc(var_or_idx, threshs) + cfg.get_var_unit(cfg.var_cordex_tasmax)
        label = "Nombre de jours"

    # ==========================================================
    # TODO.CUSTOMIZATION.END
    # ==========================================================

    else:
        title = cfg.get_var_desc(var_or_idx).capitalize()
        label = title + " (" + cfg.get_var_unit(var_or_idx) + ")"
    title = title + " (" + cfg.country.capitalize() + ", " + str(per[0]) + "-" + str(per[1]) + ")"

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


def plot_ts(var_or_idx, threshs=[]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Plot time series for individual simulations.

    Parameters
    ----------
    var_or_idx : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).
    threshs : [float], optional
        Threshold values associated with a climate index.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Emission scenarios.
    rcps = cfg.rcps + [cfg.rcp_ref]

    # Determine category.
    cat = cfg.cat_scen if var_or_idx in cfg.variables_cordex else cfg.cat_idx

    # minimum and maximum values along the y-axis
    ylim = []

    # Loop through stations.
    for stn in cfg.stns:

        # Loop through emission scenarios.
        ds_ref = None
        ds_rcp_26, ds_rcp_26_grp = [], []
        ds_rcp_45, ds_rcp_45_grp = [], []
        ds_rcp_85, ds_rcp_85_grp = [], []
        for rcp in rcps:

            # List files.
            if rcp == cfg.rcp_ref:
                p_sim_list = [cfg.get_p_obs(stn, var_or_idx)]
            else:
                if var_or_idx in cfg.variables_cordex:
                    d = cfg.get_d_sim(stn, cfg.cat_qqmap, var_or_idx)
                else:
                    d = cfg.get_d_sim(stn, cfg.cat_idx, var_or_idx)
                p_sim_list = glob.glob(d + "*_" + rcp + ".nc")

            # Exit if there is not file corresponding to the criteria.
            if (len(p_sim_list) == 0) or \
               ((len(p_sim_list) > 0) and not(os.path.isdir(os.path.dirname(p_sim_list[0])))):
                continue

            # Loop through simulation files.
            for i_sim in range(len(p_sim_list)):

                # Load dataset.
                ds = xr.open_dataset(p_sim_list[i_sim])

                # First and last years.
                year_1 = int(str(ds.time.values[0])[0:4])
                year_n = int(str(ds.time.values[len(ds.time.values) - 1])[0:4])
                if rcp == cfg.rcp_ref:
                    year_1 = max(year_1, cfg.per_ref[0])
                    year_n = min(year_n, cfg.per_ref[1])
                else:
                    year_1 = max(year_1, cfg.per_ref[0])
                    year_n = min(year_n, cfg.per_fut[1])

                # Select years.
                years_str = [str(year_1) + "-01-01", str(year_n) + "-12-31"]
                ds = ds.sel(time=slice(years_str[0], years_str[1]))
                units = ds[var_or_idx].attrs["units"] if cat == cfg.cat_scen else ds["units"]
                if units == "degree_C":
                    units = "C"

                # Calculate statistics.
                # TODO: Include coordinates in the generated dataset.
                years = ds.groupby(ds.time.dt.year).groups.keys()
                if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
                    ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
                else:
                    ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
                if "lon" in ds.dims:
                    ds = ds.isel(lon=0, lat=0)
                n_time = len(ds["time"].values)
                da = xr.DataArray(np.array(ds[var_or_idx].values), name=var_or_idx,
                                  coords=[("time", np.arange(n_time))])
                ds = da.to_dataset()
                ds["time"] = utils.reset_calendar_list(years)
                ds[var_or_idx].attrs["units"] = units

                # Convert units.
                if var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                    if ds[var_or_idx].attrs["units"] == "K":
                        ds = ds - 273.15
                        ds[var_or_idx].attrs["units"] = "C"
                elif var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpt]:
                    if ds[var_or_idx].attrs["units"] == "kg m-2 s-1":
                        ds = ds * cfg.spd
                        ds[var_or_idx].attrs["units"] = "mm"

                # Calculate minimum and maximum values along the y-axis.
                if not ylim:
                    ylim = [min(ds[var_or_idx].values), max(ds[var_or_idx].values)]
                else:
                    ylim = [min(ylim[0], min(ds[var_or_idx].values)),
                            max(ylim[1], max(ds[var_or_idx].values))]

                # Append to list of datasets.
                if rcp == cfg.rcp_ref:
                    ds_ref = ds
                elif rcp == cfg.rcp_26:
                    ds_rcp_26.append(ds)
                elif rcp == cfg.rcp_45:
                    ds_rcp_45.append(ds)
                elif rcp == cfg.rcp_85:
                    ds_rcp_85.append(ds)

            # Group by RCP.
            if rcp != cfg.rcp_ref:
                if rcp == cfg.rcp_26:
                    ds_rcp_26_grp = calc_stat_mean_min_max(ds_rcp_26, var_or_idx)
                elif rcp == cfg.rcp_45:
                    ds_rcp_45_grp = calc_stat_mean_min_max(ds_rcp_45, var_or_idx)
                elif rcp == cfg.rcp_85:
                    ds_rcp_85_grp = calc_stat_mean_min_max(ds_rcp_85, var_or_idx)

        # Generate plots.
        if cfg.opt_plot and (ds_ref is not None) or (ds_rcp_26 != []) or (ds_rcp_45 != []) or (ds_rcp_85 != []):
            msg = "scenarios" if cat == cfg.cat_scen else "indices"
            utils.log("Generating time series of " + msg + ".", True)
            p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cat, "") + var_or_idx + "_" + stn + "_rcp.png"
            plot_ts_spec(ds_ref, ds_rcp_26_grp, ds_rcp_45_grp, ds_rcp_85_grp, stn.capitalize(), var_or_idx, threshs,
                         rcps, ylim, p_fig, 1)
            p_fig = p_fig.replace("_rcp.png", "_sim.png")
            plot_ts_spec(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(), var_or_idx, threshs,
                         rcps, ylim, p_fig, 2)


def calc_stat_mean_min_max(ds_list, var_or_idx):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate mean, minimum and maximum values within a group of datasets.
    TODO: Include coordinates in the returned datasets.

    Parameters
    ----------
    ds_list : [xr.Dataset]
        Array of datasets from a given group.
    var_or_idx : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).

    Returns
    -------
    ds_mean_min_max : [xr.Dataset]
        Array of datasets with mean, minimum and maximum values.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_mean_min_max = []

    # Get years, units and coordinates.
    units = ds_list[0][var_or_idx].attrs["units"]
    year_1 = int(str(ds_list[0].time.values[0])[0:4])
    year_n = int(str(ds_list[0].time.values[len(ds_list[0].time.values) - 1])[0:4])
    n_time = year_n - year_1 + 1

    # Calculate statistics.
    arr_vals_mean = []
    arr_vals_min = []
    arr_vals_max = []
    for i_time in range(n_time):
        vals = []
        for ds in ds_list:
            vals.append(float(ds[var_or_idx][i_time].values))
        arr_vals_mean.append(np.array(vals).mean())
        arr_vals_min.append(np.array(vals).min())
        arr_vals_max.append(np.array(vals).max())

    # Build datasets.
    for i in range(1, 4):

        # Select values.
        if i == 1:
            arr_vals = arr_vals_mean
        elif i == 2:
            arr_vals = arr_vals_min
        else:
            arr_vals = arr_vals_max

        # Build dataset.
        da = xr.DataArray(np.array(arr_vals), name=var_or_idx, coords=[("time", np.arange(n_time))])
        ds = da.to_dataset()
        ds["time"] = utils.reset_calendar(ds, year_1, year_n, cfg.freq_YS)
        ds[var_or_idx].attrs["units"] = units

        ds_mean_min_max.append(ds)

    return ds_mean_min_max


def plot_ts_spec(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn, var_or_idx, threshs, rcps, ylim, p_fig, mode=1):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a time series of a climate variable or index, combining all emission scenarios.

    Parameters
    ----------
    ds_ref : xr.Dataset
        Dataset for the reference period.
    ds_rcp_26 : [xr.Dataset]
        Dataset for RCP 2.6.
    ds_rcp_45 : [xr.Dataset]
        Dataset for RCP 4.5.
    ds_rcp_85 : [xr.Dataset]
        Dataset for RCP 8.5.
    stn : str
        Station name.
    var_or_idx : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).
    threshs : float[]
        Threshold values associated with a climate index.
    rcps : [str]
        Emission scenarios.
    ylim : [int]
        Minimum and maximum values along the y-axis.
    p_fig : str
        Path of output figure.
    mode : int
        If mode is 1, 3 series are given per RCP (curves and envelopes).
        If mode is 2, n series are given per RCP (curves only).
    --------------------------------------------------------------------------------------------------------------------
    """

    # ==========================================================
    # TODO.CUSTOMIZATION.BEGIN
    # When adding a new climate index, calculate the index by
    # copying the following code block.
    # ==========================================================

    if var_or_idx == cfg.idx_tx_days_above:
        title = cfg.get_idx_desc(var_or_idx, threshs) + cfg.get_var_unit(cfg.var_cordex_tasmax)
        label = "Nombre de jours"

    # ==========================================================
    # TODO.CUSTOMIZATION.END
    # ==========================================================

    else:
        title = cfg.get_var_desc(var_or_idx).capitalize()
        label = title + " (" + cfg.get_var_unit(var_or_idx) + ")"
    title = title + " (" + stn + ")"

    # Initialize plot.
    f, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Année')
    ax.secondary_yaxis('right')
    ax.get_yaxis().tick_right()
    ax.axes.get_yaxis().set_visible(False)
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(label)
    plt.subplots_adjust(top=0.925, bottom=0.10, left=0.03, right=0.90, hspace=0.30, wspace=0.416)

    # Update plot.
    ds_mean = None
    ds_min = None
    ds_max = None
    for rcp in rcps:

        # Colors.
        color = "black"
        if rcp == cfg.rcp_ref:
            color = cfg.col_ref
        elif rcp == cfg.rcp_26:
            color = cfg.col_rcp26
        elif rcp == cfg.rcp_45:
            color = cfg.col_rcp45
        elif rcp == cfg.rcp_85:
            color = cfg.col_rcp85

        # Mode #1: Curves and envelopes.
        if mode == 1:

            if (rcp == cfg.rcp_ref) and (ds_ref is not None):
                ax.plot(ds_ref["time"], ds_ref[var_or_idx], color="black", alpha=1.0)
            else:
                if rcp == cfg.rcp_26:
                    ds_mean = ds_rcp_26[0]
                    ds_min  = ds_rcp_26[1]
                    ds_max  = ds_rcp_26[2]
                elif rcp == cfg.rcp_45:
                    ds_mean = ds_rcp_45[0]
                    ds_min  = ds_rcp_45[1]
                    ds_max  = ds_rcp_45[2]
                elif rcp == cfg.rcp_85:
                    ds_mean = ds_rcp_85[0]
                    ds_min  = ds_rcp_85[1]
                    ds_max  = ds_rcp_85[2]
                ax.plot(ds_mean["time"], ds_mean[var_or_idx], color=color, alpha=1.0)
                ax.fill_between(ds_max["time"], ds_min[var_or_idx], ds_max[var_or_idx], color=color, alpha=0.25)

        # Mode #2: Curves only.
        elif mode == 2:

            # Draw curves.
            if (rcp == cfg.rcp_ref) and (ds_ref is not None):
                ds = ds_ref
                ax.plot(ds["time"], ds[var_or_idx].values, color="black", alpha=1.0)
            elif rcp == cfg.rcp_26:
                for ds in ds_rcp_26:
                    ax.plot(ds["time"], ds[var_or_idx].values, color=color, alpha=0.5)
            elif rcp == cfg.rcp_45:
                for ds in ds_rcp_45:
                    ax.plot(ds["time"], ds[var_or_idx].values, color=color, alpha=0.5)
            elif rcp == cfg.rcp_85:
                for ds in ds_rcp_85:
                    ax.plot(ds["time"], ds[var_or_idx].values, color=color, alpha=0.5)

    # Finalize plot.
    legend_list = ["Référence"]
    if cfg.rcp_26 in rcps:
        legend_list.append("RCP 2,6")
    if cfg.rcp_45 in rcps:
        legend_list.append("RCP 4,5")
    if cfg.rcp_85 in rcps:
        legend_list.append("RCP 8,5")
    custom_lines = [Line2D([0], [0], color="black", lw=4), Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="green", lw=4), Line2D([0], [0], color="red", lw=4)]
    ax.legend(custom_lines, legend_list, loc="upper left", frameon=False)
    plt.ylim(ylim[0], ylim[1])

    # Save figure.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    plt.close()


# ======================================================================================================================
# Verification
# ======================================================================================================================


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

        p_ref   = [i for i in p_list if cfg.rcp_ref in i][i]
        p_fut   = p_ref.replace(cfg.rcp_ref + "_", "")
        p_qqmap = p_fut.replace("_4qqmap", "").replace(cfg.cat_regrid, cfg.cat_qqmap)
        ds_fut   = xr.open_dataset(p_fut)
        ds_qqmap = xr.open_dataset(p_qqmap)
        ds_obs   = xr.open_dataset(p_obs)

        # Convert date format if the need is.
        if ds_fut.time.dtype == cfg.dtype_obj:
            ds_fut["time"] = utils.reset_calendar(ds_fut)
        if ds_qqmap.time.dtype == cfg.dtype_obj:
            ds_qqmap["time"] = utils.reset_calendar(ds_qqmap)

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
        p_fut_i   = [i for i in p_list if cfg.rcp_ref in i][i].replace(cfg.rcp_ref + "_", "")
        p_qqmap_i = p_fut_i.replace("_4" + cfg.cat_qqmap, "").replace(cfg.cat_regrid, cfg.cat_qqmap)

        # Open datasets.
        ds_fut   = xr.open_dataset(p_fut_i)
        ds_qqmap = xr.open_dataset(p_qqmap_i)
        ds_obs   = xr.open_dataset(p_obs)

        # Convert date format if the need is.
        if ds_fut.time.dtype == cfg.dtype_obj:
            ds_fut["time"] = utils.reset_calendar(ds_fut)
        if ds_qqmap.time.dtype == cfg.dtype_obj:
            ds_qqmap["time"] = utils.reset_calendar(ds_qqmap)

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
