# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Plot functions.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# 2. marc-andre.bourgault@ggr.ulaval.ca (original)
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
from matplotlib.lines import Line2D
from scipy import signal
from scipy.interpolate import griddata


# ======================================================================================================================
# Aggregation
# ======================================================================================================================

def plot_year(ds_hour: xr.Dataset, ds_day: xr.Dataset, set_name: str, var: str):

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
                               cbar_kwargs=dict(orientation="vertical", pad=0.05, shrink=1, label=var_desc))
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


def plot_postprocess(p_stn, p_fut, p_qqmap, var, p_fig, title):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of observed and future periods.

    Parameters
    ----------
    p_stn : str
        Path of NetCDF file containing station data.
    p_fut : str
        Path of NetCDF file containing simulation data (future period).
    p_qqmap : str
        Path of NetCDF file containing adjusted simulation data.
    var : str
        Variable.
    title : str
        Title of figure.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    da_stn = utils.open_netcdf(p_stn)[var]
    if cfg.dim_longitude in da_stn.dims:
        da_stn = da_stn.rename({cfg.dim_longitude: cfg.dim_rlon, cfg.dim_latitude: cfg.dim_rlat})
    da_fut = utils.open_netcdf(p_fut)[var]
    da_qqmap = utils.open_netcdf(p_qqmap)[var]

    # Select control point.
    if cfg.opt_ra:
        subset_ctrl_pt = False
        if cfg.dim_rlon in da_stn.dims:
            subset_ctrl_pt = (len(da_stn.rlon) > 1) or (len(da_stn.rlat) > 1)
        elif cfg.dim_lon in da_stn.dims:
            subset_ctrl_pt = (len(da_stn.lon) > 1) or (len(da_stn.lat) > 1)
        elif cfg.dim_longitude in da_stn.dims:
            subset_ctrl_pt = (len(da_stn.longitude) > 1) or (len(da_stn.latitude) > 1)
        if subset_ctrl_pt:
            if cfg.d_bounds == "":
                da_stn   = utils.subset_ctrl_pt(da_stn)
                da_fut   = utils.subset_ctrl_pt(da_fut)
                da_qqmap = utils.subset_ctrl_pt(da_qqmap)
            else:
                da_stn   = utils.squeeze_lon_lat(da_stn)
                da_fut   = utils.squeeze_lon_lat(da_fut)
                da_qqmap = utils.squeeze_lon_lat(da_qqmap)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    delta_stn   = 0
    delta_fut   = 0
    delta_qqmap = 0
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
        coef = cfg.spd * 365
    elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        if da_stn.units == cfg.unit_K:
            delta_stn = -cfg.d_KC
        if da_fut.units == cfg.unit_K:
            delta_fut = -cfg.d_KC
        if da_qqmap.units == cfg.unit_K:
            delta_qqmap = -cfg.d_KC

    # Plot.
    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.04, right=0.99, hspace=0.695, wspace=0.416)
    fs_sup_title = 8
    fs_legend = 8
    fs_axes = 8
    legend_items = ["Simulation", "Observation"]
    if da_qqmap is not None:
        legend_items.insert(0, "Sim. ajustée")
        (da_qqmap * coef + delta_qqmap).groupby(da_qqmap.time.dt.year).mean().plot.line(color=cfg.col_sim_adj)
    (da_fut * coef + delta_fut).groupby(da_fut.time.dt.year).mean().plot.line(color=cfg.col_sim_fut)
    (da_stn * coef + delta_stn).groupby(da_stn.time.dt.year).mean().plot(color=cfg.col_obs)

    # Customize.
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

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
    da_ref = utils.open_netcdf(p_regrid_ref)[var]
    da_fut = utils.open_netcdf(p_regrid_fut)[var]

    # Select control point.
    if cfg.opt_ra:
        subset_ctrl_pt = False
        if cfg.dim_rlon in da_ref.dims:
            subset_ctrl_pt = (len(da_ref.rlon) > 1) or (len(da_ref.rlat) > 1)
        elif cfg.dim_lon in da_ref.dims:
            subset_ctrl_pt = (len(da_ref.lon) > 1) or (len(da_ref.lat) > 1)
        elif cfg.dim_longitude in da_ref.dims:
            subset_ctrl_pt = (len(da_ref.longitude) > 1) or (len(da_ref.latitude) > 1)
        if subset_ctrl_pt:
            if cfg.d_bounds == "":
                da_ref = utils.subset_ctrl_pt(da_ref)
                da_fut = utils.subset_ctrl_pt(da_fut)
            else:
                da_ref = utils.squeeze_lon_lat(da_ref)
                da_fut = utils.squeeze_lon_lat(da_fut)

    # Conversion coefficients.
    coef = 1
    delta_ref = 0
    delta_fut = 0
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
        coef = cfg.spd
    if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        if da_ref.units == cfg.unit_K:
            delta_ref = -cfg.d_KC
        if da_fut.units == cfg.unit_K:
            delta_fut = -cfg.d_KC

    # Fit.
    x     = [*range(len(da_ref.time))]
    y     = (da_ref * coef + delta_ref).values
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
    if da_ref.time.dtype == cfg.dtype_obj:
        da_ref[cfg.dim_time] = utils.reset_calendar(da_ref)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

        # Upper plot: Reference period.
        f.add_subplot(211)
        plt.plot(da_ref.time, y, color=cfg.col_sim_ref)
        plt.plot(da_ref.time, ffit, color="black")
        plt.legend(["Simulation (réf.)", "Tendance"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        plt.title("Tendance", fontsize=fs_title)

        # Lower plot: Complete simulation.
        f.add_subplot(212)
        arr_y_detrend = signal.detrend(da_fut * coef + delta_fut)
        arr_x_detrend = cfg.per_ref[0] + np.arange(0, len(arr_y_detrend), 1) / 365
        arr_y_error  = (y - ffit)
        arr_x_error = cfg.per_ref[0] + np.arange(0, len(arr_y_error), 1) / 365
        plt.plot(arr_x_detrend, arr_y_detrend, alpha=0.5, color=cfg.col_sim_fut)
        plt.plot(arr_x_error, arr_y_error, alpha=0.5, color=cfg.col_sim_ref)
        plt.legend(["Simulation", "Simulation (réf.)"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Jours", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        plt.title("Variation autour de la moyenne (prédiction basée sur une équation quartique)", fontsize=fs_title)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


# ======================================================================================================================
# Calibration
# ======================================================================================================================


def plot_calib(da_obs, da_ref, da_fut, da_qqmap, da_qqmap_ref, da_qmf, var, sup_title, p_fig):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot containing a summary of calibration.

    Parameters
    ----------
    da_obs : xr.DataArray
        Observations.
    da_ref : xr.DataArray
        Simulation for the reference period.
    da_fut : xr.DataArray
        Simulation for the future period.
    da_qqmap : xr.DataArray
        Adjusted simulation.
    da_qqmap_ref : xr.DataArray
        Adjusted simulation for the reference period.
    da_qmf : xr.DataArray
        Quantile mapping function.
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
    fs_legend    = 5
    fs_axes      = 7

    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(431)
    plt.subplots_adjust(top=0.930, bottom=0.065, left=0.070, right=0.973, hspace=0.90, wspace=0.250)

    # Quantile mapping function.
    img1 = ax.imshow(da_qmf, extent=[0, 1, 365, 1], cmap="coolwarm")
    cb = f.colorbar(img1, ax=ax)
    # cb.set_label("Ajustement", fontsize=fs_axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cb.ax.set_yticklabels(cb.ax.get_yticks(), fontsize=fs_axes)
    cb.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))
    plt.title("QMF", fontsize=fs_title)
    plt.xlabel("Quantile", fontsize=fs_axes)
    plt.ylabel("Jour de l'année", fontsize=fs_axes)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 365)
    ax.set_aspect("auto")

    legend_items = ["Observations", "Sim. (réf.)", "Sim.", "Sim. ajustée", "Sim. ajustée (réf.)"]

    # Mean values ------------------------------------------------------------------------------------------------------

    # Plot.
    f.add_subplot(432)
    draw_curves(var, da_obs, da_ref, da_fut, da_qqmap, da_qqmap_ref, cfg.stat_mean)
    plt.title("Moyenne", fontsize=fs_title)
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlim([1, 12])
    plt.xticks(np.arange(1, 13, 1))
    plt.xlabel("Mois", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Mean, Q100, Q99, Q75, Q50, Q25, Q01 and Q00 monthly values -------------------------------------------------------

    for i in range(1, len(cfg.stat_quantiles) + 1):

        plt.subplot(433 + i - 1)

        stat     = "quantile"
        quantile = cfg.stat_quantiles[i-1]
        title    = "Q_" + "{0:.2f}".format(quantile)
        if quantile == 0:
            stat = cfg.stat_min
        elif quantile == 1:
            stat = cfg.stat_max

        draw_curves(var, da_obs, da_ref, da_fut, da_qqmap, da_qqmap_ref, stat, quantile)

        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.legend(legend_items, fontsize=fs_legend, frameon=False)
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)

    # Time series ------------------------------------------------------------------------------------------------------

    # Convert date format if the need is.
    if da_qqmap.time.time.dtype == cfg.dtype_obj:
        da_qqmap[cfg.dim_time] = utils.reset_calendar(da_qqmap.time)
    if da_ref.time.time.dtype == cfg.dtype_obj:
        da_ref[cfg.dim_time] = utils.reset_calendar(da_ref.time)

    plt.subplot(313)
    da_qqmap.plot.line(alpha=0.5, color=cfg.col_sim_adj)
    da_ref.plot.line(alpha=0.5, color=cfg.col_sim_ref)
    da_obs.plot.line(alpha=0.5, color=cfg.col_obs)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.legend(["Sim. ajustée", "Sim. (réf.)", "Observations"], fontsize=fs_legend, frameon=False)
    plt.title("")

    f.suptitle(var + "_" + sup_title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    if cfg.attrs_bias in da_qqmap.attrs:
        del da_qqmap.attrs[cfg.attrs_bias]

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_calib_ts(da_obs: xr.DataArray, da_fut: xr.DataArray, da_qqmap: xr.DataArray, var: str, title: str, p_fig: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of observed and future periods.

    Parameters
    ----------
    da_obs : xr.DataArray
        Observations.
    da_fut : xr.DataArray
        Simulation for the future period.
    da_qqmap : xr.DataArray
        Adjusted simulation.
    var : str
        Variable.
    title : str
        Title of figure.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Convert date format if the need is.
    if da_obs.time.dtype == cfg.dtype_obj:
        da_obs[cfg.dim_time] = utils.reset_calendar(da_obs)
    if da_fut.time.dtype == cfg.dtype_obj:
        da_fut[cfg.dim_time] = utils.reset_calendar(da_fut)
    if da_qqmap.time.dtype == cfg.dtype_obj:
        da_qqmap[cfg.dim_time] = utils.reset_calendar(da_qqmap)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    fs_sup_title = 8
    fs_legend = 8
    fs_axes = 8
    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.21, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

    # Add curves.
    da_qqmap.plot.line(alpha=0.5, color=cfg.col_sim_adj)
    da_fut.plot.line(alpha=0.5, color=cfg.col_sim_fut)
    da_obs.plot.line(alpha=0.5, color=cfg.col_obs)

    # Customize.
    plt.legend(["Sim. ajustée", "Sim. (réf.)", "Observations"], fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def draw_curves(var, da_obs: xr.DataArray, da_ref: xr.DataArray, da_fut: xr.DataArray, da_qqmap: xr.DataArray,
                da_qqmap_ref: xr.DataArray, stat: str, quantile: float = -1.0):

    """
    --------------------------------------------------------------------------------------------------------------------
    Draw curves.

    Parameters
    ----------
    var : str
        Variable.
    da_obs : xr.DataArray
        Observations.
    da_ref : xr.DataArray
        Simulation for the reference period.
    da_fut : xr.DataArray
        Simulation for the future period.
    da_qqmap : xr.DataArray
        Adjusted simulation.
    da_qqmap_ref : xr.DataArray
        Adjusted simulation for the reference period.
    stat : str
        Statistic: {cfg.stat_max, cfg.stat_quantile, cfg.stat_mean, cfg.stat_sum}
    quantile : float, optional
        Quantile.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate statistics.
    def da_groupby(da: xr.DataArray, stat_inner: str, quantile_inner: float = -1.0) -> xr.DataArray:
        da_group = da.groupby(da.time.dt.month)
        da_group_stat = None
        if stat_inner == cfg.stat_min:
            da_group_stat = da_group.min(dim=cfg.dim_time)
        elif stat_inner == cfg.stat_max:
            da_group_stat = da_group.max(dim=cfg.dim_time)
        elif stat_inner == cfg.stat_mean:
            da_group_stat = da_group.mean(dim=cfg.dim_time)
        elif stat_inner == cfg.stat_quantile:
            da_group_stat = da_group.quantile(quantile_inner, dim=cfg.dim_time)
        elif stat_inner == cfg.stat_sum:
            n_years = da[cfg.dim_time].size / 365
            da_group_stat = da_group.sum(dim=cfg.dim_time) / n_years
        return da_group_stat

    # Determine if sum is needed.
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
        da_obs       = da_obs.resample(time="1M").sum()
        da_ref       = da_ref.resample(time="1M").sum()
        da_fut       = da_fut.resample(time="1M").sum()
        da_qqmap     = da_qqmap.resample(time="1M").sum()
        da_qqmap_ref = da_qqmap_ref.resample(time="1M").sum()

    # Calculate statistics
    if stat == cfg.stat_min:
        da_obs       = da_groupby(da_obs, cfg.stat_min)
        da_ref       = da_groupby(da_ref, cfg.stat_min)
        da_fut       = da_groupby(da_fut, cfg.stat_min)
        da_qqmap     = da_groupby(da_qqmap, cfg.stat_min)
        da_qqmap_ref = da_groupby(da_qqmap_ref, cfg.stat_min)
    elif stat == cfg.stat_max:
        da_obs       = da_groupby(da_obs, cfg.stat_max)
        da_ref       = da_groupby(da_ref, cfg.stat_max)
        da_fut       = da_groupby(da_fut, cfg.stat_max)
        da_qqmap     = da_groupby(da_qqmap, cfg.stat_max)
        da_qqmap_ref = da_groupby(da_qqmap_ref, cfg.stat_max)
    elif stat == cfg.stat_mean:
        da_obs       = da_groupby(da_obs, cfg.stat_mean)
        da_ref       = da_groupby(da_ref, cfg.stat_mean)
        da_fut       = da_groupby(da_fut, cfg.stat_mean)
        da_qqmap     = da_groupby(da_qqmap, cfg.stat_mean)
        da_qqmap_ref = da_groupby(da_qqmap_ref, cfg.stat_mean)
    elif stat == cfg.stat_sum:
        da_obs       = da_groupby(da_obs, cfg.stat_sum)
        da_ref       = da_groupby(da_ref, cfg.stat_sum)
        da_fut       = da_groupby(da_fut, cfg.stat_sum)
        da_qqmap     = da_groupby(da_qqmap, cfg.stat_sum)
        da_qqmap_ref = da_groupby(da_qqmap_ref, cfg.stat_sum)
    elif stat == cfg.stat_quantile:
        da_obs       = da_groupby(da_obs, cfg.stat_quantile, quantile)
        da_ref       = da_groupby(da_ref, cfg.stat_quantile, quantile)
        da_fut       = da_groupby(da_fut, cfg.stat_quantile, quantile)
        da_qqmap     = da_groupby(da_qqmap, cfg.stat_quantile, quantile)
        da_qqmap_ref = da_groupby(da_qqmap_ref, cfg.stat_quantile, quantile)

    # Draw curves.
    da_obs.plot.line(color=cfg.col_obs)
    da_ref.plot.line(color=cfg.col_sim_ref)
    da_fut.plot.line(color=cfg.col_sim_fut)
    da_qqmap.plot.line(color=cfg.col_sim_adj)
    da_qqmap_ref.plot.line(color=cfg.col_sim_adj_ref)


def plot_360_vs_365(ds_360: xr.Dataset, ds_365: xr.Dataset, var: str = ""):

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


def plot_rsq(rsq: np.array, n_sim: int):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a plot of root mean square error.

    Parameters
    ----------
    rsq : np.array
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


def plot_heatmap(var_or_idx: str, threshs: [float], rcp: str, per_hors: [[int]]):

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
    cat = cfg.cat_scen if var_or_idx in cfg.variables_cordex else cfg.cat_idx

    utils.log("-")
    if cat == cfg.cat_scen:
        utils.log("Scenario          : " + var_or_idx, True)
    else:
        utils.log("Index             : " + var_or_idx, True)
    utils.log("Emission scenario : " + cfg.get_rcp_desc(rcp), True)
    utils.log("-")

    # Number of years and stations.
    if rcp == cfg.rcp_ref:
        n_year = cfg.per_ref[1] - cfg.per_ref[0] + 1
    else:
        n_year = cfg.per_fut[1] - cfg.per_ref[1] + 1

    # List stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]

    # Observations -----------------------------------------------------------------------------------------------------

    if not cfg.opt_ra:

        # Get information on stations.
        # TODO.YR: Coordinates should be embedded into ds_stat below.
        p_stn = glob.glob(cfg.get_d_stn(cfg.var_cordex_tas) + "../*.csv")[0]
        df = pd.read_csv(p_stn, sep=cfg.file_sep)

        # Collect values for each station and determine overall boundaries.
        utils.log("Collecting emissions scenarios at each station.", True)
        x_bnds = []
        y_bnds = []
        data_stn = []
        for stn in stns:

            # Get coordinates.
            lon = df[df["station"] == stn][cfg.dim_lon].values[0]
            lat = df[df["station"] == stn][cfg.dim_lat].values[0]

            # Calculate statistics.
            if rcp == cfg.rcp_ref:
                ds_stat = statistics.calc_stat(cfg.cat_obs, cfg.freq_YS, cfg.freq_YS, stn, var_or_idx, rcp, None,
                                               cfg.stat_mean)
            else:
                ds_stat = statistics.calc_stat(cfg.cat_scen, cfg.freq_YS, cfg.freq_YS, stn, var_or_idx, rcp, None,
                                               cfg.stat_mean)
            if ds_stat is None:
                continue

            # Extract data from stations.
            data = [[], [], []]
            n = ds_stat.dims[cfg.dim_time]
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
        # There is a certain flexibility regarding the number of years in a dataset. Ideally, the station should not
        # have been considered in the analysis, unless there is no better option.
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
            new_grid_data[i_year, :, :] =\
                griddata((arr_x, arr_y), np.array(arr_z), (new_grid[0], new_grid[1]), fill_value=np.nan,
                         method="linear")
        da_itp = xr.DataArray(new_grid_data,
                              coords={cfg.dim_time: grid_time, cfg.dim_lat: grid_y, cfg.dim_lon: grid_x},
                              dims=[cfg.dim_time, cfg.dim_lat, cfg.dim_lon])
        ds_itp = da_itp.to_dataset(name=var_or_idx)

    # Reanalysis -------------------------------------------------------------------------------------------------------

    # There is no need to interpolate.
    else:

        # Reference period.
        if var_or_idx in cfg.variables_cordex:
            p_itp_ref = cfg.get_d_scen(cfg.obs_src, cfg.cat_obs, var_or_idx) + var_or_idx + "_" + cfg.obs_src + ".nc"
        else:
            p_itp_ref = cfg.get_d_idx(cfg.obs_src, var_or_idx) + var_or_idx + "_ref.nc"
        if rcp == cfg.rcp_ref:
            p_itp = p_itp_ref
            if not os.path.exists(p_itp):
                return
            ds_itp = utils.open_netcdf(p_itp)

        # Future period.
        else:

            # List scenarios or indices for the current RCP.
            if cat == cfg.cat_scen:
                d = cfg.get_d_scen(cfg.obs_src, cfg.cat_qqmap, var_or_idx)
            else:
                d = cfg.get_d_scen(cfg.obs_src, cfg.cat_idx, var_or_idx)
            p_sim_list = [i for i in glob.glob(d + "*.nc") if i != p_itp_ref]

            # Combine datasets.
            ds_itp = None
            n_sim = 0
            units = None
            for p_sim in p_sim_list:
                if os.path.exists(p_sim) and (rcp in p_sim):
                    ds_sim = utils.open_netcdf(p_sim)
                    if cat == cfg.cat_scen:
                        ds_sim[cfg.dim_time] = utils.reset_calendar(ds_sim.time)
                    if ds_itp is None:
                        ds_itp = ds_sim
                        if cfg.attrs_units in ds_sim[var_or_idx].attrs:
                            units = ds_sim[var_or_idx].attrs[cfg.attrs_units]
                        elif cfg.attrs_units in ds_sim.data_vars:
                            units = ds_sim[cfg.attrs_units]
                        else:
                            units = 1
                    else:
                        ds_itp[var_or_idx] = ds_itp[var_or_idx] + ds_sim[var_or_idx]
                    n_sim = n_sim + 1
            if ds_itp is None:
                return
            ds_itp[var_or_idx] = ds_itp[var_or_idx] / float(n_sim)
            ds_itp[var_or_idx].attrs[cfg.attrs_units] = units

        # Convert units (C or mm).
        if (var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]) and\
           (ds_itp[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K):
            ds_itp = ds_itp - cfg.d_KC
            ds_itp[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_C
        elif (var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]) and \
             (ds_itp[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kgm2s1):
            ds_itp = ds_itp * cfg.spd
            ds_itp[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_mm

        # Adjust coordinate names (required for clipping).
        # TODO.YR: Ideally, this should be done elsewhere.
        if cfg.dim_longitude not in list(ds_itp.dims):
            if cfg.dim_rlon in ds_itp.dims:
                ds_itp = ds_itp.rename_dims({cfg.dim_rlon: cfg.dim_longitude, cfg.dim_rlat: cfg.dim_latitude})
                ds_itp[cfg.dim_longitude] = ds_itp[cfg.dim_rlon]
                del ds_itp[cfg.dim_rlon]
                ds_itp[cfg.dim_latitude] = ds_itp[cfg.dim_rlat]
                del ds_itp[cfg.dim_rlat]
            else:
                ds_itp = ds_itp.rename_dims({cfg.dim_lon: cfg.dim_longitude, cfg.dim_lat: cfg.dim_latitude})
                ds_itp[cfg.dim_longitude] = ds_itp[cfg.dim_lon]
                del ds_itp[cfg.dim_lon]
                ds_itp[cfg.dim_latitude] = ds_itp[cfg.dim_lat]
                del ds_itp[cfg.dim_lat]

        grid_x = None
        grid_y = None

    # Maps -------------------------------------------------------------------------------------------------------------

    # Loop through horizons.
    utils.log("Generating maps.", True)
    for per_hor in per_hors:

        # Select years.
        if rcp == cfg.rcp_ref:
            year_1 = 0
            year_n = cfg.per_ref[1] - cfg.per_ref[0]
        else:
            year_1 = per_hor[0] - cfg.per_ref[1]
            year_n = per_hor[1] - cfg.per_ref[1]
        n_years = year_n - year_1 + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds_hor = ds_itp[var_or_idx][year_1:(year_n + 1)][:][:]
            if var_or_idx not in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                da_hor = ds_hor.mean(cfg.dim_time, skipna=True)
            else:
                da_hor = ds_hor.groupby(ds_hor.time.dt.year).sum(dim=cfg.dim_time).mean("year", skipna=True) * n_years
                if len(ds_itp.time.values) <= n_years:
                    da_hor = da_hor * 365
                da_hor = da_hor.where(da_hor.values >= 1)

        # Sum up quantities.
        # if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
        #     if len(ds_itp.time.values) <= n_years:
        #         da_hor = da_hor * 365
        #     da_hor = da_hor * n_years

        # Clip.
        if cfg.d_bounds != "":
            da_hor = utils.subset_shape(da_hor)

        # Plot.
        cat_fig = cfg.cat_fig + "/" + ("stns" if not cfg.opt_ra else cfg.obs_src) + \
            ("_" + cfg.region if (cfg.region != "") and cfg.opt_ra else "") + "/" + cat
        p_fig = cfg.get_d_scen("", cat_fig, var_or_idx) +\
            var_or_idx + "_" + rcp + "_" + str(per_hor[0]) + "_" + str(per_hor[1]) + ".png"
        plot_heatmap_spec(da_hor, var_or_idx, threshs, grid_x, grid_y, per_hor, p_fig, "matplotlib")


def plot_heatmap_spec(da: xr.DataArray, var_or_idx: str, threshs: [float], grid_x: [float], grid_y: [float],
                      per: [int, int], p_fig: str, map_package: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a heat map of a climate index for the reference period and for emission scenarios.
    TODO: Add a color scale that is common to all horizons.

    Parameters
    ----------
    da: xr.DataArray
        DataArray (with 2 dimensions: longitude and latitude).
    var_or_idx : str
        Climate variable (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_tx_days_above).
    threshs : [float]
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
        label = "Nbr. jours"

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
        g = sns.heatmap(ax=ax, data=da, xticklabels=grid_x, yticklabels=grid_y)
        if grid_x is not None:
            x_labels = ['{:,.2f}'.format(i) for i in grid_x]
            g.set_xticklabels(x_labels)
        if grid_y is not None:
            y_labels = ['{:,.2f}'.format(i) for i in grid_y]
            g.set_yticklabels(y_labels)

    # Using matplotlib.
    elif map_package == "matplotlib":
        fs = 10
        da.plot.pcolormesh(add_colorbar=True, add_labels=True,
                           cbar_kwargs=dict(orientation='vertical', pad=0.05, shrink=1, label=label))
        plt.title(title)
        plt.suptitle("", fontsize=fs)
        plt.xlabel("Longitude (º)", fontsize=fs)
        plt.ylabel("Latitude (º)", fontsize=fs)
        plt.tick_params(axis="x", labelsize=fs)
        plt.tick_params(axis="y", labelsize=fs)
        plt.xlim(cfg.lon_bnds)
        plt.ylim(cfg.lat_bnds)

    # Save figure.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    plt.close()


def plot_ts(var_or_idx: str, threshs: [float] = None):

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
    rcps = [cfg.rcp_ref] + cfg.rcps

    # Determine category.
    cat = cfg.cat_scen if var_or_idx in cfg.variables_cordex else cfg.cat_idx

    # Minimum and maximum values along the y-axis
    ylim = []

    # Loop through stations.
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        utils.log("Processing: '" + var_or_idx + "', '" + stn + "'", True)

        # Loop through emission scenarios.
        ds_ref = None
        ds_rcp_26, ds_rcp_26_grp = [], []
        ds_rcp_45, ds_rcp_45_grp = [], []
        ds_rcp_85, ds_rcp_85_grp = [], []
        for rcp in rcps:

            # List files.
            if rcp == cfg.rcp_ref:
                if var_or_idx in cfg.variables_cordex:
                    p_sim_list = [cfg.get_d_scen(stn, cfg.cat_obs, var_or_idx) + var_or_idx + "_" + cfg.obs_src + ".nc"]
                else:
                    p_sim_list = [cfg.get_d_idx(cfg.obs_src, var_or_idx) + var_or_idx + "_ref.nc"]
            else:
                if var_or_idx in cfg.variables_cordex:
                    d = cfg.get_d_scen(stn, cfg.cat_qqmap, var_or_idx)
                else:
                    d = cfg.get_d_scen(stn, cfg.cat_idx, var_or_idx)
                p_sim_list = glob.glob(d + "*_" + rcp + ".nc")

            # Exit if there is not file corresponding to the criteria.
            if (len(p_sim_list) == 0) or \
               ((len(p_sim_list) > 0) and not(os.path.isdir(os.path.dirname(p_sim_list[0])))):
                continue

            # Loop through simulation files.
            for i_sim in range(len(p_sim_list)):

                # Load dataset.
                ds = utils.open_netcdf(p_sim_list[i_sim]).squeeze()

                # Select control point.
                if cfg.opt_ra:
                    if cfg.d_bounds == "":
                        ds = utils.subset_ctrl_pt(ds)
                    else:
                        ds = utils.squeeze_lon_lat(ds, var_or_idx)

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

                # Remember units.
                units = ds[var_or_idx].attrs[cfg.attrs_units] if cat == cfg.cat_scen else ds.attrs[cfg.attrs_units]
                if units == "degree_C":
                    units = cfg.unit_C

                # Calculate statistics.
                # TODO: Include coordinates in the generated dataset.
                years = ds.groupby(ds.time.dt.year).groups.keys()
                if var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                    ds = ds.groupby(ds.time.dt.year).sum(keepdims=True)
                else:
                    ds = ds.groupby(ds.time.dt.year).mean(keepdims=True)
                n_time = len(ds[cfg.dim_time].values)
                da = xr.DataArray(np.array(ds[var_or_idx].values), name=var_or_idx,
                                  coords=[(cfg.dim_time, np.arange(n_time))])
                ds = da.to_dataset()
                ds[cfg.dim_time] = utils.reset_calendar_list(years)
                ds[var_or_idx].attrs[cfg.attrs_units] = units

                # Convert units.
                if var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                    if ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_K:
                        ds = ds - cfg.d_KC
                        ds[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_C
                elif var_or_idx in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
                    if ds[var_or_idx].attrs[cfg.attrs_units] == cfg.unit_kgm2s1:
                        ds = ds * cfg.spd
                        ds[var_or_idx].attrs[cfg.attrs_units] = cfg.unit_mm

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

            # Time series with simulations grouped by RCP scenario.
            cat_fig = cfg.cat_fig + "/" + cat + "/" + var_or_idx + "/"
            p_fig = cfg.get_d_scen(stn, cat_fig, "") + var_or_idx + "_" + stn + "_rcp.png"
            plot_ts_spec(ds_ref, ds_rcp_26_grp, ds_rcp_45_grp, ds_rcp_85_grp, stn.capitalize(), var_or_idx, threshs,
                         rcps, ylim, p_fig, 1)

            # Time series showing individual simulations.
            p_fig = p_fig.replace("_rcp.png", "_sim.png")
            plot_ts_spec(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(), var_or_idx, threshs,
                         rcps, ylim, p_fig, 2)


def calc_stat_mean_min_max(ds_list: [xr.Dataset], var_or_idx: str):

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
    units = ds_list[0][var_or_idx].attrs[cfg.attrs_units]
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
        arr_vals_min.append(min(np.array(vals)))
        arr_vals_max.append(max(np.array(vals)))

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
        da = xr.DataArray(np.array(arr_vals), name=var_or_idx, coords=[(cfg.dim_time, np.arange(n_time))])
        ds = da.to_dataset()
        ds[cfg.dim_time] = utils.reset_calendar(ds, year_1, year_n, cfg.freq_YS)
        ds[var_or_idx].attrs[cfg.attrs_units] = units

        ds_mean_min_max.append(ds)

    return ds_mean_min_max


def plot_ts_spec(ds_ref: xr.Dataset, ds_rcp_26: [xr.Dataset], ds_rcp_45: [xr.Dataset], ds_rcp_85: [xr.Dataset],
                 stn: str, var_or_idx: str, threshs: [float], rcps: [str], ylim: [int], p_fig: str, mode: int = 1):

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
    threshs : [float]
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
        label = "Nbr. jours"

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
    secax = ax.secondary_yaxis("right")
    secax.set_ylabel(label)
    plt.subplots_adjust(top=0.925, bottom=0.10, left=0.03, right=0.90, hspace=0.30, wspace=0.416)

    # Update plot.
    ds_mean = None
    ds_min = None
    ds_max = None
    for rcp in rcps:

        # Skip if no simulation is available for this RCP.
        if ((rcp == cfg.rcp_26) and (ds_rcp_26 == [])) or\
           ((rcp == cfg.rcp_45) and (ds_rcp_45 == [])) or\
           ((rcp == cfg.rcp_85) and (ds_rcp_85 == [])):
            continue

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
                ax.plot(ds_ref[cfg.dim_time], ds_ref[var_or_idx], color="black", alpha=1.0)
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
                ax.plot(ds_mean[cfg.dim_time], ds_mean[var_or_idx], color=color, alpha=1.0)
                ax.fill_between(np.array(ds_max[cfg.dim_time]), ds_min[var_or_idx], ds_max[var_or_idx],
                                color=color, alpha=0.25)

        # Mode #2: Curves only.
        elif mode == 2:

            # Draw curves.
            if (rcp == cfg.rcp_ref) and (ds_ref is not None):
                ds = ds_ref
                ax.plot(ds[cfg.dim_time], ds[var_or_idx].values, color="black", alpha=1.0)
            elif rcp == cfg.rcp_26:
                for ds in ds_rcp_26:
                    ax.plot(ds[cfg.dim_time], ds[var_or_idx].values, color=color, alpha=0.5)
            elif rcp == cfg.rcp_45:
                for ds in ds_rcp_45:
                    ax.plot(ds[cfg.dim_time], ds[var_or_idx].values, color=color, alpha=0.5)
            elif rcp == cfg.rcp_85:
                for ds in ds_rcp_85:
                    ax.plot(ds[cfg.dim_time], ds[var_or_idx].values, color=color, alpha=0.5)

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


def plot_ts_single(stn: str, var: str):

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

    utils.log("Processing (single): '" + stn + "', '" + var + "'", True)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Paths and NetCDF files.
    d_regrid = cfg.get_d_scen(stn, cfg.cat_regrid, var)
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
        ds_fut   = utils.open_netcdf(p_fut)
        ds_qqmap = utils.open_netcdf(p_qqmap)
        ds_obs   = utils.open_netcdf(p_obs)

        # Convert date format if the need is.
        if ds_fut.time.dtype == cfg.dtype_obj:
            ds_fut[cfg.dim_time] = utils.reset_calendar(ds_fut)
        if ds_qqmap.time.dtype == cfg.dtype_obj:
            ds_qqmap[cfg.dim_time] = utils.reset_calendar(ds_qqmap)

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
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)

        # Save plot.
        p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/verif/ts_single", var) + title + ".png"
        utils.save_plot(plt, p_fig)

        # Close plot.
        plt.close()


def plot_ts_mosaic(stn: str, var: str):

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

    utils.log("Processing (mosaic): '" + stn + "', '" + var + "'", True)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd

    # Paths and NetCDF files.
    d_regrid = cfg.get_d_scen(stn, cfg.cat_regrid, var)
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
        ds_fut   = utils.open_netcdf(p_fut_i)
        ds_qqmap = utils.open_netcdf(p_qqmap_i)
        ds_obs   = utils.open_netcdf(p_obs)

        # Convert date format if the need is.
        if ds_fut.time.dtype == cfg.dtype_obj:
            ds_fut[cfg.dim_time] = utils.reset_calendar(ds_fut)
        if ds_qqmap.time.dtype == cfg.dtype_obj:
            ds_qqmap[cfg.dim_time] = utils.reset_calendar(ds_qqmap)

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
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        if i == 0:
            plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
            sup_title = title + "_verif_ts_mosaic"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Save plot.
    p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/verif/ts_mosaic", var) + title + ".png"
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_monthly(stn: str, var: str):

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
    d_regrid = cfg.get_d_scen(stn, cfg.cat_regrid, var)
    p_list   = utils.list_files(d_regrid)
    p_obs    = cfg.get_p_obs(stn, var)
    ds_obs   = utils.open_netcdf(p_obs)
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
        ds = utils.open_netcdf(p_list[i])[var]
        if isinstance(ds.time[0].values, np.datetime64):
            ds.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().groupby("time.month").mean().\
                plot(color="blue")
        ds = utils.open_netcdf(p_list[i])[var]
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
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        if i == 0:
            sup_title = title + "_verif_monthly"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Format.
    plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)

    # Save plot.
    p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/verif/monthly", var) + sup_title + ".png"
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()
