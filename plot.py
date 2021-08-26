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
import math
import matplotlib.cbook
import matplotlib.cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.polynomial.polynomial as poly
import os.path
import rioxarray as rio  # Do not delete this line: see comment below.
import seaborn as sns
import simplejson
import utils
import warnings
import xarray as xr
import xesmf as xe
from descartes import PolygonPatch
from matplotlib import pyplot
from matplotlib.lines import Line2D
from scipy import signal
from typing import Union, List

# Package 'xesmf' can be installed with:
#   conda install -c conda-forge xesmf
#   pip install xesmf
# Package 'rioxarray' must be added even if it's not explicitly used in order to have a 'rio' variable in DataArrays.


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

    var_desc_unit = cfg.get_desc(var, set_name)  + " [" + cfg.get_unit(var, set_name) + "]"

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
    var_desc = cfg.get_desc(var, set_name) + " (" + cfg.get_unit(var, set_name) + ")"

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
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

    # Conversion coefficient.
    coef = 1
    delta_stn   = 0
    delta_fut   = 0
    delta_qqmap = 0
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
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
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

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
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
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
    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.99, hspace=0.40, wspace=0.00)
    sup_title = os.path.basename(p_fig).replace(cfg.f_ext_png, "") +\
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
        arr_y_detrend = signal.detrend(da_fut[np.isnan(da_fut) == False] * coef + delta_fut)
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
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

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

    for i in range(1, len(cfg.opt_calib_quantiles) + 1):

        plt.subplot(433 + i - 1)

        stat     = "quantile"
        quantile = cfg.opt_stat_quantiles[i-1]
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
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

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
            n_years = da[cfg.dim_time].size / 12
            da_group_stat = da_group.sum(dim=cfg.dim_time) / n_years
        return da_group_stat

    # Determine if sum is needed.
    stat_inner = stat
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evspsbl, cfg.var_cordex_evspsblpot]:
        if stat == cfg.stat_mean:
            stat_inner = cfg.stat_sum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_obs       = da_obs.resample(time="1M").sum()
            da_ref       = da_ref.resample(time="1M").sum()
            da_fut       = da_fut.resample(time="1M").sum()
            da_qqmap     = da_qqmap.resample(time="1M").sum()
            da_qqmap_ref = da_qqmap_ref.resample(time="1M").sum()

    # Calculate statistics
    da_obs       = da_groupby(da_obs, stat_inner, quantile)
    da_ref       = da_groupby(da_ref, stat_inner, quantile)
    da_fut       = da_groupby(da_fut, stat_inner, quantile)
    da_qqmap     = da_groupby(da_qqmap, stat_inner, quantile)
    da_qqmap_ref = da_groupby(da_qqmap_ref, stat_inner, quantile)

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

def plot_heatmap(da: xr.DataArray, stn: str, var_or_idx_code: str, grid_x: [float], grid_y: [float], rcp: str,
                 per: [int, int], stat: str, q: float, z_min: float, z_max: float, is_delta: bool, p_fig: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a heat map of a climate index for the reference period and for emission scenarios.
    The 'map_package' 'seaborn' was not tested.

    Parameters
    ----------
    da: xr.DataArray
        DataArray (with 2 dimensions: longitude and latitude).
    stn : str
        Station name.
    var_or_idx_code : str
        Climate variable or index code.
    grid_x: [float]
        X-coordinates.
    grid_y: [float]
        Y-coordinates.
    rcp: str
        RCP emission scenario.
    per: [int, int]
        Period of interest, for instance, [1981, 2010].
    stat: str, Optional
        Statistic = {"mean", "min", "max", "quantile"}
    q: float, Optional
        Quantile.
    z_min : float
        Minimum value (associated with color bar).
    z_max : float
        Maximum value (associated with color bar).
    is_delta : bool
        If true, indicates that 'da' corresponds to delta values.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Hardcoded parameters.
    # Number of clusters (for discrete color scale).
    if cfg.opt_map_discrete:
        n_cluster = 10
    else:
        n_cluster = 256
    # Maximum number of decimal places for colorbar ticks.
    n_dec_max = 4
    # Font size.
    fs_title      = 8
    fs_labels     = 10
    fs_ticks      = 10
    fs_ticks_cbar = 10
    if is_delta:
        fs_ticks_cbar = fs_ticks_cbar - 1
    # Resolution.
    dpi = 300

    # Extract variable name.
    var_or_idx = var_or_idx_code if var_or_idx_code in cfg.variables_cordex else cfg.extract_idx(var_or_idx_code)

    # Export to GeoTIFF.
    if cfg.f_tif in cfg.opt_map_formats:

        # Increase resolution.
        da_tif = da
        if cfg.opt_map_res > 0:
            lat_vals = np.arange(min(da_tif.latitude), max(da_tif.latitude), cfg.opt_map_res)
            lon_vals = np.arange(min(da_tif.longitude), max(da_tif.longitude), cfg.opt_map_res)
            da_tif = da_tif.rename({cfg.dim_latitude: cfg.dim_lat, cfg.dim_longitude: cfg.dim_lon})
            da_grid = xr.Dataset({cfg.dim_lat: ([cfg.dim_lat], lat_vals), cfg.dim_lon: ([cfg.dim_lon], lon_vals)})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                da_tif = xe.Regridder(da_tif, da_grid, "bilinear")(da_tif)

        # Export.
        da_tif.rio.set_crs("EPSG:4326")
        if cfg.opt_map_spat_ref != "EPSG:4326":
            da_tif.rio.set_spatial_dims(cfg.dim_lon, cfg.dim_lat, inplace=True)
            da_tif = da_tif.rio.reproject(cfg.opt_map_spat_ref)
            da_tif.values[da_tif.values == -9999] = np.nan
            da_tif = da_tif.rename({"y": cfg.dim_lat, "x": cfg.dim_lon})

        p_fig_tif = p_fig.replace(var_or_idx_code + "/", var_or_idx_code + "_" + cfg.f_tif + "/").\
            replace(cfg.f_ext_png, cfg.f_ext_tif)
        d = os.path.dirname(p_fig_tif)
        if not (os.path.isdir(d)):
            os.makedirs(d)
        da_tif.rio.to_raster(p_fig_tif)

    # Export to PNG.
    if cfg.f_png in cfg.opt_map_formats:

        # Get title and label.
        title = cfg.get_plot_title(stn, var_or_idx_code, rcp, per, stat, q) + (" (delta)" if is_delta else "")
        label = cfg.get_plot_ylabel(var_or_idx)

        # Determine color scale index.
        is_wind_var = var_or_idx in [cfg.var_cordex_uas, cfg.var_cordex_vas, cfg.var_cordex_sfcwindmax]
        if (not is_delta) and (not is_wind_var):
            cmap_idx = 0
        elif (z_min < 0) and (z_max > 0):
            cmap_idx = 1
        elif (z_min < 0) and (z_max < 0):
            cmap_idx = 2
        else:
            cmap_idx = 3

        # Temperature-related.
        if var_or_idx in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax, cfg.idx_etr, cfg.idx_tgg,
                          cfg.idx_tng, cfg.idx_tnx, cfg.idx_txx, cfg.idx_txg]:
            cmap_name = cfg.opt_map_col_temp_var[cmap_idx]
        elif var_or_idx in [cfg.idx_txdaysabove, cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen, cfg.idx_hotspellfreq,
                            cfg.idx_hotspellmaxlen, cfg.idx_tropicalnights, cfg.idx_tx90p, cfg.idx_wsdi]:
            cmap_name = cfg.opt_map_col_temp_idx_1[cmap_idx]
        elif var_or_idx in [cfg.idx_tndaysbelow, cfg.idx_tngmonthsbelow]:
            cmap_name = cfg.opt_map_col_temp_idx_2[cmap_idx]

        # Precipitation-related.
        elif var_or_idx in [cfg.var_cordex_pr, cfg.idx_prcptot, cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_sdii,
                            cfg.idx_rainqty]:
            cmap_name = cfg.opt_map_col_prec_var[cmap_idx]
        elif var_or_idx in [cfg.idx_cwd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_wetdays, cfg.idx_raindur, cfg.idx_rnnmm]:
            cmap_name = cfg.opt_map_col_prec_idx_1[cmap_idx]
        elif var_or_idx in [cfg.idx_cdd, cfg.idx_drydays, cfg.idx_dc, cfg.idx_drydurtot]:
            cmap_name = cfg.opt_map_col_prec_idx_2[cmap_idx]
        elif var_or_idx in [cfg.idx_rainstart, cfg.idx_rainend]:
            cmap_name = cfg.opt_map_col_prec_idx_3[cmap_idx]

        # Wind-related.
        elif var_or_idx in [cfg.var_cordex_uas, cfg.var_cordex_vas, cfg.var_cordex_sfcwindmax]:
            cmap_name = cfg.opt_map_col_wind_var[cmap_idx]
        elif var_or_idx in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
            cmap_name = cfg.opt_map_col_wind_idx_1[cmap_idx]

        # Default values.
        else:
            cmap_name = cfg.opt_map_col_default[cmap_idx]

        # Adjust minimum and maximum values so that zero is attributed the intermediate color in a scale or
        # use only the positive or negative side of the color scale if the other part is not required.
        if (z_min < 0) and (z_max > 0):
            vmax_abs = max(abs(z_min), abs(z_max))
            vmin = -vmax_abs
            vmax = vmax_abs
            n_cluster = n_cluster * 2
        else:
            vmin = z_min
            vmax = z_max

        # Custom color maps (not in matplotlib). The order assumes a vertical color bar.
        hex_wh  = "#ffffff"  # White.
        hex_gy  = "#808080"  # Grey.
        hex_gr  = "#008000"  # Green.
        hex_yl  = "#ffffcc"  # Yellow.
        hex_or  = "#f97306"  # Orange.
        hex_br  = "#662506"  # Brown.
        hex_rd  = "#ff0000"  # Red.
        hex_pi  = "#ffc0cb"  # Pink.
        hex_pu  = "#800080"  # Purple.
        hex_bu  = "#0000ff"  # Blue.
        hex_lbu = "#7bc8f6"  # Light blue.
        hex_lbr = "#d2b48c"  # Light brown.
        hex_sa  = "#a52a2a"  # Salmon.
        hex_tu  = "#008080"  # Turquoise.

        hex_list = None
        if "Pinks" in cmap_name:
            hex_list = [hex_wh, hex_pi]
        elif "PiPu" in cmap_name:
            hex_list = [hex_pi, hex_wh, hex_pu]
        elif "Browns" in cmap_name:
            hex_list = [hex_wh, hex_br]
        elif "YlBr" in cmap_name:
            hex_list = [hex_yl, hex_br]
        elif "BrYlGr" in cmap_name:
            hex_list = [hex_br, hex_yl, hex_gr]
        elif "YlGr" in cmap_name:
            hex_list = [hex_yl, hex_gr]
        elif "BrWhGr" in cmap_name:
            hex_list = [hex_br, hex_wh, hex_gr]
        elif "TuYlSa" in cmap_name:
            hex_list = [hex_tu, hex_yl, hex_sa]
        elif "YlTu" in cmap_name:
            hex_list = [hex_yl, hex_tu]
        elif "YlSa" in cmap_name:
            hex_list = [hex_yl, hex_sa]
        elif "LBuWhLBr" in cmap_name:
            hex_list = [hex_lbu, hex_wh, hex_lbr]
        elif "LBlues" in cmap_name:
            hex_list = [hex_wh, hex_lbu]
        elif "BuYlRd" in cmap_name:
            hex_list = [hex_bu, hex_yl, hex_rd]
        elif "LBrowns" in cmap_name:
            hex_list = [hex_wh, hex_lbr]
        elif "LBuYlLBr" in cmap_name:
            hex_list = [hex_lbu, hex_yl, hex_lbr]
        elif "YlLBu" in cmap_name:
            hex_list = [hex_yl, hex_lbu]
        elif "YlLBr" in cmap_name:
            hex_list = [hex_yl, hex_lbr]
        elif "YlBu" in cmap_name:
            hex_list = [hex_yl, hex_bu]
        elif "Turquoises" in cmap_name:
            hex_list = [hex_wh, hex_tu]
        elif "PuYlOr" in cmap_name:
            hex_list = [hex_pu, hex_yl, hex_or]
        elif "YlOrRd" in cmap_name:
            hex_list = [hex_yl, hex_or, hex_rd]
        elif "YlOr" in cmap_name:
            hex_list = [hex_yl, hex_or]
        elif "YlPu" in cmap_name:
            hex_list = [hex_yl, hex_pu]
        elif "GyYlRd" in cmap_name:
            hex_list = [hex_gy, hex_yl, hex_rd]
        elif "YlGy" in cmap_name:
            hex_list = [hex_yl, hex_gy]
        elif "YlRd" in cmap_name:
            hex_list = [hex_yl, hex_rd]
        elif "GyWhRd" in cmap_name:
            hex_list = [hex_gy, hex_wh, hex_rd]

        # Build custom map.
        if hex_list is not None:

            # List of positions.
            if len(hex_list) == 2:
                pos_list = [0.0, 1.0]
            else:
                pos_list = [0.0, 0.5, 1.0]

            # Custom map.
            if "_r" not in cmap_name:
                cmap = build_custom_cmap(hex_list, n_cluster, pos_list)
            else:
                hex_list.reverse()
                cmap = build_custom_cmap(hex_list, n_cluster, pos_list)

        # Build Matplotlib map.
        else:
            cmap = plt.cm.get_cmap(cmap_name, n_cluster)

        # Calculate ticks.
        ticks = None
        str_ticks = None
        if cfg.opt_map_discrete:
            ticks = []
            for i in range(n_cluster + 1):
                tick = i / float(n_cluster) * (vmax - vmin) + vmin
                ticks.append(tick)

            # Adjust tick precision.
            str_ticks = adjust_precision(ticks, n_dec_max)

        # Adjust minimum and maximum values.
        if ticks is None:
            vmin_adj = vmin
            vmax_adj = vmax
        else:
            vmin_adj = ticks[0]
            vmax_adj = ticks[n_cluster]

        # Create figure.
        plt.figure(figsize=(4.5, 4), dpi=dpi)
        plt.subplots_adjust(top=0.92, bottom=0.145, left=0.14, right=0.80, hspace=0.0, wspace=0.05)

        # Add mesh.
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 1])
        ax = plt.subplot(gs[0])
        cbar_ax = plt.subplot(gs[1])
        da.plot.pcolormesh(ax=ax, cbar_ax=cbar_ax, add_colorbar=True, add_labels=True,
                           cbar_kwargs=dict(orientation='vertical', pad=0.05, label=label, ticks=ticks),
                           cmap=cmap, vmin=vmin_adj, vmax=vmax_adj)

        # Format.
        ax.set_title(title, fontsize=fs_title)
        ax.set_xlabel("Longitude (º)", fontsize=fs_labels)
        ax.set_ylabel("Latitude (º)", fontsize=fs_labels)
        ax.tick_params(axis="x", labelsize=fs_ticks, length=0, rotation=90)
        ax.tick_params(axis="y", labelsize=fs_ticks, length=0)
        cbar_ax.set_ylabel(label, fontsize=fs_labels)
        cbar_ax.tick_params(labelsize=fs_ticks_cbar, length=0)
        if cfg.opt_map_discrete:
            cbar_ax.set_yticklabels(str_ticks)

        # Draw region boundary.
        draw_region_boundary(ax)

        # Save figure.
        if p_fig != "":
            utils.save_plot(plt, p_fig)

        plt.close()


def adjust_precision(vals: [float], n_dec_max: int = 4):

    """
    --------------------------------------------------------------------------------------------------------------------
    Adjust the precision of float values in a list so that each value is different than the following one.

    Parameters
    ----------
    vals : [float]
        List of values.
    n_dec_max : int, optional
        Maximum number of decimal places.
    --------------------------------------------------------------------------------------------------------------------
    """

    str_vals = None

    # Loop through potential numbers of decimal places.
    for n_dec in range(0, n_dec_max):

        # Loop through values.
        unique_vals = True
        str_vals = []
        for i in range(len(vals)):
            val = vals[i]
            if n_dec == 0:
                str_val = str(int(round(val, n_dec)))
                str_vals.append(str_val)
            else:
                str_val = str(round(val, n_dec))
                str_vals.append(str("{:." + str(n_dec) + "f}").format(float(str_val)))

            # Two consecutive rounded values are equal.
            if i > 0:
                if str_vals[i - 1] == str_vals[i]:
                    unique_vals = False

        # Stop loop if all values are unique.
        if unique_vals or (n_dec == n_dec_max):
            break

    return str_vals


def draw_region_boundary(ax):

    """
    --------------------------------------------------------------------------------------------------------------------
    Draw region boundary.

    Parameters
    ----------
    ax : Any
        ...
    --------------------------------------------------------------------------------------------------------------------
    """

    def configure_plot(ax):

        ax.set_aspect("equal")
        ax.set_anchor("C")

        return ax

    def set_plot_extent(ax, vertices):

        # Extract limits.
        x_min = x_max = y_min = y_max = None
        for i in range(len(vertices)):
            x_i = vertices[i][0]
            y_i = vertices[i][1]
            if i == 0:
                x_min = x_max = x_i
                y_min = y_max = y_i
            else:
                x_min = min(x_i, x_min)
                x_max = max(x_i, x_max)
                y_min = min(y_i, y_min)
                y_max = max(y_i, y_max)

        # Set the graph axes to the feature extents
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    def plot_feature(coordinates, myplot):

        poly = {"type": "Polygon", "coordinates": coordinates}
        patch = PolygonPatch(poly, fill=False, ec="black", alpha=0.75, zorder=2)
        myplot.add_patch(patch)

    # Read geojson file.
    with open(cfg.d_bounds) as f:
        pydata = simplejson.load(f)

    # Draw feature.
    myplot = configure_plot(ax)
    coordinates = pydata["features"][0]["geometry"]["coordinates"][0]
    vertices = coordinates[0]
    if len(vertices) == 2:
        coordinates = pydata["features"][0]["geometry"]["coordinates"]
        vertices = coordinates[0]
    set_plot_extent(myplot, vertices)
    plot_feature(coordinates, myplot)


def hex_to_rgb(value):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts hex to RGB colours

    Parameters
    ----------
    value: str
        String of 6 characters representing a hex colour.

    Returns
    -------
        list of 3 RGB values
    --------------------------------------------------------------------------------------------------------------------
    """

    value = value.strip("#")
    lv = len(value)

    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts RGB to decimal colours (i.e. divides each value by 256)

    Parameters
    ----------
    value: [int]
        List of 3 RGB values.

    Returns
    -------
        List of 3 decimal values.
    --------------------------------------------------------------------------------------------------------------------
    """

    return [v/256 for v in value]


def build_custom_cmap(hex_list: [str], n_cluster: int, pos_list: [float]=None):

    """
    --------------------------------------------------------------------------------------------------------------------
    Create a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: [str]
        List of hex code strings.
    n_cluster: int
        Number of clusters.
    pos_list: [float]
        List of positions (float between 0 and 1), same length as hex_list. Must start with 0 and end with 1.

    Returns
    -------
        Colour map.
    --------------------------------------------------------------------------------------------------------------------
    """

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if pos_list:
        pass
    else:
        pos_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[pos_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(pos_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap("custom_cmp", segmentdata=cdict, N=n_cluster)

    return cmp


def plot_ts(ds_ref: xr.Dataset, ds_rcp_26: [xr.Dataset], ds_rcp_45: [xr.Dataset], ds_rcp_85: [xr.Dataset],
            stn: str, var_or_idx_code: str, rcps: [str], ylim: [int], p_fig: str, mode: int = 1):

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
    var_or_idx_code : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index code (ex: cfg.idx_txdaysabove).
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

    var_or_idx = var_or_idx_code if (var_or_idx_code in cfg.variables_cordex) else cfg.extract_idx(var_or_idx_code)

    # Get title and label.
    title = cfg.get_plot_title(stn, var_or_idx_code)
    label = cfg.get_plot_ylabel(var_or_idx)

    # Add precision in title.
    y_param = None
    if var_or_idx not in cfg.variables_cordex:
        y_param_str = str(cfg.idx_params[cfg.idx_codes.index(var_or_idx_code)][0])
        if (var_or_idx == cfg.idx_prcptot) and (y_param_str != "nan"):
            y_param = float(y_param_str)
            title += " (seuil à " + str(int(round(y_param, 0))) + cfg.unit_mm + ")"

    # Fonts.
    fs_sup_title = 9

    # Initialize plot.
    f, ax = plt.subplots()
    ax.set_title(title, fontsize=fs_sup_title)
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

    # Loop through RCPs.
    rcps_copy = rcps.copy()
    if cfg.rcp_ref in rcps_copy:
        rcps_copy.remove(cfg.rcp_ref)
        rcps_copy = rcps_copy + [cfg.rcp_ref]
    for rcp in rcps_copy:

        # Skip if no simulation is available for this RCP.
        if ((rcp == cfg.rcp_26) and (ds_rcp_26 == [])) or\
           ((rcp == cfg.rcp_45) and (ds_rcp_45 == [])) or \
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
    custom_lines = [Line2D([0], [0], color="black", lw=4)]
    if cfg.rcp_26 in rcps:
        custom_lines.append(Line2D([0], [0], color="blue", lw=4))
    if cfg.rcp_45 in rcps:
        custom_lines.append(Line2D([0], [0], color="green", lw=4))
    if cfg.rcp_85 in rcps:
        custom_lines.append(Line2D([0], [0], color="red", lw=4))
    ax.legend(custom_lines, legend_list, loc="upper left", frameon=False)
    plt.ylim(ylim[0], ylim[1])

    # Add horizontal line.
    if y_param is not None:
        plt.axhline(y=y_param, color="black", linestyle="dashed", label=str(y_param))

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
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

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
        title = os.path.basename(p_fut).replace("4qqmap" + cfg.f_ext_nc, "verif_ts_single")
        plt.suptitle(title, fontsize=fs_title)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.title("")
        plt.suptitle(title, fontsize=fs_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)

        # Save plot.
        p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/verif/ts_single", var) + title + cfg.f_ext_png
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
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

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
        plt.ylabel(var_desc + " (" + var_unit + ")", fontsize=fs_axes)
        title = os.path.basename(p_list[i]).replace(cfg.f_ext_nc, "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        if i == 0:
            plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
            sup_title = title + "_verif_ts_mosaic"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Save plot.
    p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/verif/ts_mosaic", var) + title + cfg.f_ext_png
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_freq(ds_list: List[xr.Dataset], var: str, freq: str, title: str, plt_type: int = 0, p_fig: str = ""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (all in one plot).

    Parameters:
    ----------
    ds_list: List[xr.Dataset]
        List of datasets (mean, minimum and maximum).
    var: str
        Weather variable.
    freq: str
        Frequency.
    title: str
        Plot title.
    plt_type: int
        Plot type {0=automatically selected, 1=line, 2=bar}
        If the value
    p_fig: str
        Path of figure
    --------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_desc(var) + " (" + str.upper(var) + ")"
    var_unit = cfg.get_unit(var)

    # Plot.
    fs_title  = 8
    fs_legend = 8
    fs_axes   = 8

    # Number of values on the x-axis.
    n = len(list(ds_list[0][var].values))

    # Draw curve (mean values) and shadow (zone between minimum and maximum values).
    f, ax = plt.subplots()

    if freq == cfg.freq_MS:
        f.set_size_inches(4, 3)
        plt.subplots_adjust(top=0.93, bottom=0.13, left=0.13, right=0.97, hspace=0.10, wspace=0.10)
    else:
        f.set_size_inches(12, 3)
        plt.subplots_adjust(top=0.93, bottom=0.13, left=0.04, right=0.99, hspace=0.10, wspace=0.10)

    # Select colors.
    col_2cla = cfg.opt_plot_col_2cla_temp
    if var == cfg.var_cordex_pr:
        col_2cla = cfg.opt_plot_col_2cla_prec
    elif var in [cfg.var_cordex_uas, cfg.var_cordex_vas, cfg.var_cordex_sfcwindmax]:
        col_2cla = cfg.opt_plot_col_2cla_wind

    # Draw areas.
    if plt_type == 1:
        ax.plot(range(1, n + 1), list(ds_list[0][var].values), color=cfg.col_ref, alpha=1.0)
        ax.fill_between(np.array(range(1, n + 1)), list(ds_list[0][var].values), list(ds_list[2][var].values),
                        color=col_2cla[1], alpha=1.0)
        ax.fill_between(np.array(range(1, n + 1)), list(ds_list[0][var].values), list(ds_list[1][var].values),
                        color=col_2cla[0], alpha=1.0)
    else:
        bar_width = 1.0
        plt.bar(range(1, n + 1), list(ds_list[2][var].values), width=bar_width, color=col_2cla)
        plt.bar(range(1, n + 1), list(ds_list[0][var].values), width=bar_width, color=col_2cla)
        plt.bar(range(1, n + 1), list(ds_list[1][var].values), width=bar_width, color="white")
        ax.plot(range(1, n + 1), list(ds_list[0][var].values), color=cfg.col_ref, alpha=1.0)
        y_lim_lower = min(list(ds_list[1][var].values))
        y_lim_upper = max(list(ds_list[2][var].values))
        plt.ylim([y_lim_lower, y_lim_upper])

    # Format.
    plt.xlim([1, n])
    plt.xticks(np.arange(1, n + 1, 1 if freq == cfg.freq_MS else 30))
    plt.xlabel("Mois" if freq == cfg.freq_MS else "Journée", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.title(title, fontsize=fs_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)
    plt.suptitle("", fontsize=fs_title)

    # Format.
    plt.legend(["Moyenne", ">Moyenne", "<Moyenne"], fontsize=fs_legend)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_boxplot(ds: xr.Dataset, var: str, title: str, p_fig: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (all in one plot).

    Parameters:
    ----------
    ds_list: xr.Dataset
        Dataset (2D: month and year).
    var: str
        Weather variable.
    title: str
        Plot title.
    p_fig: str
        Path of figure
    --------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_desc(var)
    var_unit = cfg.get_unit(var)

    # Collect data.
    data = []
    for m in range(1, 13):
        data.append(ds[var].values[m - 1])

    # Draw.
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, showfliers=False)

    # Format.
    fs = 10
    plt.title(title, fontsize=fs)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
               ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], rotation=0)
    plt.xlabel("Mois", fontsize=fs)
    plt.ylabel(var_desc + " (" + var_unit + ")", fontsize=fs)
    plt.setp(bp["medians"], color="black")
    plt.show(block=False)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close("all")

