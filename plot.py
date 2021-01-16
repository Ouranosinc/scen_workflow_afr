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
import matplotlib.cbook
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
            n_years = da[cfg.dim_time].size / 12
            da_group_stat = da_group.sum(dim=cfg.dim_time) / n_years
        return da_group_stat

    # Determine if sum is needed.
    stat_inner = stat
    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
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

def plot_heatmap(da: xr.DataArray, stn: str, var_or_idx: str, grid_x: [float], grid_y: [float], rcp: str,
                 per: [int, int], z_min: float, z_max: float, p_fig: str, map_package: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a heat map of a climate index for the reference period and for emission scenarios.

    Parameters
    ----------
    da: xr.DataArray
        DataArray (with 2 dimensions: longitude and latitude).
    stn : str
        Station name.
    var_or_idx : str
        Climate variable (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_txdaysabove).
    grid_x: [float]
        X-coordinates.
    grid_y: [float]
        Y-coordinates.
    rcp: str
        RCP emission scenario.
    per: [int, int]
        Period of interest, for instance, [1981, 2010].
    z_min : float
        Minimum value (associated with color bar).
    z_max : float
        Maximum value (associated with color bar).
    p_fig : str
        Path of output figure.
    map_package: str
        Map package: {"seaborn", "matplotlib"}
    --------------------------------------------------------------------------------------------------------------------
    """

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

        p_fig_tif = p_fig.replace(var_or_idx + "/", var_or_idx + "_" + cfg.f_tif + "/").\
            replace(cfg.f_ext_png, cfg.f_ext_tif)
        d = os.path.dirname(p_fig_tif)
        if not (os.path.isdir(d)):
            os.makedirs(d)
        da_tif.rio.to_raster(p_fig_tif)

    # Export to PNG.
    if cfg.f_png in cfg.opt_map_formats:

        # Get title and label.
        title, label = get_title_label(stn, var_or_idx, rcp, per)

        plt.subplots_adjust(top=0.9, bottom=0.11, left=0.12, right=0.995, hspace=0.695, wspace=0.416)

        # Using seaborn (not tested).
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

            # Fonts.
            fs           = 10
            fs_sup_title = 9

            # Color mesh.
            if var_or_idx in [cfg.var_cordex_uas, cfg.var_cordex_vas]:
                cmap = "RdBu_r"
                vmax_abs = max(abs(z_min), abs(z_max))
                norm = colors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
            else:
                cmap = norm = None
            mesh = da.plot.pcolormesh(add_colorbar=True, add_labels=True,
                                      cbar_kwargs=dict(orientation='vertical', pad=0.05, label=label),
                                      cmap=cmap, norm=norm)
            if (z_min is not None) and (z_max is not None):
                mesh.set_clim(z_min, z_max)
            plt.title(title, fontsize=fs_sup_title)
            plt.suptitle("")
            plt.xlabel("Longitude (º)", fontsize=fs)
            plt.ylabel("Latitude (º)", fontsize=fs)
            plt.tick_params(axis="x", labelsize=fs)
            plt.tick_params(axis="y", labelsize=fs)
            plt.xlim(cfg.lon_bnds)
            plt.ylim(cfg.lat_bnds)

            # Draw region boundary.
            draw_region_boundary(mesh.axes)

        # Save figure.
        if p_fig != "":
            utils.save_plot(plt, p_fig)

        plt.close()


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

        fig = pyplot.figure(1, figsize=(10, 4), dpi=180)
        ax.set_aspect("equal")
        ax.set_anchor("C")

        return fig, ax

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
    fig, myplot = configure_plot(ax)
    coordinates = pydata["features"][0]["geometry"]["coordinates"][0]
    vertices = coordinates[0]
    if len(vertices) == 2:
        coordinates = pydata["features"][0]["geometry"]["coordinates"]
        vertices = coordinates[0]
    set_plot_extent(myplot, vertices)
    plot_feature(coordinates, myplot)


def get_title_label(stn: str, var_or_idx: str, rcp: str = None, per: [int] = None):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get title and label.

    Parameters
    ----------
    stn : str
        Station name.
    var_or_idx : str
        Climate variable  (ex: cfg.var_cordex_tasmax) or climate index (ex: cfg.idx_txdaysabove).
    rcp: str
        RCP emission scenario.
    per: [int, int], Optional
        Period of interest, for instance, [1981, 2010].
    --------------------------------------------------------------------------------------------------------------------
    """

    if var_or_idx in cfg.variables_cordex:

        title = cfg.get_var_desc(var_or_idx) + "\n(" + stn.capitalize() + \
                ("" if rcp is None else ", " + rcp) + \
                ("" if per is None else ", " + str(per[0]) + "-" + str(per[1]))
        title += ")"
        label = cfg.get_var_desc(var_or_idx) + " (" + cfg.get_var_unit(var_or_idx) + ")"

    # ==================================================================================================================
    # TODO.CUSTOMIZATION.INDEX.BEGIN
    # Generate title and label.
    # ==================================================================================================================

    else:

        title = cfg.get_idx_desc(var_or_idx) + "\n(" + stn.capitalize() + \
                ("" if rcp is None else ", " + rcp) + \
                ("" if per is None else ", " + str(per[0]) + "-" + str(per[1]))
        title += ")"
        label = ""

        # Temperature.
        if var_or_idx in [cfg.idx_txdaysabove, cfg.idx_tngmonthsbelow, cfg.idx_hotspellfreq, cfg.idx_hotspellmaxlen,
                          cfg.idx_heatwavemaxlen, cfg.idx_heatwavetotlen, cfg.idx_tropicalnights, cfg.idx_tx90p]:
            label = "Nbr"
            if var_or_idx == cfg.idx_tngmonthsbelow:
                label += " mois"
            elif var_or_idx != cfg.idx_hotspellfreq:
                label += " jours"

        elif var_or_idx == cfg.idx_txx:
            label = cfg.get_var_desc(cfg.var_cordex_tasmax) + " (" + cfg.get_var_unit(cfg.var_cordex_tasmax) + ")"

        elif var_or_idx in [cfg.idx_tnx, cfg.idx_tng]:
            label = cfg.get_var_desc(cfg.var_cordex_tasmin) + " (" + cfg.get_var_unit(cfg.var_cordex_tasmin) + ")"

        elif var_or_idx == cfg.idx_tgg:
            label = cfg.get_var_desc(cfg.var_cordex_tas) + " (" + cfg.get_var_unit(cfg.var_cordex_tas) + ")"

        elif var_or_idx == cfg.idx_etr:
            label = "Écart de température (" + cfg.get_var_unit(cfg.var_cordex_tas) + ")"

        elif var_or_idx == cfg.idx_wsdi:
            label = "Indice"

        elif var_or_idx == cfg.idx_dc:
            label = "Code"

        # Precipitation.
        elif var_or_idx in [cfg.idx_cwd, cfg.idx_cdd, cfg.idx_r10mm, cfg.idx_r20mm, cfg.idx_rnnmm, cfg.idx_wetdays,
                            cfg.idx_drydays, cfg.idx_raindur]:
            label = "Nbr jours"

        elif var_or_idx in [cfg.idx_rx1day, cfg.idx_rx5day, cfg.idx_sdii, cfg.idx_prcptot]:
            label = cfg.get_var_desc(cfg.var_cordex_pr) + " (" + cfg.get_var_unit(cfg.var_cordex_pr)
            if var_or_idx == cfg.idx_sdii:
                label += "/day"
            label += ")"

        elif var_or_idx in [cfg.idx_rainstart, cfg.idx_rainend]:
            label = "Jour de l'année"

        elif var_or_idx in [cfg.idx_wgdaysabove, cfg.idx_wxdaysabove]:
            label += "Nbr jours"

    # ==================================================================================================================
    # TODO.CUSTOMIZATION.INDEX.END
    # ==================================================================================================================

    return title, label


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

    var_or_idx = var_or_idx_code if (var_or_idx_code in cfg.variables_cordex) else cfg.get_idx_name(var_or_idx_code)

    # Get title and label.
    title, label = get_title_label(stn, var_or_idx)

    # Add precision in title.
    y_param = None
    if var_or_idx == cfg.idx_prcptot:
        y_param = cfg.idx_params[cfg.idx_names.index(var_or_idx)][0]
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        ds_plt = ds_obs.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().\
            groupby("time.month").mean()[var]

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                ds.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().\
                    groupby("time.month").mean().plot(color="blue")
        ds = utils.open_netcdf(p_list[i])[var]
        if isinstance(ds.time[0].values, np.datetime64):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                ds.sel(time=slice("2050-01-01", "2070-12-31")).resample(time="M").mean().\
                    groupby("time.month").mean().plot(color="green")
        ds_plt.plot(color="red")

        # Format.
        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        title = os.path.basename(p_list[i]).replace(cfg.f_ext_nc, "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        if i == 0:
            sup_title = title + "_verif_monthly"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Format.
    plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)

    # Save plot.
    p_fig = cfg.get_d_scen(stn, cfg.cat_fig + "/verif/monthly", var) + sup_title + cfg.f_ext_png
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()
