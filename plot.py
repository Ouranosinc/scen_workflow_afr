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
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.polynomial.polynomial as poly
import os.path
import pandas as pd
import utils
import warnings
import xarray as xr
from scipy import signal

import sys
sys.path.append("dashboard")
from dashboard import varidx_def as vi, rcp_def


# ======================================================================================================================
# Aggregation
# ======================================================================================================================

def plot_year(
    ds_hour: xr.Dataset,
    ds_day: xr.Dataset,
    var: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Time series.

    Parameters
    ----------
    ds_hour : xr.Dataset
        Dataset with hourly frequency.
    ds_day : xr.Dataset
        Dataset with daily frequency.
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    y_label = vi.VarIdx(var).get_label()

    fs = 10
    f = plt.figure(figsize=(10, 3))
    f.suptitle("Comparison entre les données horaires et agrégées", fontsize=fs)

    plt.subplots_adjust(top=0.9, bottom=0.20, left=0.10, right=0.98, hspace=0.30, wspace=0.416)
    ds_hour.plot(color="black")
    plt.title("")
    plt.xlabel("", fontsize=fs)
    plt.ylabel(y_label, fontsize=fs)

    ds_day.plot(color="orange")
    plt.title("")
    plt.xlabel("", fontsize=fs)
    plt.ylabel(y_label, fontsize=fs)

    plt.close()


def plot_dayofyear(
    ds_day: xr.Dataset,
    var: str,
    date: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Map.

    Parameters
    ----------
    ds_day : xr.Dataset
        Dataset with daily frequency.
    var : str
        Variable.
    date : str
        Date.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Data.
    ds_day_sel = ds_day.sel(time=date)

    # Plot.
    fs = 10
    y_label = vi.VarIdx(var).get_label()
    ds_day_sel.plot.pcolormesh(add_colorbar=True, add_labels=True,
                               cbar_kwargs=dict(orientation="vertical", pad=0.05, shrink=1, label=y_label))
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


def plot_postprocess(
    p_obs: str,
    p_fut: str,
    p_qqmap: str,
    var: str,
    p_fig: str,
    title: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of observed and future periods.

    Parameters
    ----------
    p_obs : str
        Path of NetCDF file containing observations.
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
    da_obs = utils.open_netcdf(p_obs)[var]
    if cfg.dim_longitude in da_obs.dims:
        da_obs = da_obs.rename({cfg.dim_longitude: cfg.dim_rlon, cfg.dim_latitude: cfg.dim_rlat})
    da_fut = utils.open_netcdf(p_fut)[var]
    da_qqmap = utils.open_netcdf(p_qqmap)[var]

    # Select control point.
    if cfg.opt_ra:
        subset_ctrl_pt = False
        if cfg.dim_rlon in da_obs.dims:
            subset_ctrl_pt = (len(da_obs.rlon) > 1) or (len(da_obs.rlat) > 1)
        elif cfg.dim_lon in da_obs.dims:
            subset_ctrl_pt = (len(da_obs.lon) > 1) or (len(da_obs.lat) > 1)
        elif cfg.dim_longitude in da_obs.dims:
            subset_ctrl_pt = (len(da_obs.longitude) > 1) or (len(da_obs.latitude) > 1)
        if subset_ctrl_pt:
            if cfg.p_bounds == "":
                da_obs   = utils.subset_ctrl_pt(da_obs)
                da_fut   = utils.subset_ctrl_pt(da_fut)
                da_qqmap = utils.subset_ctrl_pt(da_qqmap)
            else:
                da_obs   = utils.squeeze_lon_lat(da_obs)
                da_fut   = utils.squeeze_lon_lat(da_fut)
                da_qqmap = utils.squeeze_lon_lat(da_qqmap)

    # Conversion coefficient.
    coef = 1
    delta_obs   = 0
    delta_fut   = 0
    delta_qqmap = 0
    if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
        coef = cfg.spd * 365
    elif var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
        if da_obs.units == cfg.unit_K:
            delta_obs = -cfg.d_KC
        if da_fut.units == cfg.unit_K:
            delta_fut = -cfg.d_KC
        if da_qqmap.units == cfg.unit_K:
            delta_qqmap = -cfg.d_KC

    # Calculate annual mean values.
    da_qqmap_mean = None
    if da_qqmap is not None:
        da_qqmap_mean = (da_qqmap * coef + delta_qqmap).groupby(da_qqmap.time.dt.year).mean()
    da_fut_mean = (da_fut * coef + delta_fut).groupby(da_fut.time.dt.year).mean()
    da_obs_mean = (da_obs * coef + delta_obs).groupby(da_obs.time.dt.year).mean()

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
        da_qqmap_mean.plot.line(color=cfg.col_sim_adj)
    da_fut_mean.plot.line(color=cfg.col_sim_fut)
    da_obs_mean.plot(color=cfg.col_obs)

    # Customize.
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    y_label = vi.VarIdx(var).get_label()
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()

    # Save to CSV.
    if cfg.opt_save_csv[0]:
        p_csv = p_fig.replace(cfg.sep + var + cfg.sep, cfg.sep + var + "_" + cfg.f_csv + cfg.sep). \
            replace(cfg.f_ext_png, cfg.f_ext_csv)
        n_obs_mean = len(list(da_obs_mean.values))
        n_qqmap_mean = len(list(da_qqmap_mean.values))
        dict_pd = {"year": list(range(1, n_qqmap_mean + 1)),
                   "obs": list(da_obs_mean.values) + [np.nan] * (n_qqmap_mean - n_obs_mean),
                   "qqmap": list(da_qqmap_mean.values),
                   "fut": list(da_fut_mean.values)}
        df = pd.DataFrame(dict_pd)
        utils.save_csv(df, p_csv)


def plot_workflow(
    var: str,
    nq: int,
    up_qmf: float,
    time_win: int,
    p_regrid_ref: str,
    p_regrid_fut: str,
    p_fig: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of reference and future periods.

    Parameters
    ----------
    var : str
        Weather var.
    nq : int
        Number of quantiles.
    up_qmf : float
        Upper limit of quantile mapping function.
    time_win : int
        Window.
    p_regrid_ref : str
        Path of the NetCDF file containing data for the reference period.
    p_regrid_fut : str
        Path of the NetCDF file containing data for the future period.
    p_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

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
            if cfg.p_bounds == "":
                da_ref = utils.subset_ctrl_pt(da_ref)
                da_fut = utils.subset_ctrl_pt(da_fut)
            else:
                da_ref = utils.squeeze_lon_lat(da_ref)
                da_fut = utils.squeeze_lon_lat(da_fut)

    # Conversion coefficients.
    coef = 1
    delta_ref = 0
    delta_fut = 0
    if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
        coef = cfg.spd
    if var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
        if da_ref.units == cfg.unit_K:
            delta_ref = -cfg.d_KC
        if da_fut.units == cfg.unit_K:
            delta_fut = -cfg.d_KC

    # Fit.
    x     = [*range(len(da_ref.time))]
    y     = (da_ref * coef + delta_ref).values
    coefs = poly.polyfit(x, y, 4)
    ffit  = poly.polyval(x, coefs)

    # Font size.
    fs_sup_title = 8
    fs_title     = 8
    fs_legend    = 6
    fs_axes      = 8

    y_label = vi.VarIdx(var).get_label()

    # Initialize plot.
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
        plt.ylabel(y_label, fontsize=fs_axes)
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
        plt.ylabel(y_label, fontsize=fs_axes)
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


def plot_calib(
    da_obs: xr.DataArray,
    da_ref: xr.DataArray,
    da_fut: xr.DataArray,
    da_qqmap: xr.DataArray,
    da_qqmap_ref: xr.DataArray,
    da_qmf: xr.DataArray,
    var: str,
    sup_title: str,
    p_fig: str
):

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

    # Files.
    p_csv = p_fig.replace(cfg.sep + var + cfg.sep, cfg.sep + var + "_" + cfg.f_csv + cfg.sep).\
        replace(cfg.f_ext_png, cfg.f_ext_csv)

    # Quantile ---------------------------------------------------------------------------------------------------------

    # Font size.
    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 5
    fs_axes      = 7

    y_label = vi.VarIdx(var).get_label()

    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(431)
    plt.subplots_adjust(top=0.930, bottom=0.065, left=0.070, right=0.973, hspace=0.90, wspace=0.250)

    # Quantile mapping function.
    img1 = ax.imshow(da_qmf, extent=[0, 1, 365, 1], cmap="coolwarm")
    cb = f.colorbar(img1, ax=ax)
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
    draw_curves(var, da_obs, da_ref, da_fut, da_qqmap, da_qqmap_ref, cfg.stat_mean, -1, p_csv)
    plt.title("Moyenne", fontsize=fs_title)
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlim([1, 12])
    plt.xticks(np.arange(1, 13, 1))
    plt.xlabel("Mois", fontsize=fs_axes)
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Mean, Q100, Q99, Q75, Q50, Q25, Q01 and Q00 monthly values -------------------------------------------------------

    for i in range(1, len(cfg.opt_calib_quantiles) + 1):

        plt.subplot(433 + i - 1)

        stat  = "quantile"
        q     = cfg.opt_stat_quantiles[i-1]
        title = "Q_" + "{0:.2f}".format(q)
        if q == 0:
            stat = cfg.stat_min
        elif q == 1:
            stat = cfg.stat_max

        draw_curves(var, da_obs, da_ref, da_fut, da_qqmap, da_qqmap_ref, stat, q)

        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(y_label, fontsize=fs_axes)
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
    plt.ylabel(y_label, fontsize=fs_axes)
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


def plot_calib_ts(
    da_obs: xr.DataArray,
    da_fut: xr.DataArray,
    da_qqmap: xr.DataArray,
    var: str,
    title: str,
    p_fig: str
):

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

    # Font size.
    fs_sup_title = 8
    fs_legend = 8
    fs_axes = 8

    y_label = vi.VarIdx(var).get_label()

    # Initialize plot.
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
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Save plot.
    if p_fig != "":
        utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()

    # Save to CSV.
    if cfg.opt_save_csv[0]:
        p_csv = p_fig.replace(cfg.sep + var + cfg.sep, cfg.sep + var + "_" + cfg.f_csv + cfg.sep).\
            replace(cfg.f_ext_png, cfg.f_ext_csv)
        n_obs = len(list(da_obs.values))
        n_qqmap = len(list(da_qqmap.values))
        dict_pd = {"day": list(range(1, n_qqmap + 1)),
                   "obs": list(da_obs.values) + [np.nan] * (n_qqmap - n_obs),
                   "qqmap": list(da_qqmap.values),
                   "fut": list(da_fut.values)}
        df = pd.DataFrame(dict_pd)
        utils.save_csv(df, p_csv)


def draw_curves(
    var: str,
    da_obs: xr.DataArray,
    da_ref: xr.DataArray,
    da_fut: xr.DataArray,
    da_qqmap: xr.DataArray,
    da_qqmap_ref: xr.DataArray,
    stat: str,
    q: float,
    p_csv: str = ""
):

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
    q : float, optional
        Quantile.
    p_csv : str
        Path of CSV file to create.
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
    stat_actual = stat
    if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
        if stat == cfg.stat_mean:
            stat_actual = cfg.stat_sum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            da_obs       = da_obs.resample(time="1M").sum()
            da_ref       = da_ref.resample(time="1M").sum()
            da_fut       = da_fut.resample(time="1M").sum()
            da_qqmap     = da_qqmap.resample(time="1M").sum()
            da_qqmap_ref = da_qqmap_ref.resample(time="1M").sum()

    # Calculate statistics.
    da_obs       = da_groupby(da_obs, stat_actual, q)
    da_ref       = da_groupby(da_ref, stat_actual, q)
    da_fut       = da_groupby(da_fut, stat_actual, q)
    da_qqmap     = da_groupby(da_qqmap, stat_actual, q)
    da_qqmap_ref = da_groupby(da_qqmap_ref, stat_actual, q)

    # Draw curves.
    da_obs.plot.line(color=cfg.col_obs)
    da_ref.plot.line(color=cfg.col_sim_ref)
    da_fut.plot.line(color=cfg.col_sim_fut)
    da_qqmap.plot.line(color=cfg.col_sim_adj)
    da_qqmap_ref.plot.line(color=cfg.col_sim_adj_ref)

    # Save to CSV.
    if cfg.opt_save_csv[0] and (p_csv != ""):
        if stat in [cfg.stat_mean, cfg.stat_min, cfg.stat_max, cfg.stat_sum]:
            suffix = stat
        else:
            suffix = "q" + str(round(q * 100)).zfill(2)
        p_csv = p_csv.replace(cfg.f_ext_csv, "_" + suffix + cfg.f_ext_csv)
        dict_pd = {"month": list(range(1, 13)),
                   "obs": list(da_obs.values),
                   "ref": list(da_ref.values),
                   "fut": list(da_fut.values),
                   "qqmap": list(da_qqmap.values),
                   "qqmap_ref": list(da_qqmap_ref.values)}
        df = pd.DataFrame(dict_pd)
        utils.save_csv(df, p_csv)


def plot_360_vs_365(
    ds_360: xr.Dataset,
    ds_365: xr.Dataset,
    var: str = ""
):

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


def plot_rsq(
    rsq: np.array,
    n_sim: int
):

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
# Verification
# ======================================================================================================================


def plot_ts_single(
    stn: str,
    var: str
):

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

    utils.log("Processing (single): " + stn + ", " + var, True)

    # Paths and NetCDF files.
    d_regrid = cfg.get_d_scen(stn, cfg.cat_regrid, var)
    p_l   = utils.list_files(d_regrid)
    p_obs    = cfg.get_p_obs(stn, var)

    # Font size.
    fs_title  = 8
    fs_legend = 8
    fs_axes   = 8

    y_label = vi.VarIdx(var).get_label()

    # Initialize plot.
    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.18, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

    # Loop through simulation sets.
    for i in range(int(len(p_l) / 3)):

        p_ref   = [i for i in p_l if rcp_def.rcp_ref in i][i]
        p_fut   = p_ref.replace(rcp_def.rcp_ref + "_", "")
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
        plt.ylabel(y_label, fontsize=fs_axes)
        plt.title("")
        plt.suptitle(title, fontsize=fs_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)

        # Save plot.
        p_fig = cfg.get_d_scen(stn, cfg.cat_fig + cfg.sep + "verif" + cfg.sep + "ts_single", var) +\
            title + cfg.f_ext_png
        utils.save_plot(plt, p_fig)

        # Close plot.
        plt.close()


def plot_ts_mosaic(
    stn: str,
    var: str
):

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

    utils.log("Processing (mosaic): " + stn + ", " + var, True)

    # Conversion coefficient.
    coef = 1
    if var == vi.v_pr:
        coef = cfg.spd

    # Paths and NetCDF files.
    d_regrid = cfg.get_d_scen(stn, cfg.cat_regrid, var)
    p_l      = utils.list_files(d_regrid)
    p_obs    = cfg.get_p_obs(stn, var)

    # Font size.
    fs_title  = 6
    fs_legend = 6
    fs_axes   = 6

    y_label = vi.VarIdx(var).get_label()

    # Initialize plot.
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.04, right=0.99, hspace=0.40, wspace=0.30)

    # Loop through simulation sets.
    title = ""
    for i in range(int(len(p_l) / 3)):

        # Path of NetCDF files.
        p_fut_i   = [i for i in p_l if rcp_def.rcp_ref in i][i].replace(rcp_def.rcp_ref + "_", "")
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
        plt.ylabel(y_label, fontsize=fs_axes)
        title = os.path.basename(p_l[i]).replace(cfg.f_ext_nc, "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        if i == 0:
            plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
            sup_title = title + "_verif_ts_mosaic"
            plt.suptitle(sup_title, fontsize=fs_title)

    # Save plot.
    p_fig = cfg.get_d_scen(stn, cfg.cat_fig + cfg.sep + "verif" + cfg.sep + "ts_mosaic", var) + title + cfg.f_ext_png
    utils.save_plot(plt, p_fig)

    # Close plot.
    plt.close()
