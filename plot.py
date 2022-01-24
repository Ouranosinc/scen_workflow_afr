# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Plot functions.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# 2. marc-andre.bourgault@ggr.ulaval.ca (original)
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import matplotlib.cbook
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.polynomial.polynomial as poly
import os.path
import pandas as pd
import sys
import warnings
import xarray as xr
from scipy import signal
from typing import List

# Workflow libraries.
import file_utils as fu
import utils
from def_constant import const as c
from def_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import def_varidx as vi

# Color associated with specific datasets (calibration plots).
col_obs         = "green"   # Observed data.
col_sim         = "purple"  # Simulation data (non-adjusted).
col_sim_ref     = "orange"  # Simulation data (non-adjusted, reference period).
col_sim_adj     = "red"     # Simulation data (bias-adjusted).
col_sim_adj_ref = "blue"    # Simulation data (bias-adjusted, reference period).


def plot_year(
    ds_hour: xr.Dataset,
    ds_day: xr.Dataset,
    var: vi.VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Time series.

    Parameters
    ----------
    ds_hour: xr.Dataset
        Dataset with hourly frequency.
    ds_day: xr.Dataset
        Dataset with daily frequency.
    var: vi.VarIdx
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    fs = 10
    f = plt.figure(figsize=(10, 3))
    f.suptitle("Comparison entre les données horaires et agrégées", fontsize=fs)

    plt.subplots_adjust(top=0.9, bottom=0.20, left=0.10, right=0.98, hspace=0.30, wspace=0.416)
    ds_hour.plot(color="black")
    plt.title("")
    plt.xlabel("", fontsize=fs)
    plt.ylabel(var.label, fontsize=fs)

    ds_day.plot(color="orange")
    plt.title("")
    plt.xlabel("", fontsize=fs)
    plt.ylabel(var.label, fontsize=fs)

    plt.close()


def plot_dayofyear(
    ds_day: xr.Dataset,
    var: vi.VarIdx,
    date: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Map.

    Parameters
    ----------
    ds_day: xr.Dataset
        Dataset with daily frequency.
    var: vi.VarIdx
        Variable.
    date: str
        Date.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Data.
    ds_day_sel = ds_day.sel(time=date)

    # Plot.
    fs = 10
    ds_day_sel.plot.pcolormesh(add_colorbar=True, add_labels=True,
                               cbar_kwargs=dict(orientation="vertical", pad=0.05, shrink=1,
                                                label=vi.VarIdx(var.name).label))
    plt.title(date)
    plt.suptitle("", fontsize=fs)
    plt.xlabel("Longitude (º)", fontsize=fs)
    plt.ylabel("Latitude (º)", fontsize=fs)
    plt.tick_params(axis="x", labelsize=fs)
    plt.tick_params(axis="y", labelsize=fs)

    plt.close()


def plot_postprocess(
    df: pd.DataFrame,
    varidx: vi.VarIdx,
    title: str
) -> plt.Figure:

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot allowing to compare adjusted and non-adjusted curves to reference dataset.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    varidx: vi.VarIdx
        Variable or index.
    title: str
        Title of figure.

    Returns
    -------
    plt.Figure
        Plot of time series.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Initialize plot.
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

    # Legend.
    legend_items = ["Simulation", "Référence"]
    if c.cat_sim_adj in df.columns:
        legend_items.insert(0, "Sim. ajustée")

    # Draw curves.
    if c.cat_sim_adj in df.columns:
        ax.plot.line(df, c.cat_sim_adj, color=col_sim_adj)
    ax.plot.line(df, c.cat_sim, color=col_sim)
    ax.plot(df, c.cat_obs, color=col_obs)

    # Font size.
    fs        = 8
    fs_title  = fs
    fs_legend = fs
    fs_axes   = fs

    # Customize.
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    y_label = varidx.label
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.title("")
    plt.suptitle(title, fontsize=fs_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Close plot.
    plt.close()

    return fig


def plot_workflow(
    df: xr.DataArray,
    varidx: vi.VarIdx,
    units: List[str],
    title: str
) -> plt.Figure:

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot allowing to compare reference and simulated data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    varidx: viVarIdx
        Variable or index.
    units: List[str]
        Units, of reference and simulation data.
    title: str
        Title

    Returns
    -------
    plt.Figure
        Plot of time series.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = varidx.name

    # Conversion coefficients.
    coef = 1
    delta_ref = 0
    delta_sim = 0
    if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
        coef = c.spd
    if (vi_name in [c.v_tas, c.v_tasmin, c.v_tasmax]) and (len(units) > 0):
        if units[0] == c.unit_K:
            delta_ref = -c.d_KC
        if units[1] == c.unit_K:
            delta_sim = -c.d_KC

    # Fit.
    x     = [*range(len(df["year"]))]
    y     = (df[c.cat_obs] * coef + delta_ref).values
    coefs = poly.polyfit(x, y, 4)
    ffit  = poly.polyval(x, coefs)

    # Font size.
    fs           = 8
    fs_sup_title = fs
    fs_title     = fs
    fs_legend    = fs - 2
    fs_axes      = fs

    # Initialize plot.
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.99, hspace=0.40, wspace=0.00)
    plt.suptitle(title, fontsize=fs_sup_title)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

        # Upper plot: Reference period.
        ax = fig.add_subplot(211)
        ax.plot(df, "year", y, color=col_sim_ref)
        ax.plot(df, "year", ffit, color="black")
        plt.legend(["Simulation (réf.)", "Tendance"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(varidx.label, fontsize=fs_axes)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        plt.title("Tendance", fontsize=fs_title)

        # Lower plot: Complete simulation.
        fig.add_subplot(212)
        arr_y_detrend = signal.detrend(df[c.cat_sim][np.isnan(df[c.cat_sim]) == False] * coef + delta_sim)
        arr_x_detrend = cntx.per_ref[0] + np.arange(0, len(arr_y_detrend), 1) / 365
        arr_y_error  = (y - ffit)
        arr_x_error = cntx.per_ref[0] + np.arange(0, len(arr_y_error), 1) / 365
        plt.plot(arr_x_detrend, arr_y_detrend, alpha=0.5, color=col_sim)
        plt.plot(arr_x_error, arr_y_error, alpha=0.5, color=col_sim_ref)
        plt.legend(["Simulation", "Simulation (réf.)"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Jours", fontsize=fs_axes)
        plt.ylabel(varidx.label, fontsize=fs_axes)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        plt.title("Variation autour de la moyenne (prédiction basée sur une équation quartique)", fontsize=fs_title)

    # Close plot.
    plt.close()

    return fig


def plot_calib(
    da_obs: xr.DataArray,
    da_sim_ref: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    da_sim_adj_ref: xr.DataArray,
    da_qmf: xr.DataArray,
    varidx: vi.VarIdx,
    sup_title: str,
    p_fig: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot containing a summary of calibration.

    Parameters
    ----------
    da_obs: xr.DataArray
        Observed data.
    da_sim_ref: xr.DataArray
        Simulation data for the reference period.
    da_sim: xr.DataArray
        Simulation data.
    da_sim_adj: xr.DataArray
        Adjusted simulation.
    da_sim_adj_ref: xr.DataArray
        Adjusted simulation for the reference period.
    da_qmf: xr.DataArray
        Quantile mapping function.
    varidx: vi.VarIdx
        Variable or index.
    sup_title: str
        Title of figure.
    p_fig: str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = str(varidx.name)

    # Paths.
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.f_csv + cntx.sep).\
        replace(c.f_ext_png, c.f_ext_csv)

    # Quantile ---------------------------------------------------------------------------------------------------------

    # Font size.
    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 5
    fs_axes      = 7

    y_label = varidx.label

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

    legend_items = ["Référence", "Sim. (réf.)", "Sim.", "Sim. ajustée", "Sim. ajustée (réf.)"]

    # Mean values ------------------------------------------------------------------------------------------------------

    # Plot.
    ax = f.add_subplot(432)
    draw_curves(ax, varidx, da_obs, da_sim_ref, da_sim, da_sim_adj, da_sim_adj_ref, c.stat_mean, -1, p_csv)
    plt.title("Moyenne", fontsize=fs_title)
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlim([1, 12])
    plt.xticks(np.arange(1, 13, 1))
    plt.xlabel("Mois", fontsize=fs_axes)
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Mean, Q100, Q99, Q75, Q50, Q25, Q01 and Q00 monthly values -------------------------------------------------------

    for i in range(1, len(cntx.opt_calib_quantiles) + 1):

        ax = plt.subplot(433 + i - 1)

        stat  = "quantile"
        q     = cntx.opt_stat_quantiles[i-1]
        title = "Q_" + "{0:.2f}".format(q)
        if q == 0:
            stat = c.stat_min
        elif q == 1:
            stat = c.stat_max

        draw_curves(ax, varidx, da_obs, da_sim_ref, da_sim, da_sim_adj, da_sim_adj_ref, stat, q)

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
    if da_sim_adj.time.time.dtype == c.dtype_obj:
        da_sim_adj[c.dim_time] = utils.reset_calendar(da_sim_adj.time)
    if da_sim_ref.time.time.dtype == c.dtype_obj:
        da_sim_ref[c.dim_time] = utils.reset_calendar(da_sim_ref.time)

    plt.subplot(313)
    da_sim_adj.plot.line(alpha=0.5, color=col_sim_adj)
    da_sim_ref.plot.line(alpha=0.5, color=col_sim_ref)
    da_obs.plot.line(alpha=0.5, color=col_obs)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.legend(["Sim. ajustée", "Sim. (réf.)", "Référence"], fontsize=fs_legend, frameon=False)
    plt.title("")

    f.suptitle(vi_name + "_" + sup_title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    if c.attrs_bias in da_sim_adj.attrs:
        del da_sim_adj.attrs[c.attrs_bias]

    # Save plot.
    if p_fig != "":
        fu.save_plot(plt, p_fig)

    # Close plot.
    plt.close()


def plot_calib_ts(
    da_obs: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    varidx: vi.VarIdx,
    title: str,
    p_fig: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot allowing to compare reference and simulation data.

    Parameters
    ----------
    da_obs: xr.DataArray
        Observed data.
    da_sim: xr.DataArray
        Simulation data.
    da_sim_adj: xr.DataArray
        Adjusted simulation data.
    varidx: vi.VarIdx
        Variable or index.
    title: str
        Title of figure.
    p_fig: str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = str(varidx.name)

    # Paths.
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.f_csv + cntx.sep). \
        replace(c.f_ext_png, c.f_ext_csv)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.f_png in cntx.opt_diagnostic_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_diagnostic_format)))
    if not (save_fig or save_csv):
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Convert date format if the need is.
        if da_obs.time.dtype == c.dtype_obj:
            da_obs[c.dim_time] = utils.reset_calendar(da_obs)
        if da_sim.time.dtype == c.dtype_obj:
            da_sim[c.dim_time] = utils.reset_calendar(da_sim)
        if da_sim_adj.time.dtype == c.dtype_obj:
            da_sim_adj[c.dim_time] = utils.reset_calendar(da_sim_adj)

        # Create dataframe.
        n_obs = len(list(da_obs.values))
        n_sim_adj = len(list(da_sim_adj.values))
        dict_pd = {"day": list(range(1, n_sim_adj + 1)),
                   c.cat_obs: list(da_obs.values) + [np.nan] * (n_sim_adj - n_obs),
                   c.cat_sim_adj: list(da_sim_adj.values),
                   c.cat_sim: list(da_sim.values)}
        df = pd.DataFrame(dict_pd)

    # Generate and save plot.
    if save_fig:

        # Font size.
        fs           = 8
        fs_sup_title = fs
        fs_legend    = fs
        fs_axes      = fs

        # Initialize plot.
        fig = plt.figure(figsize=(15, 3))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(top=0.9, bottom=0.21, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

        # Add curves.
        ax.plot.line(df, c.cat_sim_adj, alpha=0.5, color=col_sim_adj)
        ax.plot.line(df, c.cat_sim, alpha=0.5, color=col_sim)
        ax.plot.line(df, c.cat_obs, alpha=0.5, color=col_obs)

        # Customize.
        plt.legend(["Sim. ajustée", "Sim. (réf.)", "Référence"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(varidx.label, fontsize=fs_axes)
        plt.title("")
        plt.suptitle(title, fontsize=fs_sup_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)

        # Close plot.
        plt.close()

        # Save plot.
        fu.save_plot(fig, p_fig)

    # Save CSV file.
    if save_csv:
        fu.save_csv(df, p_csv)


def draw_curves(
    ax: plt.axis,
    varidx: vi.VarIdx,
    da_obs: xr.DataArray,
    da_sim_ref: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    da_sim_adj_ref: xr.DataArray,
    stat: str,
    q: float,
    p_csv: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Draw curves.

    Parameters
    ----------
    ax: plt.axis
        Plot axis.
    varidx: vi.VarIdx
        Variable or index name.
    da_obs: xr.DataArray
        Observed data.
    da_sim_ref: xr.DataArray
        Simulation data for the reference period.
    da_sim: xr.DataArray
        Simulation data.
    da_sim_adj: xr.DataArray
        Adjusted simulation dasta.
    da_sim_adj_ref: xr.DataArray
        Adjusted simulation data for the reference period.
    stat: str
        Statistic: {def_stat.code_max, def_stat.code_quantile, def_stat.code_mean, def_stat.code_sum}
    q: float, optional
        Quantile.
    p_csv: str
        Path of CSV file to create.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = str(varidx.name)

    # Paths.
    if stat in [c.stat_mean, c.stat_min, c.stat_max, c.stat_sum]:
        suffix = stat
    else:
        suffix = "q" + str(round(q * 100)).zfill(2)
    p_csv = p_csv.replace(c.f_ext_csv, "_" + suffix + c.f_ext_csv)

    # Determine if the analysis is required.
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.f_csv in cntx.opt_diagnostic_format)))
    if not save_csv:
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Calculate statistics.
        def da_groupby(
            da: xr.DataArray,
            _stat: str,
            _q: float = -1.0
        ) -> xr.DataArray:

            da_group = da.groupby(da.time.dt.month)
            da_group_stat = None
            if _stat == c.stat_min:
                da_group_stat = da_group.min(dim=c.dim_time)
            elif _stat == c.stat_max:
                da_group_stat = da_group.max(dim=c.dim_time)
            elif _stat == c.stat_mean:
                da_group_stat = da_group.mean(dim=c.dim_time)
            elif _stat == c.stat_quantile:
                da_group_stat = da_group.quantile(_q, dim=c.dim_time)
            elif _stat == c.stat_sum:
                n_years = da[c.dim_time].size / 12
                da_group_stat = da_group.sum(dim=c.dim_time) / n_years
            return da_group_stat

        # Determine if sum is needed.
        stat_actual = stat
        if vi_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            if stat == c.stat_mean:
                stat_actual = c.stat_sum
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_obs         = da_obs.resample(time="1M").sum()
                da_sim_ref     = da_sim_ref.resample(time="1M").sum()
                da_sim         = da_sim.resample(time="1M").sum()
                da_sim_adj     = da_sim_adj.resample(time="1M").sum()
                da_sim_adj_ref = da_sim_adj_ref.resample(time="1M").sum()

        # Calculate statistics.
        da_obs         = da_groupby(da_obs, stat_actual, q)
        da_sim_ref     = da_groupby(da_sim_ref, stat_actual, q)
        da_sim         = da_groupby(da_sim, stat_actual, q)
        da_sim_adj     = da_groupby(da_sim_adj, stat_actual, q)
        da_sim_adj_ref = da_groupby(da_sim_adj_ref, stat_actual, q)

        # Create dataframe.
        dict_pd = {"month": list(range(1, 13)),
                   c.cat_obs: list(da_obs.values),
                   c.cat_sim_ref: list(da_sim_ref.values),
                   c.cat_sim: list(da_sim.values),
                   c.cat_sim_adj: list(da_sim_adj.values),
                   c.cat_sim_adj_ref: list(da_sim_adj_ref.values)}
        df = pd.DataFrame(dict_pd)

    # Draw curves.
    ax.plot.line(df, c.cat_obs, color=col_obs)
    ax.plot.line(df, c.cat_sim_ref, color=col_sim_ref)
    ax.plot.line(df, c.cat_sim, color=col_sim)
    ax.plot.line(df, c.cat_sim_adj, color=col_sim_adj)
    ax.plot.line(df, c.cat_sim_adj_ref, color=col_sim_adj_ref)

    # Save to CSV.
    if save_csv:
        fu.save_csv(df, p_csv)
