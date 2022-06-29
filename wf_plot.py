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
import wf_file_utils as fu
import wf_utils
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard.cl_stat import Stat
from dashboard.cl_varidx import VarIdx
from dashboard.dash_plot import get_cmap, get_cmap_name, get_hex_l


def plot_year(
    ds_hour: xr.Dataset,
    ds_day: xr.Dataset,
    var: VarIdx
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
    var: VarIdx
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
    var: VarIdx,
    date: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Map.

    Parameters
    ----------
    ds_day: xr.Dataset
        Dataset with daily frequency.
    var: VarIdx
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
                               cbar_kwargs=dict(orientation="vertical", pad=0.05, shrink=1, label=var.label))
    plt.title(date)
    plt.suptitle("", fontsize=fs)
    plt.xlabel("Longitude (º)", fontsize=fs)
    plt.ylabel("Latitude (º)", fontsize=fs)
    plt.tick_params(axis="x", labelsize=fs)
    plt.tick_params(axis="y", labelsize=fs)

    plt.close()


def plot_postprocess(
    df: pd.DataFrame,
    var: VarIdx,
    title: str
) -> plt.Figure:

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot allowing to compare adjusted and non-adjusted curves to reference dataset.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    var: VarIdx
        Variable.
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
    if c.CAT_SIM_ADJ in df.columns:
        legend_items.insert(0, "Sim. ajustée")

    # Draw curves.
    if c.CAT_SIM_ADJ in df.columns:
        ax.plot(df["year"], df[c.CAT_SIM_ADJ], color=c.COL_SIM_ADJ)
    ax.plot(df["year"], df[c.CAT_SIM], color=c.COL_SIM)
    ax.plot(df["year"], df[c.CAT_OBS], color=c.COL_OBS)

    # Font size.
    fs        = 8
    fs_title  = fs
    fs_legend = fs
    fs_axes   = fs

    # Customize.
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlabel("Année", fontsize=fs_axes)
    y_label = var.label
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
    varidx: VarIdx,
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
    varidx: VarIdx
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
    if varidx.is_summable:
        coef = c.SPD
    if (vi_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]) and (len(units) > 0):
        if units[0] == c.UNIT_K:
            delta_ref = -c.d_KC
        if units[1] == c.UNIT_K:
            delta_sim = -c.d_KC

    # Actual data.
    x = [*range(len(df["year"]))]
    y = (df[c.CAT_OBS] * coef + delta_ref).values

    # Predictions.
    y_wo_nan = [i for i in y if not np.isnan(i)]
    x_wo_nan = [*range(len(y_wo_nan))]
    coefs = poly.polyfit(x_wo_nan, y_wo_nan, 4)
    y_hat = list(poly.polyval(x_wo_nan, coefs)) + [np.nan] * (len(x) - len(x_wo_nan))

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
        ax.plot(x, y, color=c.COL_SIM_REF)
        ax.plot(x, y_hat, color="black")
        plt.legend(["Simulation (réf.)", "Tendance"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Jour", fontsize=fs_axes)
        plt.ylabel(varidx.label, fontsize=fs_axes)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        plt.title("Tendance", fontsize=fs_title)

        # Lower plot: Complete simulation.
        fig.add_subplot(212)
        arr_y_detrend = signal.detrend(df[c.CAT_SIM][np.isnan(df[c.CAT_SIM]).astype(int) == 0] * coef + delta_sim)
        arr_x_detrend = cntx.per_ref[0] + np.arange(0, len(arr_y_detrend), 1) / 365
        arr_y_error  = (y - y_hat)
        arr_x_error = cntx.per_ref[0] + np.arange(0, len(arr_y_error), 1) / 365
        plt.plot(arr_x_detrend, arr_y_detrend, alpha=0.5, color=c.COL_SIM)
        plt.plot(arr_x_error, arr_y_error, alpha=0.5, color=c.COL_SIM_REF)
        plt.legend(["Simulation", "Simulation (réf.)"], fontsize=fs_legend, frameon=False)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(varidx.label, fontsize=fs_axes)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)
        plt.title("Variation autour de la moyenne (prédiction basée sur une équation quartique)", fontsize=fs_title)

    # Close plot.
    plt.close()

    return fig


def plot_bias(
    da_obs: xr.DataArray,
    da_sim_ref: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    da_sim_adj_ref: xr.DataArray,
    da_qmf: xr.DataArray,
    varidx: VarIdx,
    sup_title: str,
    p_fig: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot containing a summary of bias adjustment.

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
    varidx: VarIdx
        Variable or index.
    sup_title: str
        Title of figure.
    p_fig: str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = str(varidx.name)

    # Paths.
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.F_CSV + cntx.sep).\
        replace(c.F_EXT_PNG, c.F_EXT_CSV)

    # Quantile ---------------------------------------------------------------------------------------------------------

    # Font size.
    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 5
    fs_axes      = 7

    y_label = varidx.label

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(431)
    plt.subplots_adjust(top=0.930, bottom=0.065, left=0.070, right=0.973, hspace=0.90, wspace=0.250)

    # Build color map (custom or matplotlib).
    z_min, z_max = float(da_qmf.values.min()), float(da_qmf.values.max())
    z_min = -abs(max(z_min, z_max))
    z_max = -z_min
    n_cluster = 10
    cmap_name = str(get_cmap_name(z_min, z_max))
    hex_l = get_hex_l(cmap_name)
    if hex_l is not None:
        cmap = get_cmap(cmap_name, hex_l, n_cluster)
    else:
        cmap = plt.cm.get_cmap(cmap_name, n_cluster)

    # Quantile mapping function.
    img1 = ax.imshow(da_qmf, extent=[0, 1, 365, 1], cmap=cmap, vmin=z_min, vmax=z_max)
    cb = fig.colorbar(img1, ax=ax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cb.ax.set_yticklabels(cb.ax.get_yticks(), fontsize=fs_axes)
    cb.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))
    plt.title("Fonction de Quantile Mapping", fontsize=fs_title)
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
    ax = fig.add_subplot(432)
    draw_curves(ax, varidx, da_obs, da_sim_ref, da_sim, da_sim_adj, da_sim_adj_ref, Stat(c.STAT_MEAN), p_csv)
    plt.title("Moyenne", fontsize=fs_title)
    plt.legend(legend_items, fontsize=fs_legend, frameon=False)
    plt.xlim([1, 12])
    plt.xticks(np.arange(1, 13, 1))
    plt.xlabel("Mois", fontsize=fs_axes)
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    # Mean, c100, c099, c075, c050, c025, c001 and c000 monthly values -------------------------------------------------

    for i in range(1, len(cntx.opt_bias_centiles) + 1):

        ax = plt.subplot(433 + i - 1)

        stat = Stat(c.STAT_CENTILE, cntx.opt_tbl_centiles[i - 1])
        title = stat.desc

        if stat.centile == 0:
            stat = Stat(c.STAT_MIN)
        elif stat.centile == 1:
            stat = Stat(c.STAT_MAX)

        draw_curves(ax, varidx, da_obs, da_sim_ref, da_sim, da_sim_adj, da_sim_adj_ref, stat, p_csv)

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
    if da_sim_adj.time.time.dtype == c.DTYPE_OBJ:
        da_sim_adj[c.DIM_TIME] = wf_utils.reset_calendar(da_sim_adj.time)
    if da_sim_ref.time.time.dtype == c.DTYPE_OBJ:
        da_sim_ref[c.DIM_TIME] = wf_utils.reset_calendar(da_sim_ref.time)

    plt.subplot(313)
    da_sim_adj.plot.line(alpha=0.5, color=c.COL_SIM_ADJ)
    da_sim_ref.plot.line(alpha=0.5, color=c.COL_SIM_REF)
    da_obs.plot.line(alpha=0.5, color=c.COL_OBS)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.legend(["Sim. ajustée", "Sim. (réf.)", "Référence"], fontsize=fs_legend, frameon=False)
    plt.title("")

    fig.suptitle(vi_name + "_" + sup_title, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    if c.ATTRS_BIAS in da_sim_adj.attrs:
        del da_sim_adj.attrs[c.ATTRS_BIAS]

    # Save plot.
    if p_fig != "":
        fu.save_plot(fig, p_fig)

    # Close plot.
    plt.close()


def plot_bias_ts(
    da_obs: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    varidx: VarIdx,
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
    varidx: VarIdx
        Variable or index.
    title: str
        Title of figure.
    p_fig: str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    vi_name = str(varidx.name)

    # Paths.
    p_csv = p_fig.replace(cntx.sep + vi_name + cntx.sep, cntx.sep + vi_name + "_" + c.F_CSV + cntx.sep). \
        replace(c.F_EXT_PNG, c.F_EXT_CSV)

    # Determine if the analysis is required.
    save_fig = (cntx.opt_force_overwrite or ((not os.path.exists(p_fig)) and (c.F_PNG in cntx.opt_diagnostic_format)))
    save_csv = (cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_diagnostic_format)))
    if not (save_fig or save_csv):
        return

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Convert date format if the need is.
        if da_obs.time.dtype == c.DTYPE_OBJ:
            da_obs[c.DIM_TIME] = wf_utils.reset_calendar(da_obs)
        if da_sim.time.dtype == c.DTYPE_OBJ:
            da_sim[c.DIM_TIME] = wf_utils.reset_calendar(da_sim)
        if da_sim_adj.time.dtype == c.DTYPE_OBJ:
            da_sim_adj[c.DIM_TIME] = wf_utils.reset_calendar(da_sim_adj)

        # Create dataframe.
        n_obs = len(list(da_obs.values))
        n_sim_adj = len(list(da_sim_adj.values))
        dict_pd = {"day": list(range(1, n_sim_adj + 1)),
                   c.CAT_OBS: list(da_obs.values) + [np.nan] * (n_sim_adj - n_obs),
                   c.CAT_SIM_ADJ: list(da_sim_adj.values),
                   c.CAT_SIM: list(da_sim.values)}
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
        x_column = "day" if "day" in df.columns else "month"
        ax.plot(df[x_column], df[c.CAT_SIM_ADJ], alpha=0.5, color=c.COL_SIM_ADJ)
        ax.plot(df[x_column], df[c.CAT_SIM], alpha=0.5, color=c.COL_SIM)
        ax.plot(df[x_column], df[c.CAT_OBS], alpha=0.5, color=c.COL_OBS)

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
    varidx: VarIdx,
    da_obs: xr.DataArray,
    da_sim_ref: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    da_sim_adj_ref: xr.DataArray,
    stat: Stat,
    p_csv: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Draw curves.

    Parameters
    ----------
    ax: plt.axis
        Plot axis.
    varidx: VarIdx
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
    stat: Stat
        Statistic.
    p_csv: str
        Path of CSV file to create.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Paths.
    if stat.code in [c.STAT_MEAN, c.STAT_MIN, c.STAT_MAX, c.STAT_SUM]:
        suffix = stat.code
    else:
        suffix = stat.centile_as_str
    p_csv = p_csv.replace(c.F_EXT_CSV, "_" + suffix + c.F_EXT_CSV)

    # Determine if the analysis is required.
    save_csv = cntx.opt_force_overwrite or ((not os.path.exists(p_csv)) and (c.F_CSV in cntx.opt_diagnostic_format))

    # Load existing CSV file.
    if (not cntx.opt_force_overwrite) and os.path.exists(p_csv):
        df = pd.read_csv(p_csv)

    # Prepare data.
    else:

        # Calculate statistics.
        def da_groupby(
            da: xr.DataArray,
            _stat: Stat
        ) -> xr.DataArray:

            da_group = da.groupby(da.time.dt.month)
            da_group_stat = None
            if _stat.code == c.STAT_MIN:
                da_group_stat = da_group.min(dim=c.DIM_TIME)
            elif _stat.code == c.STAT_MAX:
                da_group_stat = da_group.max(dim=c.DIM_TIME)
            elif _stat.code == c.STAT_MEAN:
                da_group_stat = da_group.mean(dim=c.DIM_TIME)
            elif _stat.code == c.STAT_CENTILE:
                da_group_stat = da_group.quantile(float(_stat.centile) / 100.0, dim=c.DIM_TIME)
            elif _stat.code == c.STAT_SUM:
                n_years = da[c.DIM_TIME].size / 12
                da_group_stat = da_group.sum(dim=c.DIM_TIME) / n_years
            return da_group_stat

        # Determine if sum is needed.
        stat_actual = stat
        if varidx.is_summable:
            if stat.code == c.STAT_MEAN:
                stat_actual.code = c.STAT_SUM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                da_obs         = da_obs.resample(time="1M").sum()
                da_sim_ref     = da_sim_ref.resample(time="1M").sum()
                da_sim         = da_sim.resample(time="1M").sum()
                da_sim_adj     = da_sim_adj.resample(time="1M").sum()
                da_sim_adj_ref = da_sim_adj_ref.resample(time="1M").sum()

        # Calculate statistics.
        da_obs         = da_groupby(da_obs, stat_actual)
        da_sim_ref     = da_groupby(da_sim_ref, stat_actual)
        da_sim         = da_groupby(da_sim, stat_actual)
        da_sim_adj     = da_groupby(da_sim_adj, stat_actual)
        da_sim_adj_ref = da_groupby(da_sim_adj_ref, stat_actual)

        # Create dataframe.
        dict_pd = {"month": list(range(1, 13)),
                   c.CAT_OBS: list(da_obs.values),
                   c.CAT_SIM_REF: list(da_sim_ref.values),
                   c.CAT_SIM: list(da_sim.values),
                   c.CAT_SIM_ADJ: list(da_sim_adj.values),
                   c.CAT_SIM_ADJ_REF: list(da_sim_adj_ref.values)}
        df = pd.DataFrame(dict_pd)

    # Draw curves.
    ax.plot(df["month"], df[c.CAT_OBS], color=c.COL_OBS)
    ax.plot(df["month"], df[c.CAT_SIM_REF], color=c.COL_SIM_REF)
    ax.plot(df["month"], df[c.CAT_SIM], color=c.COL_SIM)
    ax.plot(df["month"], df[c.CAT_SIM_ADJ], color=c.COL_SIM_ADJ)
    ax.plot(df["month"], df[c.CAT_SIM_ADJ_REF], color=c.COL_SIM_ADJ_REF)

    # Save to CSV.
    if save_csv and (p_csv != ""):
        fu.save_csv(df, p_csv)
