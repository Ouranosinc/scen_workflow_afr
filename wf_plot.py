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
import holoviews as hv
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
from typing import List, Optional, Union

# Workflow libraries.
import wf_file_utils as fu
import wf_utils
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
sys.path.append("scen_workflow_afr_dashboard")
from scen_workflow_afr_dashboard import dash_file_utils as dfu
from scen_workflow_afr_dashboard.cl_stat import Stat
from scen_workflow_afr_dashboard.cl_varidx import VarIdx
from scen_workflow_afr_dashboard.dash_plot import get_cmap, get_cmap_name, get_hex_l


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
    da_ref: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_ref: xr.DataArray,
    da_sim_adj: xr.DataArray,
    da_sim_adj_ref: xr.DataArray,
    da_qmf: xr.DataArray,
    varidx: VarIdx,
    sim_code: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot containing a summary of bias adjustment.

    Parameters
    ----------
    da_ref: xr.DataArray
        Reference data.
    da_sim: xr.DataArray
        Simulation data.
    da_sim_ref: xr.DataArray
        Simulation data for the reference period.
    da_sim_adj: xr.DataArray
        Adjusted simulation.
    da_sim_adj_ref: xr.DataArray
        Adjusted simulation for the reference period.
    da_qmf: xr.DataArray
        Quantile mapping function.
    varidx: VarIdx
        Variable or index.
    sim_code: str
        Simulation code.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract information on variable.
    vi_code = str(varidx.code)
    vi_name = str(varidx.name)

    # Paths.
    p_csv = cntx.p_fig(c.VIEW_BIAS, vi_code, vi_name, c.F_CSV, sim_code)
    p_fig = cntx.p_fig(c.VIEW_BIAS, vi_code, vi_name, c.F_PNG, sim_code)

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
    draw_curves(ax, varidx, da_ref, da_sim_ref, da_sim, da_sim_adj, da_sim_adj_ref, Stat(c.STAT_MEAN), p_csv)
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

        draw_curves(ax, varidx, da_ref, da_sim_ref, da_sim, da_sim_adj, da_sim_adj_ref, stat, p_csv)

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
    da_ref.plot.line(alpha=0.5, color=c.COL_OBS)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(y_label, fontsize=fs_axes)
    plt.legend(["Sim. ajustée", "Sim. (réf.)", "Référence"], fontsize=fs_legend, frameon=False)
    plt.title("")

    fig.suptitle(vi_name + "_" + sim_code, fontsize=fs_sup_title)
    plt.tick_params(axis="x", labelsize=fs_axes)
    plt.tick_params(axis="y", labelsize=fs_axes)

    if c.ATTRS_BIAS in da_sim_adj.attrs:
        del da_sim_adj.attrs[c.ATTRS_BIAS]

    # Save plot.
    if c.F_PNG in cntx.opt_diagnostic_format:
        fu.save_plot(fig, p_fig)

    # Close plot.
    plt.close()


def plot_bias_ts(
    da_obs: xr.DataArray,
    da_sim: xr.DataArray,
    da_sim_adj: xr.DataArray,
    varidx: VarIdx,
    sim_code: str
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
    sim_code: str
        Simulation code.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Extract information on variable.
    vi_code = str(varidx.code)
    vi_name = str(varidx.name)

    # Paths.
    p_csv = cntx.p_fig(c.VIEW_BIAS, vi_code, vi_name, c.F_CSV, sim_code=sim_code, suffix="_ts")
    p_fig = cntx.p_fig(c.VIEW_BIAS, vi_code, vi_name, c.F_PNG, sim_code=sim_code, suffix="_ts")

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
        plt.suptitle(sim_code, fontsize=fs_sup_title)
        plt.tick_params(axis="x", labelsize=fs_axes)
        plt.tick_params(axis="y", labelsize=fs_axes)

        # Close plot.
        plt.close()

        # Save plot.
        if c.F_PNG in cntx.opt_diagnostic_format:
            fu.save_plot(fig, p_fig)

    # Save CSV file.
    if save_csv:
        fu.save_csv(df, p_csv)


def draw_curves(
    ax: plt.axis,
    varidx: VarIdx,
    da_ref: xr.DataArray,
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
    da_ref: xr.DataArray
        Reference data.
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
                da_ref         = da_ref.resample(time="1M").sum()
                da_sim_ref     = da_sim_ref.resample(time="1M").sum()
                da_sim         = da_sim.resample(time="1M").sum()
                da_sim_adj     = da_sim_adj.resample(time="1M").sum()
                da_sim_adj_ref = da_sim_adj_ref.resample(time="1M").sum()

        # Calculate statistics.
        da_ref         = da_groupby(da_ref, stat_actual)
        da_sim_ref     = da_groupby(da_sim_ref, stat_actual)
        da_sim         = da_groupby(da_sim, stat_actual)
        da_sim_adj     = da_groupby(da_sim_adj, stat_actual)
        da_sim_adj_ref = da_groupby(da_sim_adj_ref, stat_actual)

        # Create dataframe.
        dict_pd = {"month": list(range(1, 13)),
                   c.CAT_OBS: list(da_ref.values),
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


def gen_map_stations(
    df_stn: pd.DataFrame,
    df_poi: Union[pd.DataFrame, None],
    var_name: str,
    n_years_req: Optional[int] = 1,
    param_per_l: Optional[List[float]] = None,
    per: Optional[List[int]] = None,
    month_l: Optional[List[int]] = None,
    method_id: Optional[str] = "wmo",
    p_bounds: Optional[str] = "",
    p_fig: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generation a map of stations.

    This may require:
        $ conda install -c pyviz geoviews
        $ pip install geoviews

    Parameters
    ----------
    df_stn: pd.DataFrame
        DataFrame containing the weather stations.
    df_poi: Union[pd.DataFrame, None]
        DataFrame containing the points of interest.
    var_name: str
        Climate variable name.
    n_years_req: Optional[int]
        Minimum number of years required.
    param_per_l: Optional[List[float]]
        Parameters associated with the amount of values required in a period (based on missing or available values).
    per: Optional[List[int]]
        Period of interest (first and last year).
    month_l: Optional[List[int]]
        Months to consider.
    method_id: Optional[str]
        Identifier of evaluation method = {1=WMO, 2=Percentage}
        "wmo": Number of missing values [total count, consecutive count] in a month for this month to be considered
               as having not enough values (WMO).
        "pct": Minimum number of values per period (year or month) in a period for this period to be considered as
               having enough values.
    p_bounds: Optional[str]
        Path of a GEOJSON file containing polygons.
    p_fig: Optional[str]
        Path of output figure (with PNG extension).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Initialize parameters.
    if param_per_l is None:
        if method_id == "wmo":
            param_per_l = [11, 5]
        elif method_id == "pct":
            param_per_l = [0.35]

    # Generate figure.
    if (p_fig != "") and\
       (not os.path.exists(p_fig.replace(c.F_EXT_PNG, "")) or not os.path.exists(p_fig + c.F_EXT_PNG)):

        # Clean-up the DataFrame.
        df_stn_clean = pd.DataFrame()

        # Combine station identifier and name.
        df_stn_clean["station"] = df_stn["stn_id"] + "-" + df_stn["stn_name"]

        # Combine coordinates.
        df_stn_clean[c.DIM_LONGITUDE] = df_stn[c.DIM_LONGITUDE]
        df_stn_clean[c.DIM_LONGITUDE].map(lambda x: '{0:.4f}'.format(x))
        df_stn_clean[c.DIM_LATITUDE] = df_stn[c.DIM_LATITUDE]
        df_stn_clean[c.DIM_LATITUDE].map(lambda x: '{0:.4f}'.format(x))
        df_stn_clean["n_years_w_data"] = df_stn["n_years_w_data"]
        df_stn_clean[c.DIM_LONGITUDE] = df_stn[c.DIM_LONGITUDE]
        df_stn_clean[c.DIM_LATITUDE] = df_stn[c.DIM_LATITUDE]

        # List years.
        year_l = []
        column_l = list(df_stn.columns)
        for i in range(len(column_l)):
            try:
                float(column_l[i])
                year_l.append(column_l[i])
            except ValueError:
                pass

        # Collect years with data.
        year_str_l = []
        for i in range(len(df_stn)):
            df_stn_i = pd.DataFrame(df_stn.iloc[i][year_l])
            df_stn_i.reset_index(inplace=True)
            df_stn_i.columns = ["year", "availability"]
            year_i_l = list(df_stn_i[df_stn_i["availability"] > 0]["year"].astype(int).values)
            year_str_l.append(",".join(str(i) for i in year_i_l))
        df_stn_clean["years"] = year_str_l

        # List of months.
        months_str = ("".join(("" if i == 0 else "-") + str(month_l[i]) for i in range(len(month_l))))

        # List of periods.
        per_str = str(per[0]) + "-" + str(per[1]) if (per is not None) and (len(per) >= 2) else ""

        # Title.
        title = var_name + " - At least " + str(n_years_req) + " year(s) with "
        if method_id == "wmo":
            title += "fewer than " + str(param_per_l[0]) + " values/month missing (fewer than " + \
                     str(param_per_l[1]) + " consecutive values/month)"
        else:
            title += "fewer than " + str(param_per_l[0] * 100.0) + "% of values/month missing"
        title += (" during months [" + months_str.replace("-", ",") + "]" if len(month_l) > 0 else "")
        if (per is not None) and (len(per) >= 2):
            title += " (" + per_str + ")"

        # Load the vertices of each one of the polygons comprised in the GEOJSON file.
        # Note that orthoimages are disabled when there is at least one polygon to display. Otherwise, some of the
        # polygons are shown/hidden after zooming in/out.
        df_region_l = []
        if p_bounds != "":
            df_region_l = dfu.load_geojson(p_bounds, "pandas", first_only=False)
            tiles = None
        else:
            tiles = "CartoLight"

        # Determine map limits, considering stations, points of interest and regions.
        x_lim, y_lim = None, None
        lon_l, lat_l = [], []
        if (df_stn is not None) and (len(df_stn) > 0):
            lon_l = lon_l + list(df_stn[c.DIM_LONGITUDE].values)
            lat_l = lat_l + list(df_stn[c.DIM_LATITUDE].values)
        if (df_poi is not None) and (len(df_poi) > 0):
            lon_l = lon_l + list(df_poi[c.DIM_LONGITUDE].values)
            lat_l = lat_l + list(df_poi[c.DIM_LATITUDE].values)
        if p_bounds != "":
            for i in range(len(df_region_l)):
                lon_l = lon_l + list(df_region_l[i][c.DIM_LONGITUDE])
                lat_l = lat_l + list(df_region_l[i][c.DIM_LATITUDE])
        if len(lon_l) > 0:
            x_lim = (min(lon_l), max(lon_l))
            y_lim = (min(lat_l), max(lat_l))

        # Determine the size of visual components.
        # The logic is that a size of 10 is suitable to map the Canada as a whole (80 degrees wide). Point size
        # increases as the width of map (in degrees) decreases.
        size = 10
        if len(df_stn) > 0:
            size = min(max(10, size * 80 / (max(x_lim) - min(x_lim))), 80)

        # Figure.
        xlabel = c.DIM_LONGITUDE.capitalize() + " (°)"
        ylabel = c.DIM_LATITUDE.capitalize() + " (°)"
        hv.extension("bokeh")
        renderer = hv.renderer("bokeh")
        plot = None

        # Draw region boundaries.
        if p_bounds != "":
            for i in range(len(df_region_l)):
                plot_i = df_region_l[i].hvplot.polygons(x=c.DIM_LONGITUDE, y=c.DIM_LATITUDE, geo=True,
                                                        color="lightgrey", line_color="darkgrey", line_width=1,
                                                        alpha=0.7)
                plot = plot_i if plot is None else plot * plot_i

        # Draw locations (weather stations).
        if len(df_stn_clean) > 0:
            plot_i = df_stn_clean.hvplot.points(x=c.DIM_LONGITUDE, y=c.DIM_LATITUDE, geo=True, tiles=tiles,
                                                color="red", alpha=1.0, size=size, title=title,
                                                hover_cols=["station", "years"], xlabel=xlabel, ylabel=ylabel)
            plot = plot_i if plot is None else plot * plot_i
            tiles = None

        # Draw locations (points of interest).
        if (df_poi is not None) and (len(df_poi) > 0):
            plot_i = df_poi.hvplot.points(x=c.DIM_LONGITUDE, y=c.DIM_LATITUDE, geo=True, tiles=tiles,
                                          color="green", alpha=1.0, size=size,
                                          title=title, hover_cols=["name", "region"], xlabel=xlabel, ylabel=ylabel)
            plot = plot_i if plot is None else plot * plot_i

        # Plot options (setting x_lim and y_lim is not working).
        plot = plot.options(width=1920, height=1080)
        # if len(df_region_l) == 0:
        #     plot = plot.options(xlim=x_lim, ylim=y_lim)

        # Save figure to HTML.
        renderer.save(plot, p_fig.replace(c.F_EXT_PNG, ""))

        # Save figure to PNG.
        renderer.save(plot, p_fig.replace(c.F_EXT_PNG, ""), fmt=c.F_PNG)
