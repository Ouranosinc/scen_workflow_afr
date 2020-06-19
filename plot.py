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
import os.path
import utils
import xarray as xr
import xclim.subset as subset


def plot_year(ds_hour, ds_day, set_name, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Time series.

    Parameters
    ----------
    ds_hour : xarray
        Dataset with hourly frequency.
    ds_day : xarray
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
    ds_day : xarray
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
    if cfg.path_bounds != "":
        ds_day_sel = subset.subset_shape(ds_day_sel, cfg.path_bounds)

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


def plot_ts_indices(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn, idx_name, idx_threshs, rcps, xlim, fn_fig):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate a time series comprising the reference period and all emission scenarios.

    Parameters
    ----------
    ds_ref : xarray[]
        Dataset for the reference period.
    ds_rcp_26 : xarray[]
        Dataset for RCP 2.6.
    ds_rcp_45 : xarray[]
        Dataset for RCP 4.5.
    ds_rcp_85 : xarray[]
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
    fn_fig : str
        File name of figure.
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
    if fn_fig != "":
        dir_fig = os.path.dirname(fn_fig)
        if not (os.path.isdir(dir_fig)):
            os.makedirs(dir_fig)
        plt.savefig(fn_fig)

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

    print("Processing (single): variable = " + var + "; station = " + stn)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Paths and NetCDF files.
    path_regrid = cfg.get_path_sim(stn, cfg.cat_regrid, var)
    files       = utils.list_files(path_regrid)
    fn_obs      = cfg.get_path_obs(stn, var)

    # Plot.
    fs_sup_title = 8
    fs_legend    = 8
    fs_axes      = 8

    f = plt.figure(figsize=(15, 3))
    f.add_subplot(111)
    plt.subplots_adjust(top=0.9, bottom=0.18, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

    # Loop through simulation sets.
    for i in range(int(len(files) / 3)):

        fn_ref   = [i for i in files if "ref" in i][i]
        fn_fut   = fn_ref.replace("ref_", "")
        fn_qqmap = fn_fut.replace("_4qqmap", "").replace(cfg.cat_regrid, cfg.cat_qqmap)
        ds_fut   = xr.open_dataset(fn_fut)
        ds_qqmap = xr.open_dataset(fn_qqmap)
        ds_obs   = xr.open_dataset(fn_obs)
        # ERROR: ds_fut.time.values = ds_fut.time.values.astype(cfg.dtype_64)
        # ERROR: ds_qqmap.time.values = ds_qqmap.time.values.astype(cfg.dtype_64)

        # TODO: Determine if there is a way to convert date format.
        if (ds_fut.time.dtype == cfg.dtype_obj) or (ds_qqmap.time.dtype == cfg.dtype_obj):
            return

        # Curves.
        (ds_obs[var]).plot(alpha=0.5)
        (ds_fut[var]).plot()
        (ds_qqmap[var]).plot(alpha=0.5)

        # Format.
        plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
        sup_title = os.path.basename(fn_fut).replace("4qqmap.nc", "verif_ts_single")
        plt.suptitle(sup_title, fontsize=fs_sup_title)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.title("")
        plt.suptitle(sup_title, fontsize=fs_sup_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)

        if cfg.opt_plt_save:
            fn_fig = sup_title + ".png"
            path_fig = cfg.get_path_sim(stn, cfg.cat_fig + "/verif/ts_single", var)
            if not (os.path.isdir(path_fig)):
                os.makedirs(path_fig)
            fn_fig = path_fig + fn_fig
            plt.savefig(fn_fig)
        # DEBUG: Need to add a breakpoint below to visualize plot.
        if cfg.opt_plt_close:
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

    print("Processing (mosaic): variable = " + var + "; station = " + stn)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_cordex_pr:
        coef = cfg.spd

    # Paths and NetCDF files.
    path_regrid = cfg.get_path_sim(stn, cfg.cat_regrid, var)
    files       = utils.list_files(path_regrid)
    fn_obs      = cfg.get_path_obs(stn, var)

    # Plot.
    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 6
    fs_axes      = 6
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.04, right=0.99, hspace=0.40, wspace=0.30)

    # Loop through simulation sets.
    sup_title = ""
    for i in range(int(len(files) / 3)):

        # NetCDF files.
        fn_fut_i   = [i for i in files if "ref" in i][i].replace("ref_", "")
        fn_qqmap_i = fn_fut_i.replace("_4" + cfg.cat_qqmap, "").replace(cfg.cat_regrid, cfg.cat_qqmap)

        # Open datasets.
        ds_fut   = xr.open_dataset(fn_fut_i)
        ds_qqmap = xr.open_dataset(fn_qqmap_i)
        ds_obs   = xr.open_dataset(fn_obs)
        # ERROR: ds_fut.time.values = ds_fut.time.values.astype(cfg.dtype_64)
        # ERROR: ds_qqmap.time.values = ds_qqmap.time.values.astype(cfg.dtype_64)

        # TODO: Determine if there is a way to convert date format.
        if (ds_fut.time.dtype == cfg.dtype_obj) or (ds_qqmap.time.dtype == cfg.dtype_obj):
            continue

        # Curves.
        plt.subplot(7, 7, i + 1)
        (ds_fut[var] * coef).plot()
        (ds_qqmap[var] * coef).plot(alpha=0.5)
        (ds_obs[var] * coef).plot(alpha=0.5)

        # Format.
        plt.xlabel("", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        title = os.path.basename(files[i]).replace(".nc", "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)
        if i == 0:
            plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)
            sup_title = title + "_verif_ts_mosaic"
            plt.suptitle(sup_title, fontsize=fs_sup_title)

    if cfg.opt_plt_save:
        fn_fig = sup_title + ".png"
        path_fig = cfg.get_path_sim(stn, cfg.cat_fig + "/verif/ts_mosaic", var)
        if not (os.path.isdir(path_fig)):
            os.makedirs(path_fig)
        fn_fig = path_fig + fn_fig
        plt.savefig(fn_fig)
    # DEBUG: Need to add a breakpoint below to visualize plot.
    if cfg.opt_plt_close:
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

    print("Processing (monthly): variable = " + var + "; station = " + stn)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # NetCDF files.
    path_regrid = cfg.get_path_sim(stn, cfg.cat_regrid, var)
    files       = utils.list_files(path_regrid)
    fn_obs      = cfg.get_path_obs(stn, var)
    ds_obs      = xr.open_dataset(fn_obs)
    ds_plt = ds_obs.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().groupby("time.month").\
        mean()[var]

    # Plot.
    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 6
    fs_axes      = 6
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.04, right=0.99, hspace=0.40, wspace=0.30)

    # Loop through simulation sets.
    sup_title = ""
    for i in range(int(len(files) / 3)):

        # Plot.
        plt.subplot(7, 7, i + 1)

        # Curves.
        ds = xr.open_dataset(files[i])[var]
        if isinstance(ds.time[0].values, np.datetime64):
            ds.sel(time=slice("1980-01-01", "2010-12-31")).resample(time="M").mean().groupby("time.month").mean().\
                plot(color="blue")
        ds = xr.open_dataset(files[i])[var]
        if isinstance(ds.time[0].values, np.datetime64):
            ds.sel(time=slice("2050-01-01", "2070-12-31")).resample(time="M").mean().groupby("time.month").mean().\
                plot(color="green")
        ds_plt.plot(color="red")

        # Format.
        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        title = os.path.basename(files[i]).replace(".nc", "")
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)
        if i == 0:
            sup_title = title + "_verif_monthly"
            plt.suptitle(sup_title, fontsize=fs_sup_title)

    # Format.
    plt.legend(["sim", cfg.cat_qqmap, cfg.cat_obs], fontsize=fs_legend)

    if cfg.opt_plt_save:
        fn_fig = sup_title + ".png"
        path_fig = cfg.get_path_sim(stn, cfg.cat_fig + "/verif/monthly", var)
        if not (os.path.isdir(path_fig)):
            os.makedirs(path_fig)
        fn_fig = path_fig + fn_fig
        plt.savefig(fn_fig)
    # DEBUG: Need to add a breakpoint below to visualize plot.
    if cfg.opt_plt_close:
        plt.close()
