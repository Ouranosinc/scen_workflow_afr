# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions that allow to verify generated NetCDF files.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import config as cfg
import utils

# Other packages.
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr


def plot_ts_single(stn_name, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (one simulation: one plot).

    Parameters:
    ----------
    stn_name: str
        Station name.
    var: str
        Weather variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Processing (single): variable = " + var + "; station = " + stn_name)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Paths and NetCDF files.
    path_regrid = cfg.get_path_sim(stn_name, cfg.cat_regrid, var)
    files       = utils.list_files(path_regrid)
    fn_obs      = cfg.get_path_obs(stn_name, var)

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
            path_fig = cfg.get_path_sim(stn_name, cfg.cat_fig + "/verif/ts_single", var)
            if not (os.path.isdir(path_fig)):
                os.makedirs(path_fig)
            fn_fig = path_fig + fn_fig
            plt.savefig(fn_fig)
        # DEBUG: Need to add a breakpoint below to visualize plot.
        if cfg.opt_plt_close:
            plt.close()


def plot_ts_mosaic(stn_name, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (all in one plot).

    Parameters:
    ----------
    stn_name: str
        Station name.
    var: str
        Weather variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Processing (mosaic): variable = " + var + "; station = " + stn_name)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_pr:
        coef = cfg.spd

    # Paths and NetCDF files.
    path_regrid = cfg.get_path_sim(stn_name, cfg.cat_regrid, var)
    files       = utils.list_files(path_regrid)
    fn_obs      = cfg.get_path_obs(stn_name, var)

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
        path_fig = cfg.get_path_sim(stn_name, cfg.cat_fig + "/verif/ts_mosaic", var)
        if not (os.path.isdir(path_fig)):
            os.makedirs(path_fig)
        fn_fig = path_fig + fn_fig
        plt.savefig(fn_fig)
    # DEBUG: Need to add a breakpoint below to visualize plot.
    if cfg.opt_plt_close:
        plt.close()


def plot_monthly(stn_name, var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Verify all simulations (all in one plot).

    Parameters:
    ----------
    stn_name: str
        Station name.
    var: str
        Weather variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Processing (monthly): variable = " + var + "; station = " + stn_name)

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # NetCDF files.
    path_regrid = cfg.get_path_sim(stn_name, cfg.cat_regrid, var)
    files       = utils.list_files(path_regrid)
    fn_obs      = cfg.get_path_obs(stn_name, var)
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
        path_fig = cfg.get_path_sim(stn_name, cfg.cat_fig + "/verif/monthly", var)
        if not (os.path.isdir(path_fig)):
            os.makedirs(path_fig)
        fn_fig = path_fig + fn_fig
        plt.savefig(fn_fig)
    # DEBUG: Need to add a breakpoint below to visualize plot.
    if cfg.opt_plt_close:
        plt.close()


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Module verif launched.")

    # Loop through station names.
    for stn_name in cfg.stn_names:

        # Loop through variables.
        for var in cfg.variables:

            # Produce plots.
            try:
                plot_ts_single(stn_name, var)
                plot_ts_mosaic(stn_name, var)
                plot_monthly(stn_name, var)
            except FileExistsError:
                print("Unable to locate the required files!")
                pass

    print("Module verif completed successfully.")


if __name__ == "__main__":
    run()
