# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Quantile mapping functions.
# This requires installing package SciTools (when xr.DataArray.time.dtype=cfg.dtype_obj).
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import config as cfg
import utils

# Ouranos packages.
from xsd.qm import train, predict

# Other packages.
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr


def bias_correction_loop(stn_name, var):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias correction.

    Parameters
    ----------
    stn_name : str
        Station name.
    var : str
        Weather variable.
    -------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Path of directory containing regrid files.
    path_regrid = cfg.get_path_out(stn_name, cfg.cat_regrid, var)

    # List of files in 'path'.
    files = utils.list_files(path_regrid)

    # Loop through simulations sets (3 files per simulation set).
    n_set = int(len(files) / 3)
    if len(cfg.idx_sim) == 0:
        cfg.idx_sim = range(0, n_set)
    for i in cfg.idx_sim:
        list     = files[i * 3].split("/")
        sim_name = list[len(list) - 1].replace(var + "_", "").replace(".nc", "")

        # Best parameter set.
        error_best      = -1

        # Loop through nq values.
        for nq in cfg.nq_calib:

            # Loop through up_qmf values.
            for up_qmf in cfg.up_qmf_calib:

                # Loop through time_int values.
                for time_int in cfg.time_int_calib:

                    print("Correcting i=" + str(i) + ", up_qmf=" + str(up_qmf) + ", time_int=" + str(time_int))

                    # NetCDF file.
                    fn_obs = cfg.get_path_obs(stn_name, var)
                    fn_ref = [i for i in files if "ref" in i][i]
                    fn_fut = fn_ref.replace("ref_", "")
                    fn_qqmap = cfg.get_path_out(stn_name, cfg.cat_qqmap, var)

                    # Figures.
                    fn_fig = fn_fut.split("/")[-1].replace("4qqmap.nc", "calib.png")
                    sup_title = os.path.basename(fn_fig) + "_time_int_" + str(time_int) + "_up_qmf_" + str(up_qmf) + \
                        "_nq_" + str(nq)
                    path_fig = cfg.get_path_out(stn_name, cfg.cat_fig + "/calib", var)
                    if not (os.path.isdir(path_fig)):
                        os.makedirs(path_fig)
                    fn_fig = path_fig + fn_fig

                    # Examine bias correction.
                    bias_correction(var, nq, up_qmf, time_int, fn_obs, fn_ref, fn_fut, fn_qqmap, fn_fig, sup_title)

                    if not cfg.opt_calib_extra:
                        continue

                    # Extra --------------------------------------------------------------------------------------------

                    ds_obs = xr.open_dataset(fn_obs)
                    ds_fut = xr.open_dataset(fn_fut)

                    # TODO: The following block of code does not work. Compatibility with a xarray.DataArray that
                    #       contains a time attribute of type cfg.dtype_obj requires the package SciTools which is
                    #       not available in Python 3.
                    if (ds_fut.time.dtype != cfg.dtype_obj) and (ds_obs.time.dtype != cfg.dtype_obj):

                        # Conversion coefficient.
                        coef = 1
                        if var == cfg.var_pr:
                            coef = cfg.spd

                        fs_sup_title = 8
                        fs_legend = 8
                        fs_axes = 8
                        f = plt.figure(figsize=(15, 3))
                        f.add_subplot(111)
                        plt.subplots_adjust(top=0.9, bottom=0.21, left=0.04, right=0.99, hspace=0.695, wspace=0.416)

                        # Precipitation.
                        (ds_fut[var] * coef).plot(alpha=0.5)
                        # ERROR: (ds_qqmap[var]*coef).plot(alpha=0.5)
                        if var == cfg.var_pr:
                            (ds_obs[var] * coef).plot()
                        # Other variables.
                        else:
                            dt = 0
                            if var in [cfg.var_tas, cfg.var_tasmax, cfg.var_tasmin]:
                                dt = 273.15
                            (ds_obs[var] + dt).plot()
                        plt.legend(["sim", "obs"], fontsize=fs_legend)
                        plt.xlabel("Année", fontsize=fs_axes)
                        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
                        plt.title("")
                        plt.suptitle(sup_title, fontsize=fs_sup_title)
                        plt.tick_params(axis='x', labelsize=fs_axes)
                        plt.tick_params(axis='y', labelsize=fs_axes)

                        if cfg.opt_plt_save:
                            plt.savefig(fn_fig.replace(".png", "_ts.png"))
                        # DEBUG: Need to add a breakpoint below to visualize plot.
                        if cfg.opt_plt_close:
                            plt.close()

                    # TODO: Calculate the error between observations and simulation for the reference period.
                    error_current = -1

                    # Set nq, up_qmf and time_int for the current simulation.
                    if cfg.opt_calib_auto and ((error_best < 0) or (error_current < error_best)):
                        cfg.nq[sim_name][stn_name][var]       = float(nq)
                        cfg.up_qmf[sim_name][stn_name][var]   = up_qmf
                        cfg.time_int[sim_name][stn_name][var] = float(time_int)


def bias_correction(var, nq, up_qmf, time_int, fn_obs, fn_ref, fn_fut, fn_qqmap, fn_fig, sup_title):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias correction.

    Parameters
    ----------
    var : str
        Weather variable.
    nq : float
        Number of quantiles
    up_qmf : float
        Upper limit for quantile mapping function.
    time_int : float
        Windows size (i.e. number of days before + number of days after).
    fn_obs : str
        NetCDF file for observations.
    fn_ref : str
        NetCDF file for reference period.
    fn_fut : str
        NetCDF file for future period.
    fn_qqmap : str
        NetCDF file of qqmap.
    fn_fig : str
        File of figure associated with the plot.
    sup_title : str
        Title of figure.
    -------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Datasets.
    ds_obs = xr.open_dataset(fn_obs)[var].squeeze()
    ds_ref = xr.open_dataset(fn_ref)[var]
    ds_fut = xr.open_dataset(fn_fut)[var]

    # DELETE: Amount of precipitation in winter.
    # DELETE: if var == cfg.var_pr:
    # DELETE:     pos_ref = (ds_ref.time.dt.month >= 11) | (ds_ref.time.dt.month <= 3)
    # DELETE:     pos_fut = (ds_fut.time.dt.month >= 11) | (ds_fut.time.dt.month <= 3)
    # DELETE:     ds_ref[pos_ref] = 1e-8
    # DELETE:     ds_fut[pos_fut] = 1e-8
    # DELETE:     ds_ref.values[ds_ref<1e-7] = 0
    # DELETE:     ds_fut.values[ds_fut<1e-7] = 0

    # Information for post-processing ---------------------------------------------------------------------------------

    # Precipitation.
    if var == cfg.var_pr:
        kind = "*"
        ds_obs.interpolate_na(dim="time")
    # Temperature.
    elif var in [cfg.var_tas, cfg.var_tasmin, cfg.var_tasmax]:
        ds_obs  = ds_obs + 273.15
        ds_obs  = ds_obs.interpolate_na(dim="time")
        kind = "+"
    # Other variables.
    else:
        ds_obs = ds_obs.interpolate_na(dim="time")
        kind = "+"

    # Calculate/read quantiles -----------------------------------------------------------------------------------------

    # Calculate QMF.
    ds_qmf = train(ds_ref.squeeze(), ds_obs.squeeze(), int(nq), cfg.group, kind, time_int, detrend_order=cfg.detrend_order)

    # Calculate QQMAP.
    if cfg.opt_calib_qqmap:
        if var == cfg.var_pr:
            ds_qmf.values[ds_qmf > up_qmf] = up_qmf
        ds_qqmap     = predict(ds_fut.squeeze(), ds_qmf, interp=True, detrend_order=cfg.detrend_order)

    # Read QQMAP.
    else:
        ds_qqmap = xr.open_dataset(fn_qqmap)[var]

    ds_qqmap_per = ds_qqmap.where((ds_qqmap.time.dt.year >= cfg.per_ref[0]) &
                                  (ds_qqmap.time.dt.year <= cfg.per_ref[1]), drop=True)

    # Quantile ---------------------------------------------------------------------------------------------------------

    fs_sup_title = 8
    fs_title     = 6
    fs_legend    = 4
    fs_axes      = 7

    f = plt.figure(figsize=(9, 6))
    f.add_subplot(331)
    plt.subplots_adjust(top=0.9, bottom=0.126, left=0.070, right=0.973, hspace=0.695, wspace=0.416)

    ds_qmf.plot()
    plt.xlabel("Quantile", fontsize=fs_axes)
    plt.ylabel("Jour de l'année", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    # Mean annual values -----------------------------------------------------------------------------------------------

    # Plot.
    f.add_subplot(332)
    if var == cfg.var_pr:
        draw_curves(var, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, "sum")
    else:
        draw_curves(var, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, "mean")
    plt.title(var_desc, fontsize=fs_title)
    plt.legend([cfg.cat_qqmap, "sim", cfg.cat_obs, cfg.cat_qqmap+"_all", "fut"], fontsize=fs_legend)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    # Maximum, Q99, Q75 and mean monthly values ------------------------------------------------------------------------

    for i in range(1, 5):

        plt.subplot(333 + i - 1)

        title    = " (mensuel)"
        quantile = -1
        if i == 1:
            title = "Maximum" + title
        elif i == 2:
            title    = "Q99" + title
            quantile = 0.99
        elif i == 3:
            title    = "Q75" + title
            quantile = 0.75
        else:
            title = "Moyenne" + title

        if i == 1:
            draw_curves(var, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, "max")
        elif (i == 2) or (i == 3):
            draw_curves(var, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, "quantile", quantile)
        else:
            draw_curves(var, ds_qqmap_per, ds_obs, ds_ref, ds_fut, ds_qqmap, "mean")

        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel("Mois", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.legend([cfg.cat_qqmap+"-ref", "sim-ref", cfg.cat_obs, cfg.cat_qqmap+"_all", "sim-all"], fontsize=fs_legend)
        plt.title(title, fontsize=fs_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)

    # Time series ------------------------------------------------------------------------------------------------------

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_pr:
        coef = cfg.spd

    # TODO: The following block of code does not work. Compatibility with a xarray.DataArray that contains a time
    #       attribute of type cfg.dtype_obj requires the package SciTools which is not available for Python 3.
    if (ds_qqmap.time.dtype != cfg.dtype_obj) and (ds_ref.time.dtype != cfg.dtype_obj):
        plt.subplot(313)
        (ds_qqmap * coef).plot(alpha=0.5)
        (ds_ref * coef).plot(alpha=0.5)
        (ds_obs * coef).plot(alpha=0.5)
        if var == cfg.var_pr:
            plt.ylim([0, 300])
        # Other variables.
        else:
            pass
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.legend([cfg.cat_qqmap, "sim", cfg.cat_obs], fontsize=fs_legend)
        plt.title("")
    else:
        pass

    f.suptitle(sup_title, fontsize=fs_sup_title)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)

    del ds_qqmap.attrs['bias_corrected']
    if cfg.opt_plt_save:
        plt.savefig(fn_fig)
    # DEBUG: Need to add a breakpoint below to visualize plot.
    if cfg.opt_plt_close:
        plt.close()


def draw_curves(var, ds, ds_obs, ds_ref, ds_fut, ds_qqmap, stat, quantile=-1.0):

    """
    --------------------------------------------------------------------------------------------------------------------
    Draw curves.

    Parameters
    ----------
    var : str
        Weather variable.
    ds : xarray.dataset
        ...
    ds_obs : xarray.dataset
        ...
    ds_ref : xarray.dataset
        Dataset for the reference period.
    ds_fut : xarray.dataset
        Dataset for the future period.
    ds_qqmap : xarray.dataset
        Dataset for the qqmap.
    stat : {"max", "quantile", "mean", "sum"}
        Statistic.
    quantile : float, optional
        Quantile.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Conversion coefficient.
    coef = 1
    if var == cfg.var_pr:
        coef = cfg.spd

    # Draw curves.
    if stat == "max":
        (ds * coef).groupby(ds.time.dt.month).max().plot()
        (ds_ref * coef).groupby(ds_ref.time.dt.month).max().plot()
        (ds_obs * coef).groupby(ds_obs.time.dt.month).max().plot()
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).max().plot()
        (ds_fut * coef).groupby(ds_fut.time.dt.month).max().plot()
    elif stat == "quantile":
        (ds * coef).groupby(ds.time.dt.month).quantile(quantile).plot()
        (ds_ref * coef).groupby(ds_ref.time.dt.month).quantile(quantile).plot()
        (ds_obs * coef).groupby(ds_obs.time.dt.month).quantile(quantile).plot()
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).quantile(quantile).plot()
        (ds_fut * coef).groupby(ds_fut.time.dt.month).quantile(quantile).plot()
    elif stat == "mean":
        (ds * coef).groupby(ds.time.dt.month).mean().plot()
        (ds_ref * coef).groupby(ds_ref.time.dt.month).mean().plot()
        (ds_obs * coef).groupby(ds_obs.time.dt.month).mean().plot()
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.month).mean().plot()
        (ds_fut * coef).groupby(ds_fut.time.dt.month).mean().plot()
    elif stat == "sum":
        (ds * coef).groupby(ds.time.dt.year).sum().plot()
        (ds_ref * coef).groupby(ds_ref.time.dt.year).sum().plot()
        (ds_obs * coef).groupby(ds_obs.time.dt.year).sum().plot()
        (ds_qqmap * coef).groupby(ds_qqmap.time.dt.year).sum().plot()
        (ds_fut * coef).groupby(ds_fut.time.dt.year).sum().plot()


def physical_coherence(stn_name, var):

    """
    # ------------------------------------------------------------------------------------------------------------------
    Verifies physical coherence.

    Parameters
    ----------
    stn_name: str
        Station name.
    var : [str]
        List of variables.
    --------------------------------------------------------------------------------------------------------------------
    """

    path_qqmap   = cfg.get_path_out(stn_name, cfg.cat_qqmap, var[0])
    files        = utils.list_files(path_qqmap)
    file_tasmin  = files
    file_tasmax  = [i.replace(cfg.var_tasmin, cfg.var_tasmax) for i in files]

    for i in range(len(file_tasmin)):
        i = 10
        print(stn_name + "____________" + file_tasmax[i])
        out_tasmax = xr.open_dataset(file_tasmax[i])
        out_tasmin = xr.open_dataset(file_tasmin[i])

        pos = out_tasmax[var[1]] < out_tasmin[var[0]]

        val_max = out_tasmax[var[1]].values[pos]
        val_min = out_tasmin[var[0]].values[pos]

        out_tasmax[var[1]].values[pos] = val_min
        out_tasmin[var[0]].values[pos] = val_max

        os.remove(file_tasmax[i])
        os.remove(file_tasmin[i])

        out_tasmax.to_netcdf(file_tasmax[i])
        out_tasmin.to_netcdf(file_tasmin[i])


def adjust_date_format(ds):

    """
    --------------------------------------------------------------------------------------------------------------------
    Adjusts date format.

    Parameters
    ----------
    ds : ...
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    dates = ds.cftime_range(start="0001", periods=24, freq="MS", calendar=cfg.cal_noleap)
    da    = xr.DataArray(np.arange(24), coords=[dates], dims=["time"], name="ds")

    return da


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    TODO: Quantify error numerically to facilitate calibration. "bias_correction" could return the error.
    TODO: Determine why an exception is launched at i = 9. It has to do with datetime format.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Module calib launched.")

    # Loop through stations.
    for stn_name in cfg.stn_names:

        # Loop through variables.
        for var in cfg.variables:

            # Bias correction.
            if cfg.opt_calib_bias:

                bias_correction_loop(stn_name, var)

            # Physical coherence.
            if cfg.opt_calib_coherence:
                physical_coherence(stn_name, [cfg.var_tasmin, cfg.var_tasmax])

    print("Module calib completed successfully.")


if __name__ == "__main__":
    run()
