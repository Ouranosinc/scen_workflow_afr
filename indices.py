# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions related to climate indices.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import math
import numpy as np
import os.path
import pandas as pd
import plot
import statistics
import utils
import warnings
import xarray as xr
import xclim.indices as indices
import xclim.subset as subset
from scipy.interpolate import griddata


def calc_idx_ts(idx_name, idx_threshs):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a time series.

    Parameters:
    idx_names : str
        Index name.
    idx_threshs : [float]
        Threshold values associated with each indicator.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Reference period.
    years_ref   = [cfg.per_ref[0], cfg.per_ref[1]]

    # Future period.
    years_fut   = [cfg.per_ref[1], cfg.per_fut[1]]

    # Emission scenarios.
    rcps = [cfg.rcp_ref] + cfg.rcps

    # ==========================================================
    # TODO.CUSTOMIZATION.BEGIN
    # When adding a new climate index, specify the required
    # variable(s) by copying the following code block.
    # ==========================================================

    # Select variables.
    vars = []
    if idx_name == cfg.idx_tx_days_above:
        vars = [cfg.var_cordex_tasmax]

    # ==========================================================
    # TODO.CUSTOMIZATION.END
    # ==========================================================

    # Loop through stations.
    for stn in cfg.stns:

        # Verify if this variable is available for the current station.
        vars_avail = True
        for var in vars:
            if not(os.path.isdir(cfg.get_d_sim(stn, cfg.cat_qqmap, var))):
                vars_avail = False
                break
        if not(vars_avail):
            continue

        # Loop through emissions scenarios.
        ds_ref = None
        ds_rcp_26 = None
        ds_rcp_45 = None
        ds_rcp_85 = None
        for rcp in rcps:

            utils.log("-", True)
            utils.log("Index             : " + idx_name, True)
            utils.log("Station           : " + stn, True)
            utils.log("Emission scenario : " + cfg.get_rcp_desc(rcp), True)
            utils.log("-", True)

            # Analysis of simulation files -----------------------------------------------------------------------------

            utils.log("Collecting simulation files.", True)

            # List simulation files. As soon as there is no file for one variable, the analysis for the current RCP
            # needs to abort.
            n_sim = 0
            p_sim = []
            for var in vars:
                if rcp == cfg.rcp_ref:
                    p_sim_i = cfg.get_d_sim(stn, cfg.cat_obs, "") + var + "/" + var + "_" + stn + ".nc"
                    if os.path.exists(p_sim_i) and (type(p_sim_i) is str):
                        p_sim_i = [p_sim_i]
                else:
                    p_format = cfg.get_d_sim(stn, cfg.cat_qqmap, "") + var + "/*_" + rcp + ".nc"
                    p_sim_i = glob.glob(p_format)
                if not(p_sim_i):
                    p_sim = []
                else:
                    p_sim.append(p_sim_i)
                    n_sim = len(p_sim_i)
            if not p_sim:
                continue

            utils.log("Calculating climate indices", True)

            # Loop through simulations.
            for i_sim in range(0, n_sim):

                # Scenarios --------------------------------------------------------------------------------------------

                # Load datasets (one per variable).
                ds_scen = []

                for var in range(0, len(vars)):
                    ds = xr.open_dataset(p_sim[var][i_sim])
                    ds_scen.append(ds)

                # Adjust units.
                idx_threshs_str = []
                for i in range(0, len(ds_scen)):
                    if rcp == cfg.rcp_ref:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh) + " C")
                    else:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh + 273.15) + " K")

                # Indices ----------------------------------------------------------------------------------------------

                # TODO: Below, unit conversion should not be required. The unit in the file produced by the scenario
                #       workflow is "degree_C", but it should be "C". This can be fixed eventually, but it's not a
                #       priority.

                idx_units = None

                # ==========================================================
                # TODO.CUSTOMIZATION.BEGIN
                # When adding a new climate index, calculate the index by
                # copying the following code block.
                # ==========================================================

                # Number of days where daily maximum temperature exceeds a threshold value.
                if idx_name == cfg.idx_tx_days_above:
                    ds_scen_tasmax = ds_scen[0][cfg.var_cordex_tasmax]
                    if rcp == cfg.rcp_ref:
                        ds_scen_tasmax["units"] = "C"
                        ds_scen_tasmax.attrs["units"] = "C"
                    idx_thresh_str_tasmax = idx_threshs_str[0]
                    arr_idx = indices.tx_days_above(ds_scen_tasmax, idx_thresh_str_tasmax).values
                    idx_units = 1

                # ==========================================================
                # TODO.CUSTOMIZATION.END
                # ==========================================================

                # Create dataset.
                da_idx = xr.DataArray(arr_idx)
                da_idx.name = idx_name
                ds_idx = da_idx.to_dataset()
                ds_idx["units"] = idx_units
                ds_idx = ds_idx.rename_dims({"dim_0": "time"})
                if rcp == cfg.rcp_ref:
                    ds_idx = ds_idx.rename_dims({"dim_1": "lon", "dim_2": "lat"})
                else:
                    ds_idx = ds_idx.expand_dims(lon=1)
                    ds_idx = ds_idx.expand_dims(lat=1)
                ds_idx.attrs["standard_name"] = idx_name
                ds_idx.attrs["long_name"] = idx_name
                ds_idx.assign_attrs({"lon": ds_scen[0]["lon"]})
                ds_idx.assign_attrs({"lat": ds_scen[0]["lat"]})

                # Adjust calendar.
                year_1 = cfg.per_fut[0]
                year_n = cfg.per_fut[1]
                if rcp == cfg.rcp_ref:
                    year_1 = max(cfg.per_ref[0], int(str(ds_scen[0]["time"][0].values)[0:4]))
                    year_n = min(cfg.per_ref[1], int(str(ds_scen[0]["time"][len(ds_scen[0]["time"]) - 1].values)[0:4]))
                ds_idx["time"] =  utils.reset_calendar(ds_idx, year_1, year_n, cfg.freq_YS)

                # Save result to NetCDF file.
                if rcp == cfg.rcp_ref:
                    p_idx = cfg.get_p_obs(stn, idx_name)
                else:
                    p_idx = cfg.get_d_sim(stn, cfg.cat_idx, idx_name) +\
                            os.path.basename(p_sim[0][i_sim]).replace(vars[0], idx_name)
                utils.save_dataset(ds_idx, p_idx)

            # Calculate statistic.
            def calc_stat(cat, stn, idx_name, rcp, years, stat):

                # Calculate statistic.
                ds_idx = statistics.calc_stat(cat, cfg.freq_YS, stn, idx_name, rcp, cfg.per_fut, stat)
                if var == cfg.var_cordex_pr:
                    ds_idx = ds_idx * 365

                # Select years.
                years_str   = [str(years[0]) + "-01-01", str(years[1]) + "-12-31"]
                ds_idx = ds_idx.sel(time=slice(years_str[0], years_str[1])).sel(lon=0, lat=0)

                return ds_idx

            # Calculate statistics (future period).
            if rcp != cfg.rcp_ref:

                # Calculate statistics.
                year_1 = max(cfg.per_ref[1], int(str(ds_scen[0]["time"][0].values)[0:4]))
                year_n = min(cfg.per_fut[1], int(str(ds_scen[0]["time"][len(ds_scen[0]["time"]) - 1].values)[0:4]))
                ds_idx_mean = calc_stat(cfg.cat_sim, stn, idx_name, rcp, [year_1, year_n], cfg.stat_mean)
                ds_idx_min  = calc_stat(cfg.cat_sim, stn, idx_name, rcp, [year_1, year_n], cfg.stat_min)
                ds_idx_max  = calc_stat(cfg.cat_sim, stn, idx_name, rcp, [year_1, year_n], cfg.stat_max)

                # Combine datasets.
                ds_fut  = [ds_idx_mean, ds_idx_min, ds_idx_max]
                if rcp == cfg.rcp_26:
                    ds_rcp_26 = ds_fut
                elif rcp == cfg.rcp_45:
                    ds_rcp_45 = ds_fut
                elif rcp == cfg.rcp_85:
                    ds_rcp_85 = ds_fut

            # Calculate statistics (reference period).
            # The minimum and maximum values are calculated using simulation data and future period. This is
            # different than the mean value that comes from observations and reference period.
            if rcp == rcps[len(rcps) - 1]:

                # Calculate statistics.
                year_1      = max(cfg.per_ref[0], int(str(ds_scen[0]["time"][0].values)[0:4]))
                year_n      = min(cfg.per_ref[1], int(str(ds_scen[0]["time"][len(ds_scen[0]["time"]) - 1].values)[0:4]))
                ds_idx_mean = calc_stat(cfg.cat_obs, stn, idx_name, "ref", [year_1, year_n], cfg.stat_mean)
                ds_idx_min  = calc_stat(cfg.cat_sim, stn, idx_name, "*", [year_1, year_n], cfg.stat_min)
                ds_idx_max  = calc_stat(cfg.cat_sim, stn, idx_name, "*", [year_1, year_n], cfg.stat_max)

                # Combine datasets.
                ds_ref  = [ds_idx_mean, ds_idx_min, ds_idx_max]

            # Generate plot.
            if (rcp == rcps[len(rcps) - 1]) and cfg.opt_plot:
                utils.log("Generating time series of indices.", True)
                p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cfg.cat_idx, "") + idx_name + "_" + stn + ".png"
                plot.plot_idx_ts(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(),
                                 idx_name, idx_threshs, rcps, [years_ref[0], years_fut[1]], p_fig)


def calc_idx_heatmap(idx_name, idx_threshs, rcp, per_hors, stat=cfg.stat_mean):

    """
    --------------------------------------------------------------------------------------------------------------------
    Interpolate data within a zone.

    Parameters
    ----------
    idx_name: str
        Climate index.
    idx_threshs: [float]
        Climate index thresholds.
    rcp: str
        Emission scenario.
    per_hors: [[int]]
        Horizons.
    stat: str
        Statistics {cfg.stat_mean, cfg.stat_min, cfg.stat_max}.
    --------------------------------------------------------------------------------------------------------------------
    """

    utils.log("-", True)
    utils.log("Index             : " + idx_name, True)
    utils.log("Emission scenario : " + cfg.get_rcp_desc(rcp), True)
    utils.log("-", True)

    # Number of years and stations.
    if rcp == cfg.rcp_ref:
        n_year = cfg.per_ref[1] - cfg.per_ref[0] + 1
    else:
        n_year = cfg.per_fut[1] - cfg.per_ref[1] + 1
    n_stn = len(cfg.stns)

    # ==========================================================
    # TODO.CUSTOMIZATION.BEGIN
    # When adding a new climate index, calculate the index by
    # copying the following code block.
    # ==========================================================

    # Determine variable.
    var = ""
    if idx_name == cfg.idx_tx_days_above:
        var = cfg.var_cordex_tasmax

    # ==========================================================
    # TODO.CUSTOMIZATION.END
    # ==========================================================

    # Get information on stations.
    # TODO.YR: Coordinates should be embedded into ds_stat below.
    p_stn = glob.glob(cfg.get_d_stn(cfg.var_cordex_tas) + "../*.csv")[0]
    df = pd.read_csv(p_stn, sep=cfg.file_sep)

    # Collect values for each station and determine overall boundaries.
    utils.log("Collecting emissions scenarios at each station.", True)
    x_bnds = []
    y_bnds = []
    data_stn = []
    for stn in cfg.stns:

        # Get coordinates.
        lon = df[df["station"] == stn]["lon"].values[0]
        lat = df[df["station"] == stn]["lat"].values[0]

        # Calculate statistics.
        if rcp == cfg.rcp_ref:
            ds_stat = statistics.calc_stat(cfg.cat_obs, cfg.freq_YS, stn, idx_name, rcp, None, cfg.stat_mean)
        else:
            ds_stat = statistics.calc_stat(cfg.cat_sim, cfg.freq_YS, stn, idx_name, rcp, None, cfg.stat_mean)
        if ds_stat is None:
            continue

        # Extract data from stations.
        data = [[], [], []]
        n = ds_stat.dims["time"]
        for year in range(0, n):

            # Collect data.
            x = float(lon)
            y = float(lat)
            z = float(ds_stat[idx_name][0][0][year])
            if math.isnan(z):
                z = float(0)
            data[0].append(x)
            data[1].append(y)
            data[2].append(z)

            # Update overall boundaries (round according to the variable 'step').
            if x_bnds == []:
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
    # There is a certain flexibility regarding the number of years in a dataset. Ideally, the station should not have
    # been considered in the analysis, unless there is no better option.
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
        new_grid_data[i_year, :, :] = griddata((arr_x, arr_y), arr_z, (new_grid[0], new_grid[1]), fill_value=np.nan,
                                               method="linear")
    da_idx = xr.DataArray(new_grid_data,
                          coords={"time": grid_time, "lat": grid_y, "lon": grid_x}, dims=["time", "lat", "lon"])
    ds_idx = da_idx.to_dataset(name=idx_name)

    # Clip to country boundaries.
    # TODO: Clipping is no longer working when launching the script from a terminal.
    if cfg.d_bounds != "":
        try:
            ds_idx = subset.subset_shape(ds_idx, cfg.d_bounds)
        except TypeError:
            utils.log("Unable to use a mask.", True)

    # Loop through horizons.
    if cfg.opt_plot:
        utils.log("Generating maps.", True)
        for per_hor in per_hors:

            # Select years.
            if rcp == cfg.rcp_ref:
                year_1 = 0
                year_n = cfg.per_ref[1] - cfg.per_ref[0]
            else:
                year_1 = per_hor[0] - cfg.per_ref[1]
                year_n = per_hor[1] - cfg.per_ref[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ds_hor = ds_idx[cfg.idx_tx_days_above][year_1:(year_n+1)][:][:].mean("time", skipna=True)

            # Plot.
            p_fig = cfg.get_d_sim("", cfg.cat_fig + "/" + cfg.cat_idx, "") +\
                    idx_name + "_" + rcp + "_" + str(per_hor[0]) + "_" + str(per_hor[1]) + ".png"
            plot.plot_idx_heatmap(ds_hor, idx_name, idx_threshs, grid_x, grid_y, per_hor, p_fig, "matplotlib")


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate indices.
    utils.log("=")
    utils.log("Step #6a  Calculation of indices.")
    for i in range(0, len(cfg.idx_names)):

        # Parameters.
        idx_name = cfg.idx_names[i]
        idx_threshs = cfg.idx_threshs[i]

        # Calculate index.
        # calc_idx_ts(idx_name, idx_threshs)

    # Map indices.
    # Interpolation requires multiples stations.
    utils.log("=")
    utils.log("Step #6b  Generation of index maps.")
    if len(cfg.stns) > 1:
        for i in range(0, len(cfg.idx_names)):

            # Parameters.
            idx_name = cfg.idx_names[i]
            idx_threshs = cfg.idx_threshs[i]

            # Reference period.
            calc_idx_heatmap(idx_name, idx_threshs, cfg.rcp_ref, [cfg.per_ref])

            # Future period.
            for rcp in cfg.rcps:
                calc_idx_heatmap(idx_name, idx_threshs, rcp, cfg.per_hors)


if __name__ == "__main__":
    run()
