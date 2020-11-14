# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions related to climate indices.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import os.path
import pandas as pd
import plot
import statistics
import utils
import xarray as xr
import xclim.indices as indices


def generate(idx_name: str, idx_threshs: [float]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate a time series.

    Parameters:
    idx_name : str
        Index name.
    idx_threshs : [float]
        Threshold values associated with each indicator.
    --------------------------------------------------------------------------------------------------------------------
    """

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
    stns = cfg.stns if not cfg.opt_ra else [cfg.obs_src]
    for stn in stns:

        # Verify if this variable is available for the current station.
        vars_avail = True
        for var in vars:
            if not(os.path.isdir(cfg.get_d_scen(stn, cfg.cat_qqmap, var))):
                vars_avail = False
                break
        if not vars_avail:
            continue

        # Loop through emissions scenarios.
        for rcp in rcps:

            utils.log("-")
            utils.log("Index             : " + idx_name, True)
            utils.log("Station           : " + stn, True)
            utils.log("Emission scenario : " + cfg.get_rcp_desc(rcp), True)
            utils.log("-")

            # Analysis of simulation files -----------------------------------------------------------------------------

            utils.log("Collecting simulation files.", True)

            # List simulation files. As soon as there is no file for one variable, the analysis for the current RCP
            # needs to abort.
            n_sim = 0
            p_sim = []
            for var in vars:
                if rcp == cfg.rcp_ref:
                    p_sim_i = cfg.get_d_scen(stn, cfg.cat_obs, "") + var + "/" + var + "_" + stn + ".nc"
                    if os.path.exists(p_sim_i) and (type(p_sim_i) is str):
                        p_sim_i = [p_sim_i]
                else:
                    p_format = cfg.get_d_scen(stn, cfg.cat_qqmap, "") + var + "/*_" + rcp + ".nc"
                    p_sim_i = glob.glob(p_format)
                if not p_sim_i:
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

                # Skip iteration if the file already exists.
                if rcp == cfg.rcp_ref:
                    p_idx = cfg.get_d_idx(stn, idx_name) + idx_name + "_ref.nc"
                else:
                    p_idx = cfg.get_d_scen(stn, cfg.cat_idx, idx_name) +\
                            os.path.basename(p_sim[0][i_sim]).replace(vars[0], idx_name)
                if os.path.exists(p_idx) and (not cfg.opt_force_overwrite):
                    continue

                # Load datasets (one per variable).
                ds_scen = []
                for i_var in range(0, len(vars)):
                    var = vars[i_var]
                    ds = utils.open_netcdf(p_sim[i_var][i_sim])
                    if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
                        if ds[var].attrs[cfg.attrs_units] == cfg.unit_K:
                            ds[var] = ds[var] - cfg.d_KC
                            ds[var].attrs[cfg.attrs_units] = cfg.unit_C
                        elif rcp == cfg.rcp_ref:
                            ds[var][cfg.attrs_units] = cfg.unit_C
                            ds[var].attrs[cfg.attrs_units] = cfg.unit_C
                    ds_scen.append(ds)

                # Adjust units.
                idx_threshs_str = []
                for _ in range(0, len(ds_scen)):
                    if rcp == cfg.rcp_ref:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh) + " " + cfg.unit_C)
                    else:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh + cfg.d_KC) + " " + cfg.unit_K)

                # Indices ----------------------------------------------------------------------------------------------

                # TODO.YR: Below, unit conversion should not be required. The unit in the file produced by the scenario
                #          workflow is "degree_C", but it should be "C". This can be fixed eventually, but it's not a
                #          priority.

                idx_units = None
                arr_idx = None

                # ==========================================================
                # TODO.CUSTOMIZATION.BEGIN
                # When adding a new climate index, calculate the index by
                # copying the following code block.
                # ==========================================================

                # Number of days where daily maximum temperature exceeds a threshold value.
                if idx_name == cfg.idx_tx_days_above:
                    ds_scen_tasmax = ds_scen[0][cfg.var_cordex_tasmax]
                    idx_thresh_str_tasmax = idx_threshs_str[0]
                    arr_idx = indices.tx_days_above(ds_scen_tasmax, idx_thresh_str_tasmax).values
                    idx_units = cfg.unit_1

                # ==========================================================
                # TODO.CUSTOMIZATION.END
                # ==========================================================

                # Create data array.
                da_idx = xr.DataArray(arr_idx)
                da_idx.name = idx_name

                # Create dataset.
                # For unknown reason, we cannot assign units to the data array (the saved NetCDF file will not open).
                ds_idx = da_idx.to_dataset()
                ds_idx.attrs[cfg.attrs_units] = idx_units
                if "dim_0" in list(ds_idx.dims):
                    ds_idx = ds_idx.rename_dims({"dim_0": cfg.dim_time})
                if "dim_1" in list(ds_idx.dims):
                    ds_idx = ds_idx.rename_dims({"dim_1": cfg.dim_lat, "dim_2": cfg.dim_lon})
                else:
                    ds_idx = ds_idx.expand_dims(lon=1)
                    ds_idx = ds_idx.expand_dims(lat=1)
                ds_idx.attrs[cfg.attrs_sname] = idx_name
                ds_idx.attrs[cfg.attrs_lname] = idx_name
                ds_idx = utils.copy_coordinates(ds_scen[0], ds_idx)
                ds_idx[idx_name] = utils.copy_coordinates(ds_scen[0][vars[0]], ds_idx[idx_name])

                # Adjust calendar.
                year_1 = cfg.per_fut[0]
                year_n = cfg.per_fut[1]
                if rcp == cfg.rcp_ref:
                    year_1 = max(cfg.per_ref[0], int(str(ds_scen[0][cfg.dim_time][0].values)[0:4]))
                    year_n = min(cfg.per_ref[1],
                                 int(str(ds_scen[0][cfg.dim_time][len(ds_scen[0][cfg.dim_time]) - 1].values)[0:4]))
                ds_idx[cfg.dim_time] = utils.reset_calendar(ds_idx, year_1, year_n, cfg.freq_YS)

                # Save result to NetCDF file.
                desc = "/" + idx_name + "/" + os.path.basename(p_idx)
                utils.save_netcdf(ds_idx, p_idx, desc=desc)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Indices ----------------------------------------------------------------------------------------------------------

    # Calculate indices.
    utils.log("=")
    msg = "Step #6   Calculation of indices is "
    if cfg.opt_idx:

        msg = msg + "running"
        utils.log(msg, True)

        for i in range(0, len(cfg.idx_names)):
            generate(cfg.idx_names[i], cfg.idx_threshs[i])

    else:
        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)

    # Statistics -------------------------------------------------------------------------------------------------------

    utils.log("=")
    msg = "Step #7a  Calculation of statistics (indices) is "
    if cfg.opt_stat:

        msg = msg + "running"
        utils.log(msg)
        statistics.calc_stats(cfg.cat_idx)

    else:

        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)

    utils.log("-")
    msg = "Step #7b  Export to CSV files (indices) is "
    if cfg.opt_save_csv and not cfg.opt_ra:

        msg = msg + "running"
        utils.log(msg)

        utils.log("-")
        utils.log("Step #7b1 Generating times series (indices).")
        statistics.calc_time_series(cfg.cat_idx)

        if not cfg.opt_ra:
            utils.log("-")
            utils.log("Step #7b2 Converting NetCDF to CSV files (indices).")
            statistics.conv_nc_csv(cfg.cat_idx)

    else:

        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)

    # Plots ------------------------------------------------------------------------------------------------------------

    # Generate plots.
    if cfg.opt_plot:

        if not cfg.opt_save_csv:
            utils.log("=")
            utils.log("Step #8b  Generating time series (indices).")
            statistics.calc_time_series(cfg.cat_idx)

    # Generate maps.
    # Heat maps are not generated from data at stations:
    # - the result is not good with a limited number of stations;
    # - calculation is very slow (something is wrong).
    if cfg.opt_ra and (cfg.opt_plot_heat or cfg.opt_save_csv):

        utils.log("=")
        utils.log("Step #8c  Generating heat maps (indices).")

        for i in range(len(cfg.idx_names)):

            # Get the minimum and maximum values in the statistics file.
            idx = cfg.idx_names[i]
            p_stat = cfg.get_d_scen(cfg.obs_src, cfg.cat_stat, idx) + idx + "_" + cfg.obs_src + ".csv"
            if not os.path.exists(p_stat):
                z_min = z_max = -1
            else:
                df_stats = pd.read_csv(p_stat, sep=",")
                vals = df_stats[(df_stats["stat"] == cfg.stat_min) |
                                (df_stats["stat"] == cfg.stat_max) |
                                (df_stats["stat"] == "none")]["val"]
                z_min = min(vals)
                z_max = max(vals)

            # Reference period.
            statistics.calc_heatmap(cfg.idx_names[i], cfg.idx_threshs[i], cfg.rcp_ref, [cfg.per_ref], z_min, z_max)

            # Future period.
            for rcp in cfg.rcps:
                statistics.calc_heatmap(cfg.idx_names[i], cfg.idx_threshs[i], rcp, cfg.per_hors, z_min, z_max)


if __name__ == "__main__":
    run()
