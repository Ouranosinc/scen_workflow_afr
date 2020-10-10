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
import os.path
import plot
import utils
import xarray as xr
import xclim.indices as indices


def generate(idx_name, idx_threshs):

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

    # List stations.
    if cfg.opt_ra:
        stns = [cfg.obs_src]
    else:
        stns = cfg.stns

    # Loop through stations.
    for stn in stns:

        # Verify if this variable is available for the current station.
        vars_avail = True
        for var in vars:
            if not(os.path.isdir(cfg.get_d_sim(stn, cfg.cat_qqmap, var))):
                vars_avail = False
                break
        if not vars_avail:
            continue

        # Loop through emissions scenarios.
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

                # Load datasets (one per variable).
                ds_scen = []

                for var in range(0, len(vars)):
                    ds = utils.open_netcdf(p_sim[var][i_sim])
                    ds_scen.append(ds)

                # Adjust units.
                idx_threshs_str = []
                for _ in range(0, len(ds_scen)):
                    if rcp == cfg.rcp_ref:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh) + " C")
                    else:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh + cfg.d_KC) + " K")

                # Indices ----------------------------------------------------------------------------------------------

                # TODO: Below, unit conversion should not be required. The unit in the file produced by the scenario
                #       workflow is "degree_C", but it should be "C". This can be fixed eventually, but it's not a
                #       priority.

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
                    if rcp == cfg.rcp_ref:
                        ds_scen_tasmax[cfg.attrs_units] = "C"
                        ds_scen_tasmax.attrs[cfg.attrs_units] = "C"
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
                ds_idx.attrs[cfg.attrs_units] = idx_units
                if "dim_0" in list(ds_idx.dims):
                    ds_idx = ds_idx.rename_dims({"dim_0": cfg.dim_time})
                if "dim_1" in list(ds_idx.dims):
                    ds_idx = ds_idx.rename_dims({"dim_1": cfg.dim_lon, "dim_2": cfg.dim_lat})
                else:
                    ds_idx = ds_idx.expand_dims(lon=1)
                    ds_idx = ds_idx.expand_dims(lat=1)
                ds_idx.attrs[cfg.attrs_sname] = idx_name
                ds_idx.attrs[cfg.attrs_lname] = idx_name
                if cfg.dim_longitude in list(ds_scen[0].dims):
                    ds_idx.assign_attrs({cfg.dim_longitude: ds_scen[0][cfg.dim_longitude]})
                    ds_idx.assign_attrs({cfg.dim_latitude: ds_scen[0][cfg.dim_latitude]})
                elif cfg.dim_rlon in list(ds_scen[0].dims):
                    ds_idx.assign_attrs({cfg.dim_rlon: ds_scen[0][cfg.dim_rlon]})
                    ds_idx.assign_attrs({cfg.dim_rlat: ds_scen[0][cfg.dim_rlat]})
                elif cfg.dim_lon in list(ds_scen[0].dims):
                    ds_idx.assign_attrs({cfg.dim_lon: ds_scen[0][cfg.dim_lon]})
                    ds_idx.assign_attrs({cfg.dim_lat: ds_scen[0][cfg.dim_lat]})

                # Adjust calendar.
                year_1 = cfg.per_fut[0]
                year_n = cfg.per_fut[1]
                if rcp == cfg.rcp_ref:
                    year_1 = max(cfg.per_ref[0], int(str(ds_scen[0][cfg.dim_time][0].values)[0:4]))
                    year_n = min(cfg.per_ref[1],
                                 int(str(ds_scen[0][cfg.dim_time][len(ds_scen[0][cfg.dim_time]) - 1].values)[0:4]))
                ds_idx[cfg.dim_time] = utils.reset_calendar(ds_idx, year_1, year_n, cfg.freq_YS)

                # Save result to NetCDF file.
                if rcp == cfg.rcp_ref:
                    p_idx = cfg.get_p_obs(stn, idx_name)
                else:
                    p_idx = cfg.get_d_sim(stn, cfg.cat_idx, idx_name) +\
                            os.path.basename(p_sim[0][i_sim]).replace(vars[0], idx_name)
                utils.save_netcdf(ds_idx, p_idx, "idx")


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Calculate indices.
    msg = "Step #6   Calculation of climate indices is "
    if cfg.opt_idx:

        msg = msg + "running"
        utils.log(msg, True)

        for i in range(0, len(cfg.idx_names)):
            generate(cfg.idx_names[i], cfg.idx_threshs[i])

    else:
        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)

    # Generate plots.
    if cfg.opt_plot:

        utils.log("=")
        utils.log("Generating time series.", True)

        for i in range(len(cfg.idx_names)):
            plot.plot_ts(cfg.idx_names[i], cfg.idx_threshs[i])

        # Perform interpolation (requires multiples stations).
        # Heat maps are not generated:
        # - the result is not good with a limited number of stations;
        # - calculation is very slow (something is wrong).
        if cfg.opt_plot_heat and (len(cfg.stns) > 1):

            utils.log("=")
            utils.log("Generating heat maps.", True)

            for i in range(len(cfg.idx_names)):

                # Reference period.
                plot.plot_heatmap(cfg.idx_names[i], cfg.idx_threshs[i], cfg.rcp_ref, [cfg.per_ref])

                # Future period.
                for rcp in cfg.rcps:
                    plot.plot_heatmap(cfg.idx_names[i], cfg.idx_threshs[i], rcp, cfg.per_hors)


if __name__ == "__main__":
    run()
