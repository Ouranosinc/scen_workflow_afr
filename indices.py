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
import pandas as pd
import os.path
import plot
import xarray as xr
import xclim.indices as indices


def calc_index(idx_name, idx_threshs):

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate an index.
    TODO: Determine if the curve needs to correspond to the mean or median value. It is currently the mean value, but it
          would make more sense to use median in this type of analysis.
          Determine if the curve needs to come from observations or from climate scenarios. It is currently associated
          with observations, and I think this is the way to do it. However, the lower and upper limits (the grey zone)
          are associated with climate scenarios.

    Parameters:
    idx_names : str
        Index name.
    idx_threshs : [float]
        Threshold values associated with each indicator.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Configuration.
    # DEBUG: cfg.path_sim = "/exec/yrousseau/sim_climat/burkina/pcci_marc_andre/"
    # DEBUG: rcps = ["ref", cfg.rcp_45, cfg.rcp_85]
    # DEBUG: cfg.variables = [cfg.var_cordex_tasmax]

    # Reference period.
    years_ref   = [cfg.per_ref[0], cfg.per_ref[1]]
    dates_ref   = [str(years_ref[0]) + "-01-01", str(years_ref[1]) + "-12-31"]

    # Future period.
    years_fut   = [cfg.per_ref[1], cfg.per_fut[1]]
    dates_fut   = [str(years_fut[0]) + "-01-01", str(years_fut[1]) + "-12-31"]
    n_years_fut = cfg.per_fut[1] - cfg.per_ref[1] + 1

    # Emission scenarios.
    rcps = ["ref"]
    for rcp in cfg.rcps:
        rcps.append(rcp)

    # Loop through stations.
    for stn in cfg.stns:

        # Loop through emissions scenarios.
        ds_rcp_26 = None
        ds_rcp_45 = None
        ds_rcp_85 = None
        for rcp in rcps:

            # NetCDF files ---------------------------------------------------------------------------------------------

            # Select variables.
            vars = []
            if idx_name == cfg.idx_tx_days_above:
                vars = [cfg.var_cordex_tasmax]

            # List simulation files. As soon as there is no file for one variable, the analysis for the current RCP
            # needs to abort.
            n_sim = 0
            fn_sim = []
            for var in vars:
                if rcp == "ref":
                    fn_sim_i = cfg.get_path_sim(stn, cfg.cat_obs, "") + var + "_" + stn + "_4qqmap.nc"
                    if os.path.exists(fn_sim_i) and (type(fn_sim_i) is str):
                        fn_sim_i = [fn_sim_i]
                else:
                    fn_format = cfg.get_path_sim(stn, cfg.cat_qqmap, "") + var + "/*_" + rcp + ".nc"
                    fn_sim_i = glob.glob(fn_format)
                if fn_sim_i:
                    fn_sim.append(fn_sim_i)
                    n_sim = len(fn_sim_i)
                else:
                    fn_sim = []
            if not fn_sim:
                continue

            # Loop through simulations.
            ds_idx_ref_mean = None
            ds_idx_ref_min = None
            ds_idx_ref_max = None
            ds_idx_fut_mean = None
            ds_idx_fut_min = None
            ds_idx_fut_max = None
            ds_idx_ref = None
            ds_idx_fut = None
            for sim in range(0, n_sim):

                # Scenarios --------------------------------------------------------------------------------------------

                # Load datasets (one per variable).
                ds_scen = []

                for var in range(0, len(vars)):
                    ds = xr.open_dataset(fn_sim[var][sim])
                    ds_scen.append(ds)

                # Adjust units.
                idx_threshs_str = []
                for i in range(0, len(ds_scen)):
                    if rcp == "ref":
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh) + " C")
                    else:
                        for idx_thresh in idx_threshs:
                            idx_threshs_str.append(str(idx_thresh + 273.15) + " K")

                # Indices ----------------------------------------------------------------------------------------------

                # Calculate index.
                ds_idx = None
                # Number of days where daily maximum temperature exceed a threshold.
                if idx_name == cfg.idx_tx_days_above:
                    ds_scen_tasmax = ds_scen[0][cfg.var_cordex_tasmax]
                    # TODO: The following two lines should not be required. The unit in the file produced by the
                    #       scenario workflow is currently "degree_C", but should be "C". This needs to be fixed.
                    if rcp == "ref":
                        ds_scen_tasmax["units"] = "C"
                        ds_scen_tasmax.attrs["units"] = "C"
                    idx_thresh_str_tasmax = idx_threshs_str[0]
                    ds_idx = indices.tx_days_above(ds_scen_tasmax, idx_thresh_str_tasmax)

                # Subset based on dates.
                ds_idx_ref = ds_idx.sel(time=slice(dates_ref[0], dates_ref[1]))
                ds_idx_fut = None
                if rcp == "ref":
                    ds_idx_ref = ds_idx_ref.isel(lon=0, lat=0)
                else:
                    ds_idx_fut = ds_idx.sel(time=slice(dates_fut[0], dates_fut[1]))

                # Convert to a specific time format.
                year_1 = int(str(ds_idx_ref['time'][0].values)[0:4])
                year_n = int(str(ds_idx_ref['time'][len(ds_idx_ref['time'])-1].values)[0:4])
                ds_idx_ref['time'] = pd.date_range(str(year_1)+"-01-01", periods=year_n-year_1+1, freq="YS")
                if rcp != "ref":
                    ds_idx_fut['time'] = pd.date_range(dates_fut[0], periods=n_years_fut, freq="YS")

                # Calculate minimum and maximum values.
                if ds_idx_ref_min is None:
                    ds_idx_ref_min = ds_idx_ref
                    ds_idx_ref_max = ds_idx_ref
                    ds_idx_ref_mean = ds_idx_ref / n_sim
                    if rcp != "ref":
                        ds_idx_fut_min = ds_idx_fut
                        ds_idx_fut_max = ds_idx_fut
                        ds_idx_fut_mean = ds_idx_fut / n_sim
                else:
                    ds_idx_ref_min = xr.ufuncs.minimum(ds_idx_ref_min, ds_idx_ref)
                    ds_idx_ref_max = xr.ufuncs.maximum(ds_idx_ref_max, ds_idx_ref)
                    ds_idx_ref_mean = ds_idx_ref_mean + ds_idx_ref / n_sim
                    if rcp != "ref":
                        if ds_idx_fut_min is None:
                            ds_idx_fut_min = ds_idx_fut
                            ds_idx_fut_max = ds_idx_fut
                            ds_idx_fut_mean = ds_idx_fut / n_sim
                        else:
                            ds_idx_fut_min = xr.ufuncs.minimum(ds_idx_fut_min, ds_idx_fut)
                            ds_idx_fut_max = xr.ufuncs.maximum(ds_idx_fut_max, ds_idx_fut)
                            ds_idx_fut_mean = ds_idx_fut_mean + ds_idx_fut / n_sim

            # Combine statistics into an array (for each RCP).
            ds_ref = [ds_idx_ref_mean, ds_idx_ref_min, ds_idx_ref_max]
            if rcp != "ref":
                ds_fut = [ds_idx_fut_mean, ds_idx_fut_min, ds_idx_fut_max]
                if rcp == cfg.rcp_26:
                    ds_rcp_26 = ds_fut
                elif rcp == cfg.rcp_45:
                    ds_rcp_45 = ds_fut
                elif rcp == cfg.rcp_85:
                    ds_rcp_85 = ds_fut

            # Create NetCDF files.
            path_sim_templ = cfg.get_path_sim(stn, cfg.cat_regrid, "") + idx_name + "/" + idx_name +\
                             "_<rcp>_<stat>_4qqmap.nc"
            dir_out = os.path.dirname(path_sim_templ)
            if not (os.path.isdir(dir_out)):
                os.makedirs(dir_out)
            if rcp == "ref":
                path_sim_ref = path_sim_templ.replace("<rcp>", rcp)
                ds_idx_ref_mean.to_netcdf(path_sim_ref.replace("<stat>", "mean"))
            else:
                path_sim_ref = path_sim_templ.replace("<rcp>", "ref")
                ds_idx_ref_min.to_netcdf(path_sim_ref.replace("<stat>", "min"))
                ds_idx_ref_max.to_netcdf(path_sim_ref.replace("<stat>", "max"))
                path_sim_fut = path_sim_templ.replace("<rcp>", rcp)
                ds_idx_fut_min.to_netcdf(path_sim_fut.replace("<stat>", "min"))
                ds_idx_fut_max.to_netcdf(path_sim_fut.replace("<stat>", "max"))
                ds_idx_fut_mean.to_netcdf(path_sim_fut.replace("<stat>", "mean"))

            # Calculate the X-year mean.
            for years_hor in cfg.per_hor:
                dates_hor = [str(years_hor[0]) + "-01-01", str(years_hor[1]) + "-12-31"]
                val = -1.0
                if rcp == "ref":
                    val = ds_idx_ref.sel(time=slice(dates_ref[0], dates_ref[1])).mean()
                else:
                    val = ds_idx_fut.sel(time=slice(dates_hor[0], dates_hor[1])).mean()

            # Generate plot.
            if rcp == rcps[len(rcps) - 1]:
                path_fig = cfg.get_path_sim(stn, cfg.cat_fig + "/indices", "")
                if not (os.path.isdir(path_fig)):
                    os.makedirs(path_fig)
                fn_fig = path_fig + stn + ".png"
                plot.plot_ts_indices(ds_ref, ds_rcp_26, ds_rcp_45, ds_rcp_85, stn.capitalize(), idx_name, idx_threshs, rcps,
                                     [years_ref[0], years_fut[1]], fn_fig)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through indices.
    for i in range(0, len(cfg.idx_names)):

        # Calculate index.
        calc_index(cfg.idx_names[i], cfg.idx_threshs[i])


if __name__ == "__main__":
    run()
