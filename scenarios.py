# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Workflow functions.
#
# TODO: Build a function that verifies the amount of data that is available in a dataset using:
#       ds.notnull().groupby('time.year').sum('time') or
#       xclim.core.checks.missing_[pct|any|wmo]
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# 2. bourgault.marcandre@ouranos.ca (original)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import scenarios_calib
import config as cfg
import rcm
import utils

# Ouranos packages.
from xclim.indices import sfcwind_2_uas_vas
from xsd.qm import train, predict

# Other packages.
import datetime
import glob
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import os
import xarray as xr
from scipy import signal
from scipy.interpolate import griddata


def read_obs_csv(var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts observations to NetCDF.

    Parameters
    ----------
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Station list file and station files.
    path_stn    = cfg.get_path_stn("", "")
    fn_stn_list = glob.glob(path_stn + "*.csv")
    fn_stn      = glob.glob(path_stn + var + "/*.csv")

    # Compile data.
    for i in range(0, len(fn_stn)):

        # DEBUG: i = 0
        stn = os.path.basename(fn_stn[i]).replace(".nc", "").split("_")[1]
        # DEBUG: print(stn)

        if not(stn in cfg.stns):
            continue

        obs  = pd.read_csv(fn_stn[i], sep=";")
        time = pd.to_datetime(
            obs["annees"].astype("str") + "-" + obs["mois"].astype("str") + "-" + obs["jours"].astype("str"))

        # Find longitude and latitude.
        lon_lat_data = pd.read_csv(fn_stn_list[0], sep=";")
        lon = float(lon_lat_data[lon_lat_data["station"] == stn]["lon"])
        lat = float(lon_lat_data[lon_lat_data["station"] == stn]["lat"])

        # Wind ---------------------------------------------------------------------------------------------------------

        if var in [cfg.var_cordex_uas, cfg.var_cordex_vas]:

            obs_dd = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("vv", axis=1)), index=time, columns=[stn])
            obs_vv = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("dd", axis=1)), index=time, columns=[stn])

            # Direction or DD.
            data_dd        = obs_dd[stn].values
            data_xarray_dd = np.expand_dims(np.expand_dims(data_dd, axis=1), axis=2)
            windfromdir    = xr.DataArray(data_xarray_dd, coords=[("time", time), ("lon", [lon]), ("lat", [lat])])

            # Velocity or VV.
            data_vv        = obs_vv[stn].values
            data_xarray_vv = np.expand_dims(np.expand_dims(data_vv, axis=1), axis=2)

            wind           = xr.DataArray(data_xarray_vv, coords=[("time", time), ("lon", [lon]), ("lat", [lat])])
            wind.attrs["units"] = "m s-1"

            uas, vas = sfcwind_2_uas_vas(wind, windfromdir)

            # Information.
            if var == cfg.var_cordex_uas:
                ds = uas.to_dataset()
            else:
                ds = vas.to_dataset()

        # Temperature or precipitation ---------------------------------------------------------------------------------

        elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmax, cfg.var_cordex_tasmin, cfg.var_cordex_pr]:

            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])

            # Precipitation (involves converting from mm to kg m-2 s-1.
            if var == cfg.var_cordex_pr:

                data        = obs[stn].values / cfg.spd
                data_xarray = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
                da          = xr.DataArray(data_xarray, coords=[("time", time), ("lon", [lon]), ("lat", [lat])])

                da.name                   = var
                da.attrs["standard_name"] = "precipitation_flux"
                da.attrs["long_name"]     = "Precipitation"
                da.attrs["units"]         = "kg m-2 s-1"
                # TODO.MAB: There is probably a better CF Convention for point-based data.
                da.attrs["grid_mapping"]  = "regular_lon_lat"
                da.attrs["comments"] = "station data converted from Total Precip (mm) using a density of 1000 kg/m³"

            # Temperature.
            else:

                data        = obs[stn].values
                data_xarray = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
                da          = xr.DataArray(data_xarray, coords=[("time", time), ("lon", [lon]), ("lat", [lat])])

                da.name                   = var
                da.attrs["standard_name"] = "temperature"
                da.attrs["long_name"]     = "temperature"
                da.attrs["units"]         = "degree_C"
                # TODO.MAB: There is probably a better CF Convention for point-based data.
                da.attrs["grid_mapping"]  = "regular_lon_lat"
                da.attrs["comments"]      = "station data converted from degree C"

            # Information.
            ds = da.to_dataset()

        else:
            ds = None

        # Wind, temperature or precipitation ---------------------------------------------------------------------------

        # Add attributes to lon, lat, time, elevation, and the grid.
        # TODO.MAB: There is probably a better CF Convention for point-based data.
        da      = xr.DataArray(np.full(len(time), np.nan), [("time", time)])
        da.name = "regular_lon_lat"
        da.attrs["grid_mapping_name"] = "lonlat"

        # Create dataset and add attributes.
        ds["regular_lon_lat"]         = da
        ds.lon.attrs["standard_name"] = "longitude"
        ds.lon.attrs["long_name"]     = "longitude"
        ds.lon.attrs["units"]         = "degrees_east"
        ds.lon.attrs["axis"]          = "X"
        ds.lat.attrs["standard_name"] = "latitude"
        ds.lat.attrs["long_name"]     = "latitude"
        ds.lat.attrs["units"]         = "degrees_north"
        ds.lat.attrs["axis"]          = "Y"
        ds.attrs["Station Name"]      = stn
        # ds.attrs["Province"]           = station["province"]
        # ds.attrs["Climate Identifier"] = station["ID"]
        # ds.attrs["WMO Identifier"]     = station["WMO_ID"]
        # ds.attrs["TC Identifier"]      = station["TC_ID"]
        # ds.attrs["Institution"]        = "Environment and Climate Change Canada"

        # Save data.
        ds.to_netcdf(path_stn + var + "/" + ds.attrs["Station Name"] + ".nc")
        ds.close()
        da.close()


def extract(var, fn_obs, fn_rcp_hist, fn_rcp_proj, fn_raw, fn_regrid):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extracts data.

    Parameters
    ----------
    var : str
        Weather variable.
    fn_obs : str
        Path of the file containing observations.
    fn_rcp_hist : str
        Path of the RCP file for the historical data.
    fn_rcp_proj : str
        Path of the RCP file for the projected data.
    fn_raw : str
        Path of the directory containing raw data.
    fn_regrid : str
        Path of the file containing regrid data.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Extracting...")

    # Load observations.
    ds_obs = xr.open_dataset(fn_obs)

    # Define longitude and latitude of station.
    lat_stn = round(float(ds_obs.lat.values), 1)
    lon_stn = round(float(ds_obs.lon.values), 1)

    # TODO.MAB: Something could be done to define the search radius as a function of the occurrence
    #           (or not) of a pixel storm (data anomaly).

    # Define a squared zone around the station.
    lat_bnds = [lat_stn - cfg.radius, lat_stn + cfg.radius]
    lon_bnds = [lon_stn - cfg.radius, lon_stn + cfg.radius]

    # Data extraction --------------------------------------------------------------------------------------------------

    # The idea is to extract historical and projected data based on a range of longitude,
    # latitude, years.

    print("Extracting data for " + fn_raw + "...")

    path_tmp = cfg.get_path_sim("", cfg.cat_raw, var)
    ds = rcm.extract_variable(fn_rcp_hist, fn_rcp_proj, var, lat_bnds, lon_bnds,
                              priority_timestep=cfg.priority_timestep[cfg.variables_cordex.index(var)], tmpdir=path_tmp)

    print("Data extraction completed!")

    # Temporal interpolation -------------------------------------------------------------------------------------------

    # This checks if the data is daily. If it is sub-daily, resample to daily. This can take a
    # while, but should save us computation time later.

    if cfg.opt_scen_itp_time:

        print("Performing temporal interpolation...")

        if ds.time.isel(time=[0, 1]).diff(dim="time").values[0] <\
                np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0]:
            print("Data is sub-daily! Resampling... Go take a coffee, this can take a while.")
            ds = ds.resample(time="1D").mean(dim="time", keep_attrs=True)
        else:
            print("Data is daily.")

        print("Temporal interpolation completed!")

    # Spatial interpolation --------------------------------------------------------------------------------------------

    ds_regrid = None
    if cfg.opt_scen_itp_space:

        print("Performing spatial interpolation...")

        # Method 1: Convert data to a new grid.
        if cfg.opt_scen_regrid:

            new_grid = np.meshgrid(ds_obs.lon.values.ravel(), ds_obs.lat.values.ravel())
            if np.min(new_grid[0]) > 0:
                new_grid[0] -= 360
            time_len = len(ds.time)
            new_grid_data = np.empty((time_len, ds_obs.lat.shape[0], ds_obs.lon.shape[0]))

            for time in range(0, time_len):
                new_grid_data[time, :, :] = griddata(
                    (ds.lon.values.ravel(), ds.lat.values.ravel()),
                    ds[var][time, :, :].values.ravel(),
                    (new_grid[0], new_grid[1]),
                    fill_value=np.nan, method="linear")

            new_data = xr.DataArray(new_grid_data,
                                    coords={"time": ds.time[0:time_len],
                                            "lat": float(ds_obs.lat.values),
                                            "lon": float(ds_obs.lon.values)},
                                    dims=["time", "rlat", "rlon"], attrs=ds.attrs)
            ds_regrid = new_data.to_dataset(name=var)

        # Method 2: Take nearest information.
        else:

            ds_regrid = ds.sel(rlat=lat_stn, rlon=lon_stn, method="nearest", tolerance=1)

        print("Spatial interpolation completed!")

    # Save as NetCDF ---------------------------------------------------------------------------------------------------

    if "ds_regrid" in locals():
        print("Writing NetCDF: " + fn_regrid)
        if os.path.exists(fn_regrid):
            os.remove(fn_regrid)
        ds_regrid.to_netcdf(fn_regrid)

    if "ds" in locals():
        print("Writing NetCDF: " + fn_raw)
        if os.path.exists(fn_raw):
            os.remove(fn_raw)
        ds.to_netcdf(fn_raw)

    # if ("ds_obs" in locals()) and not (os.path.isfile(fn_obs)):
    #     print("Writing NetCDF: " + fn_obs)
    #     if os.path.exists(fn_obs):
    #         os.remove(fn_obs)
    #     ds_obs.to_netcdf(fn_obs)

    print("Extraction completed!")


def preprocess(var, fn_obs, fn_obs_fut, fn_regrid, fn_regrid_ref, fn_regrid_fut):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs pre-processing.

    Parameters
    ----------
    var : str
        Variable.
    fn_obs : str
        Path of file that contains observation data.
    fn_obs_fut : str
        Path of file that contains observation data for the future period.
    fn_regrid : str
        Path of directory that contains regrid data.
    fn_regrid_ref : str
        Path of directory that contains regrid data for the reference period.
    fn_regrid_fut : str
        Path of directory that contains regrid data for the future period.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Pre-processing....")

    ds_obs = xr.open_dataset(fn_obs)
    ds_fut = xr.open_dataset(fn_regrid)

    # Observations -----------------------------------------------------------------------------------------------------

    # Interpolate temporally by dropping February 29th and by sub-selecting between reference years.
    ds_obs = ds_obs.sel(time=~((ds_obs.time.dt.month == 2) & (ds_obs.time.dt.day == 29)))
    ds_obs = ds_obs.where(
        (ds_obs.time.dt.year >= cfg.per_ref[0]) & (ds_obs.time.dt.year <= cfg.per_ref[1]), drop=True)

    # Add perturbation.
    if var == cfg.var_cordex_pr:
        ds_obs[var].values = ds_obs[var].values + np.random.rand(ds_obs[var].values.shape[0], 1, 1) * 1e-12

    # Write NetCDF file.
    if not (os.path.isfile(fn_obs_fut)):
        dir_obs_fut = os.path.dirname(fn_obs_fut)
        if not (os.path.isdir(dir_obs_fut)):
            os.makedirs(dir_obs_fut)
        ds_obs.to_netcdf(fn_obs_fut)

    # Simulated climate ------------------------------------------------------------------------------------------------

    # TODO.MAB: This needs to be modified.
    if not (os.path.isfile(fn_regrid)) and (var == cfg.var_cordex_clt):
        print(fn_regrid + " does not exist. Skipping it. \n")
    else:
        print("Creating NetCDF: " + fn_regrid)

        # Simulated climate (future period) ----------------------------------------------------------------------------

        # Make sure that the data we post-process is within the boundaries.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_clt]:
            ds_fut[var].values[ds_fut[var] < 0] = 0
            if var == cfg.var_cordex_clt:
                ds_fut[var].values[ds_fut[var] > 100] = 100

        # Add small perturbation.
        ds_fut[var].values = ds_fut[var].values + np.random.rand(ds_fut[var].values.shape[0]) * 1e-12

        # Remove February 29th.
        ds_fut = ds_fut.sel(time=~((ds_fut.time.dt.month == 2) & (ds_fut.time.dt.day == 29)))

        # Convert to a 365-day calendar.
        if isinstance(ds_fut.time.values[0], np.datetime64):
            ds_fut_365 = ds_fut
        else:
            cf = ds_fut.time.values[0].calendar
            if cf in [cfg.cal_noleap, cfg.cal_365day]:
                ds_fut_365 = ds_fut
            elif cf in [cfg.cal_360day]:
                ds_fut_365 = utils.calendar(ds_fut)
            else:
                logging.error("Calendar type not recognized")
                raise ValueError

        if os.path.exists(fn_regrid_fut):
            os.remove(fn_regrid_fut)
        ds_fut_365.to_netcdf(fn_regrid_fut)

        # DEBUG: Plot 365 versus 360 calendar.
        if cfg.opt_plt_365vs360:
            plt.plot((np.arange(1, 361) / 360) * 365, ds_fut[var][:360].values)
            plt.plot(np.arange(1, 366), ds_fut_365[var].values[:365], alpha=0.5)
            plt.close()

        # Simulated climate (reference period) -------------------------------------------------------------------------

        ds_ref = ds_fut_365.sel(time=slice(str(cfg.per_ref[0]), str(cfg.per_ref[1])))
        # DELETE: ds_ref = ds_fut.where((ds_fut.time.dt.year >= cfg.per_ref[0]) & \
        # DELETE:                       (ds_fut.time.dt.year <= cfg.per_ref[1]), drop=True)

        if var in [cfg.var_cordex_pr, cfg.var_cordex_clt]:
            pos = np.where(np.squeeze(ds_ref[var].values) > 0.01)[0]
            ds_ref[var][pos] = 1e-12

        if os.path.exists(fn_regrid_ref):
            os.remove(fn_regrid_ref)
        ds_ref.to_netcdf(fn_regrid_ref)

    print("Pre-processing completed!")


def postprocess(var, stn, nq, up_qmf, time_int, fn_obs, fn_regrid_ref, fn_regrid_fut, fn_qqmap):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs post-processing.

    Parameters
    ----------
    var : str
        Weather variable.
    stn : str
        Station name.
    nq : int
        ...
    up_qmf : float
        ...
    time_int : int
        ...
    fn_obs : str
        Path of file containing observations for the future period.
    fn_regrid_ref : str
        Path of file containing regrid data for the reference period.
    fn_regrid_fut : str
        Path of file containing regrid data for the future period.
    fn_qqmap : str
        Path of NetCDF file containing QQ data.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Post-processing...")

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Load datasets.
    ds_obs = xr.open_dataset(fn_obs)[var]
    ds_fut = xr.open_dataset(fn_regrid_fut)[var]
    ds_ref = xr.open_dataset(fn_regrid_ref)[var]

    # Figures.
    fn_fig    = fn_regrid_fut.split("/")[-1].replace("_4qqmap.nc", "_postprocess.png")
    sup_title = fn_fig[:-4] + "_time_int_" + str(time_int) + "_up_qmf_" + str(up_qmf) + "_nq_" + str(nq)
    path_fig  = cfg.get_path_sim(stn, cfg.cat_fig + "/postprocess", var)
    if not (os.path.isdir(path_fig)):
        os.makedirs(path_fig)
    fn_fig = path_fig + fn_fig

    # Future -----------------------------------------------------------------------------------------------------------

    # Interpolation of 360 calendar to no-leap calendar.
    if isinstance(ds_fut.time.values[0], np.datetime64):
        ds_fut_365 = ds_fut
    else:
        cf = ds_fut.time.values[0].calendar
        if cf in [cfg.cal_noleap, cfg.cal_365day]:
            ds_fut_365 = ds_fut
        elif cf in [cfg.cal_360day]:
            ds_fut_365 = utils.calendar(ds_fut)
        else:
            logging.error("Calendar type not recognized.")
            raise ValueError

    # Plot 365 versus 360 data.
    if cfg.opt_plt_365vs360:
        plt.plot((np.arange(1, 361) / 360) * 365, ds_fut[:360].values)
        plt.plot(np.arange(1, 366), ds_fut_365[:365].values, alpha=0.5)
        plt.close()

    # Observation ------------------------------------------------------------------------------------------------------

    if var == cfg.var_cordex_pr:
        kind = "*"
        ds_obs.interpolate_na(dim="time")
    elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        ds_obs = ds_obs + 273.15
        ds_obs = ds_obs.interpolate_na(dim="time")
        kind = "+"
    else:
        ds_obs = ds_obs.interpolate_na(dim="time")
        kind = "+"

    # Calculate QMF.
    # TODO.MAB: The detrend_order seems to be not working.
    ds_qmf = train(ds_ref.squeeze(), ds_obs.squeeze(), nq, cfg.group, kind, time_int,
                   detrend_order=cfg.detrend_order)
    if var == cfg.var_cordex_pr:
        ds_qmf.values[ds_qmf > up_qmf] = up_qmf

    # Apply transfer function.
    ds_qqmap = None
    try:
        ds_qqmap = predict(ds_fut_365.squeeze(), ds_qmf, interp=True, detrend_order=cfg.detrend_order)
        del ds_qqmap.attrs["bias_corrected"]
        ds_qqmap.to_netcdf(fn_qqmap)
    except ValueError as err:
        print("Failed to create NetCDF: " + fn_qqmap)
        print(format(err))
        pass

    # Plot PP_FUT_OBS.
    if cfg.opt_plt_pp_fut_obs:

        # Conversion coefficient.
        coef = 1
        if var == cfg.var_cordex_pr:
            coef = cfg.spd

        # Plot.
        f = plt.figure(figsize=(15, 3))
        f.add_subplot(111)
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.04, right=0.99, hspace=0.695, wspace=0.416)
        fs_sup_title = 8
        fs_legend   = 8
        fs_axes     = 8
        legend = ["sim", cfg.cat_obs]
        if ds_qqmap is not None:
            legend.insert(0, "pp")
            (ds_qqmap * coef).groupby(ds_qqmap.time.dt.year).mean().plot()
        (ds_fut_365 * coef).groupby(ds_fut_365.time.dt.year).mean().plot()
        (ds_obs * coef).groupby(ds_obs.time.dt.year).mean().plot()
        plt.legend(legend, fontsize=fs_legend)
        plt.xlabel("Année", fontsize=fs_axes)
        plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
        plt.title("")
        plt.suptitle(sup_title, fontsize=fs_sup_title)
        plt.tick_params(axis='x', labelsize=fs_axes)
        plt.tick_params(axis='y', labelsize=fs_axes)

        if cfg.opt_plt_save:
            plt.savefig(fn_fig)
        # DEBUG: Need to add a breakpoint below to visualize plot.
        if cfg.opt_plt_close:
            plt.close()

    print("Post-processing completed!")


def gen_plot_ref_fut(var, nq, up_qmf, time_int, fn_regrid_ref, fn_regrid_fut, fn_fig):
    """
    --------------------------------------------------------------------------------------------------------------------
    Generates a plot of reference and future periods.

    Parameters
    ----------
    var : str
        Weather var.
    nq : int
        ...
    up_qmf : float
        ...
    time_int : int
        ...
    fn_regrid_ref : str
        Path of the NetCDF file containing data for the reference period.
    fn_regrid_fut : str
        Path of the NetCDF file containing data for the future period.
    fn_fig : str
        Path of output figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Weather variable description and unit.
    var_desc = cfg.get_var_desc(var)
    var_unit = cfg.get_var_unit(var)

    # Load datasets.
    ds_ref = xr.open_dataset(fn_regrid_ref)[var]
    ds_fut = xr.open_dataset(fn_regrid_fut)[var]

    # Fit.
    x     = [*range(len(ds_ref.time))]
    y     = ds_ref.values
    coefs = poly.polyfit(x, y, 4)
    ffit  = poly.polyval(x, coefs)

    # Initialize plot.
    fs_sup_title = 8
    fs_title     = 8
    fs_legend    = 6
    fs_axes      = 8
    f = plt.figure(figsize=(15, 6))
    f.add_subplot(211)
    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.04, right=0.99, hspace=0.40, wspace=0.00)
    sup_title = os.path.basename(fn_fig).replace(".png", "") +\
        "_time_int_" + str(time_int) + "_up_qmf_" + str(up_qmf) + "_nq_" + str(nq)
    plt.suptitle(sup_title, fontsize=fs_sup_title)

    # Upper plot: Reference period.
    f.add_subplot(211)
    plt.plot(ds_ref.time, y)
    plt.plot(ds_ref.time, ffit)
    plt.legend(["référence", "tendance"], fontsize=fs_legend)
    plt.xlabel("Année", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)
    plt.title("Période de référence", fontsize=fs_title)

    # Lower plot: Complete simulation.
    f.add_subplot(212)
    plt.plot(signal.detrend(ds_fut), alpha=0.5)
    plt.plot(y - ffit, alpha=0.5)
    plt.legend(["simulation", "référence"], fontsize=fs_legend)
    plt.xlabel("Jours", fontsize=fs_axes)
    plt.ylabel(var_desc + " [" + var_unit + "]", fontsize=fs_axes)
    plt.tick_params(axis='x', labelsize=fs_axes)
    plt.tick_params(axis='y', labelsize=fs_axes)
    plt.title("Période de simulation", fontsize=fs_title)

    if cfg.opt_plt_save:
        plt.savefig(fn_fig)
    # DEBUG: Need to add a breakpoint below to visualize plot.
    if cfg.opt_plt_close:
        plt.close()


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.

    Raises
    ------
    ValueError
        If calendar type not recognized.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Module scenarios launched.")

    # Create directory.
    path_out = cfg.get_path_sim("", "", "")
    if not(os.path.isdir(path_out)):
        os.makedirs(path_out)

    # Step #2: Data selection.

    # Step #2a: Convert observations from .csv to NetCDF files.
    if cfg.opt_scen_read_obs_netcdf:
        for var in cfg.variables_cordex:
            read_obs_csv(var)

    # Step #2b: List CORDEX files.
    list_cordex = utils.list_cordex(cfg.path_cordex, cfg.rcps)

    # Loop through variables.
    for idx_var in range(0, len(cfg.variables_cordex)):
        var = cfg.variables_cordex[idx_var]

        # Select file name for observation.
        path_stn = cfg.get_path_stn(var, "")
        fn_stn   = glob.glob(path_stn + "*.nc")

        # Loop through stations.
        for idx_stn in range(0, len(fn_stn)):

            # Station name.
            fn_stn   = fn_stn[idx_stn]
            stn = os.path.basename(fn_stn).replace(".nc", "")
            if not (stn in cfg.stns):
                continue

            # Create directories.
            path_obs    = cfg.get_path_sim(stn, cfg.cat_obs, var)
            path_raw    = cfg.get_path_sim(stn, cfg.cat_raw, var)
            path_fut    = cfg.get_path_sim(stn, cfg.cat_qqmap, var)
            path_regrid = cfg.get_path_sim(stn, cfg.cat_regrid, var)
            if not (os.path.isdir(path_obs)):
                os.makedirs(path_obs)
            if not (os.path.isdir(path_raw)):
                os.makedirs(path_raw)
            if not (os.path.isdir(path_fut)):
                os.makedirs(path_fut)
            if not (os.path.isdir(path_regrid)):
                os.makedirs(path_regrid)

            # Loop through RCPs.
            for rcp in cfg.rcps:

                # Loop through simulations.
                for idx_sim in range(len(list_cordex[rcp])):

                    sim      = list_cordex[rcp][idx_sim]
                    sim_hist = list_cordex[rcp + "_historical"][idx_sim]

                    # Get simulation name.
                    c = sim.split("/")
                    sim_name = c[cfg.get_idx_inst()] + "_" + c[cfg.get_idx_gcm()]
                    print("Processing: variable = " + var + "; station = " + stn + ": " + sim_name)

                    # Files within CORDEX or CORDEX-NA.
                    if "CORDEX" in sim:
                        fn_raw = path_raw + var + "_" + c[cfg.get_idx_inst()] + "_" +\
                                 c[cfg.get_idx_gcm()].replace("*", "_") + ".nc"
                    elif len(sim) == 3:
                        fn_raw = path_raw + var + "_Ouranos_" + sim + ".nc"
                    else:
                        fn_raw = None

                    # Skip simulations that are included in the exception list.
                    for sim_except in cfg.sim_excepts:
                        if sim_except in fn_raw:
                            continue

                    # Paths and NetCDF files.
                    fn_regrid     = fn_raw.replace(cfg.cat_raw, cfg.cat_regrid)
                    fn_qqmap      = fn_raw.replace(cfg.cat_raw, cfg.cat_qqmap)
                    fn_regrid_ref = fn_regrid[0:len(fn_regrid) - 3] + "_ref_4" + cfg.cat_qqmap + ".nc"
                    fn_regrid_fut = fn_regrid[0:len(fn_regrid) - 3] + "_4" + cfg.cat_qqmap + ".nc"
                    fn_obs_fut    = cfg.get_path_obs(stn, var)
                    fn_obs        = cfg.get_path_stn(var, stn)

                    # Figures.
                    fn_fig   = fn_regrid_fut.split("/")[-1].replace("4qqmap.nc", "wflow.png")
                    path_fig = cfg.get_path_sim(stn, cfg.cat_fig + "/wflow", var)
                    if not (os.path.isdir(path_fig)):
                        os.makedirs(path_fig)
                    fn_fig = path_fig + fn_fig

                    # Step #3: Spatial and temporal extraction.
                    # Step #4: Grid transfer or interpolation.
                    if cfg.opt_scen_extract or not (os.path.isfile(fn_raw)):
                        extract(var, fn_obs, sim_hist, sim, fn_raw, fn_regrid)

                    # Step #5: Post-processing.

                    # Adjusting the datasets associated with observations and simulated conditions
                    # (for the reference and future periods) to ensure that calendar is based on 365 days per year and
                    # that values are within boundaries (0-100%).
                    if cfg.opt_scen_preprocess:
                        preprocess(var, fn_obs, fn_obs_fut, fn_regrid, fn_regrid_ref, fn_regrid_fut)

                    # Step #5a: Calculate adjustment factors.
                    if cfg.opt_calib:
                        scenarios_calib.run()
                    nq       = cfg.nq[sim_name][stn][idx_var]
                    up_qmf   = cfg.up_qmf[sim_name][stn][idx_var]
                    time_int = cfg.time_int[sim_name][stn][idx_var]

                    # Step #5b: Statistical downscaling.
                    # Step #5c: Bias correction.
                    if cfg.opt_scen_postprocess or not (os.path.isfile(fn_qqmap)):
                        skip_postprocess = False
                        for var_sim_except in cfg.var_sim_excepts:
                            if var_sim_except in fn_raw:
                                skip_postprocess = True
                                break
                        if not skip_postprocess:
                            postprocess(var, stn, int(nq), up_qmf, int(time_int), fn_obs, fn_regrid_ref,
                                        fn_regrid_fut, fn_qqmap)

                    # Generates extra plot.
                    if cfg.opt_plt_ref_fut:
                        gen_plot_ref_fut(var, int(nq), up_qmf, int(time_int), fn_regrid_ref, fn_regrid_fut, fn_fig)

    print("Module scenarios completed successfully.")


if __name__ == "__main__":
    run()
