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

import config as cfg
import datetime
import glob
import numpy as np
import os
import pandas as pd
import plot
import rcm
import scenarios_calib
import utils
import xarray as xr
from qm import train, predict
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
    d_stn      = cfg.get_d_stn(var)
    p_stn_info = glob.glob(d_stn + "../*.csv")
    p_stn_list = glob.glob(d_stn + "*.csv")
    p_stn_list.sort()

    # Compile data.
    for i in range(0, len(p_stn_list)):

        stn = os.path.basename(p_stn_list[i]).replace(".nc", "").split("_")[1]

        if not(stn in cfg.stns):
            continue

        obs  = pd.read_csv(p_stn_list[i], sep=";")
        time = pd.to_datetime(
            obs["annees"].astype("str") + "-" + obs["mois"].astype("str") + "-" + obs["jours"].astype("str"))

        # Find longitude and latitude.
        lon_lat_data = pd.read_csv(p_stn_info[0], sep=";")
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

            # Calculate wind components.
            uas, vas = utils.sfcwind_2_uas_vas(wind, windfromdir)

            if var == cfg.var_cordex_uas:
                data_xarray = uas
            else:
                data_xarray = vas
            da = xr.DataArray(data_xarray, coords=[("time", time), ("lon", [lon]), ("lat", [lat])])
            da.name = var
            if var == cfg.var_cordex_uas:
                da.attrs["standard_name"] = "eastward_wind"
                da.attrs["long_name"] = "Eastward near-surface wind"
            else:
                da.attrs["standard_name"] = "northward_wind"
                da.attrs["long_name"] = "Northward near-surface wind"
            da.attrs["units"] = "m s-1"
            da.attrs["grid_mapping"] = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

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
                da.attrs["grid_mapping"]  = "regular_lon_lat"
                da.attrs["comments"]      = "station data converted from Total Precip (mm) using a density of 1000 kg/m³"

            # Temperature.
            else:

                data        = obs[stn].values
                data_xarray = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
                da          = xr.DataArray(data_xarray, coords=[("time", time), ("lon", [lon]), ("lat", [lat])])

                da.name                   = var
                da.attrs["standard_name"] = "temperature"
                da.attrs["long_name"]     = "temperature"
                da.attrs["units"]         = "degree_C"
                da.attrs["grid_mapping"]  = "regular_lon_lat"
                da.attrs["comments"]      = "station data converted from degree C"

            # Create dataset.
            ds = da.to_dataset()

        else:
            ds = None

        # Wind, temperature or precipitation ---------------------------------------------------------------------------

        # Add attributes to lon, lat, time, elevation, and the grid.
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
        p_obs = d_stn + var + "_" + ds.attrs["Station Name"] + ".nc"
        utils.save_dataset(ds, p_obs)
        ds.close()
        da.close()


def extract(var, p_obs, p_rcp_ref, p_rcp_fut, p_raw, p_regrid):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extracts data.
    TODO.MAB: Something could be done to define the search radius as a function of the occurrence (or not) of a pixel
              storm (data anomaly).

    Parameters
    ----------
    var : str
        Weather variable.
    p_obs : str
        Path of the file containing observations.
    p_rcp_ref : str
        Path of the RCP file for the historical data.
    p_rcp_fut : str
        Path of the RCP file for the projected data.
    p_raw : str
        Path of the directory containing raw data.
    p_regrid : str
        Path of the file containing regrid data.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load observations.
    ds_obs = xr.open_dataset(p_obs)

    # Define longitude and latitude of station.
    lat_stn = round(float(ds_obs.lat.values), 1)
    lon_stn = round(float(ds_obs.lon.values), 1)

    # Define a squared zone around the station.
    lat_bnds = [lat_stn - cfg.radius, lat_stn + cfg.radius]
    lon_bnds = [lon_stn - cfg.radius, lon_stn + cfg.radius]

    # Data extraction --------------------------------------------------------------------------------------------------

    # The idea is to extract historical and projected data based on a range of longitude, latitude, years.

    utils.log("Extracting data from NetCDF file (raw)", True)

    p_sim = cfg.get_d_sim("", cfg.cat_raw, var)
    ds = rcm.extract_variable(p_rcp_ref, p_rcp_fut, var, lat_bnds, lon_bnds,
                              priority_timestep=cfg.priority_timestep[cfg.variables_cordex.index(var)], tmpdir=p_sim)

    # Temporal interpolation -------------------------------------------------------------------------------------------

    # This checks if the data is daily. If it is sub-daily, resample to daily. This can take a while, but should save us
    # computation time later.

    if cfg.opt_scen_itp_time:

        msg = "Temporal interpolation is "

        if ds.time.isel(time=[0, 1]).diff(dim="time").values[0] <\
                np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0]:
            msg = msg + "running"
            utils.log(msg, True)
            ds = ds.resample(time="1D").mean(dim="time", keep_attrs=True)
        else:
            msg = msg + "not required (data is daily)"
            utils.log(msg, True)

    # Spatial interpolation --------------------------------------------------------------------------------------------

    ds_regrid = None
    if cfg.opt_scen_itp_space:

        msg = "Spatial interpolation is "

        # Method 1: Convert data to a new grid.
        if cfg.opt_scen_regrid:

            msg = msg + "running"
            utils.log(msg, True)

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

            msg = msg + "not required"
            utils.log(msg, True)

            ds_regrid = ds.sel(rlat=lat_stn, rlon=lon_stn, method="nearest", tolerance=1)

    # Save as NetCDF ---------------------------------------------------------------------------------------------------

    if "ds" in locals():
        utils.log("Writing NetCDF file (raw)", True)
        utils.save_dataset(ds, p_raw)

    if "ds_regrid" in locals():
        utils.log("Writing NetCDF file (regrid)", True)
        utils.save_dataset(ds_regrid, p_regrid)


def preprocess(var, p_obs, p_obs_fut, p_regrid, p_regrid_ref, p_regrid_fut):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs pre-processing.

    Parameters
    ----------
    var : str
        Variable.
    p_obs : str
        Path of file that contains observation data.
    p_obs_fut : str
        Path of file that contains observation data for the future period.
    p_regrid : str
        Path of directory that contains regrid data.
    p_regrid_ref : str
        Path of directory that contains regrid data for the reference period.
    p_regrid_fut : str
        Path of directory that contains regrid data for the future period.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_obs = xr.open_dataset(p_obs)
    ds_fut = xr.open_dataset(p_regrid)

    # Observations -----------------------------------------------------------------------------------------------------

    # Interpolate temporally by dropping February 29th and by sub-selecting between reference years.
    ds_obs = ds_obs.sel(time=~((ds_obs.time.dt.month == 2) & (ds_obs.time.dt.day == 29)))
    ds_obs = ds_obs.where(
        (ds_obs.time.dt.year >= cfg.per_ref[0]) & (ds_obs.time.dt.year <= cfg.per_ref[1]), drop=True)

    # Add perturbation.
    if var == cfg.var_cordex_pr:
        ds_obs[var].values = ds_obs[var].values + np.random.rand(ds_obs[var].values.shape[0], 1, 1) * 1e-12

    # Save NetCDF file.
    utils.save_dataset(ds_obs, p_obs_fut)

    # Simulated climate (future period) --------------------------------------------------------------------------------

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
            utils.log("Calendar type not recognized", True)
            raise ValueError

    # Save dataset.
    utils.save_dataset(ds_fut_365, p_regrid_fut)

    # DEBUG: Plot 365 versus 360 calendar.
    if cfg.opt_plt_365vs360:
        plot.plot_360_vs_365(ds_fut, ds_fut_365, var)

    # Simulated climate (reference period) -----------------------------------------------------------------------------

    ds_ref = ds_fut_365.sel(time=slice(str(cfg.per_ref[0]), str(cfg.per_ref[1])))

    if var in [cfg.var_cordex_pr, cfg.var_cordex_clt]:
        pos = np.where(np.squeeze(ds_ref[var].values) > 0.01)[0]
        ds_ref[var][pos] = 1e-12

    # Save dataset.
    utils.save_dataset(ds_ref, p_regrid_ref)


def postprocess(var, stn, nq, up_qmf, time_int, p_obs, p_regrid_ref, p_regrid_fut, p_qqmap):

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
    p_obs : str
        Path of file containing observations for the future period.
    p_regrid_ref : str
        Path of file containing regrid data for the reference period.
    p_regrid_fut : str
        Path of file containing regrid data for the future period.
    p_qqmap : str
        Path of NetCDF file containing QQ data.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    ds_obs = xr.open_dataset(p_obs)[var]
    ds_fut = xr.open_dataset(p_regrid_fut)[var]
    ds_ref = xr.open_dataset(p_regrid_ref)[var]

    # Figures.
    fn_fig  = p_regrid_fut.split("/")[-1].replace("_4qqmap.nc", "_postprocess.png")
    title   = fn_fig[:-4] + "_time_int_" + str(time_int) + "_up_qmf_" + str(up_qmf) + "_nq_" + str(nq)
    p_fig   = cfg.get_d_sim(stn, cfg.cat_fig + "/postprocess", var) + fn_fig

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
            utils.log("Calendar type not recognized.", True)
            raise ValueError

    # Plot 365 versus 360 data.
    if cfg.opt_plt_365vs360:
        plot.plot_360_vs_365(ds_fut, ds_fut_365)

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
        if p_qqmap != "":
            utils.save_dataset(ds_qqmap, p_qqmap)
    except ValueError as err:
        utils.log("Failed to create QQMAP NetCDF file.", True)
        utils.log(format(err), True)
        pass

    # Plot PP_FUT_OBS.
    if cfg.opt_plt_pp_fut_obs:
        plot.plot_postprocess_fut_obs(ds_obs, ds_fut_365, ds_qqmap, var, p_fig, title)

    return ds_qqmap


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

    # Create directory.
    d_exec = cfg.get_d_sim("", "", "")
    if not(os.path.isdir(d_exec)):
        os.makedirs(d_exec)

    # Step #2: Data selection.

    # Step #2a: Convert observations from CSV to NetCDF files.
    # This creates one .nc file per variable-station in ~/<country>/<project>/<stn>/obs/<source>/<var>/.
    utils.log("=")
    utils.log("Step #2c  Converting observations from CSV to NetCDF files")
    if cfg.opt_scen_read_obs_netcdf:
        for var in cfg.variables_cordex:
            read_obs_csv(var)

    # Step #2b: List directories potentially containing CORDEX files (but not necessarily for all selected variables).
    utils.log("=")
    utils.log("Step #2d  Listing directories with CORDEX files")
    list_cordex = utils.list_cordex(cfg.d_cordex, cfg.rcps)

    # Loop through variables.
    for idx_var in range(0, len(cfg.variables_cordex)):
        var = cfg.variables_cordex[idx_var]

        # Select file name for observation.
        d_stn = cfg.get_d_stn(var)
        p_stn_list = glob.glob(d_stn + "*.nc")
        p_stn_list.sort()

        # Loop through stations.
        for idx_stn in range(0, len(p_stn_list)):

            # Station name.
            stn = os.path.basename(p_stn_list[idx_stn]).replace(".nc", "").replace(var + "_", "")
            if not (stn in cfg.stns):
                continue

            # Create directories.
            d_obs    = cfg.get_d_sim(stn, cfg.cat_obs, var)
            d_raw    = cfg.get_d_sim(stn, cfg.cat_raw, var)
            d_fut    = cfg.get_d_sim(stn, cfg.cat_qqmap, var)
            d_regrid = cfg.get_d_sim(stn, cfg.cat_regrid, var)
            if not (os.path.isdir(d_obs)):
                os.makedirs(d_obs)
            if not (os.path.isdir(d_raw)):
                os.makedirs(d_raw)
            if not (os.path.isdir(d_fut)):
                os.makedirs(d_fut)
            if not (os.path.isdir(d_regrid)):
                os.makedirs(d_regrid)

            # Loop through RCPs.
            for rcp in cfg.rcps:

                # Loop through simulations.
                for idx_sim in range(len(list_cordex[rcp])):

                    sim      = list_cordex[rcp][idx_sim]
                    sim_hist = list_cordex[rcp + "_historical"][idx_sim]

                    # Get simulation name.
                    c = sim.split("/")
                    sim_name = c[cfg.get_idx_inst()] + "_" + c[cfg.get_idx_gcm()]

                    utils.log("=")
                    utils.log("Variable   : " + var)
                    utils.log("Station    : " + stn)
                    utils.log("RCP        : " + rcp)
                    utils.log("Simulation : " + sim_name)
                    utils.log("=")

                    # Skip iteration if the variable 'var' is not available in the current directory.
                    p_sim_list      = glob.glob(sim + "/" + var + "/*.nc")
                    p_sim_list_hist = glob.glob(sim_hist + "/" + var + "/*.nc")
                    if (len(p_sim_list) == 0) or (len(p_sim_list_hist) == 0):
                        utils.log("Skipping iteration: data not available for simulation-variable.", True)
                        continue

                    # Files within CORDEX or CORDEX-NA.
                    if "CORDEX" in sim:
                        p_raw = d_raw + var + "_" + c[cfg.get_idx_inst()] + "_" +\
                                c[cfg.get_idx_gcm()].replace("*", "_") + ".nc"
                    elif len(sim) == 3:
                        p_raw = d_raw + var + "_Ouranos_" + sim + ".nc"
                    else:
                        p_raw = None

                    # Skip iteration if the simulation or simulation-variable is in the exception list.
                    is_sim_except = False
                    for sim_except in cfg.sim_excepts:
                        if sim_except in p_raw:
                            is_sim_except = True
                            utils.log("Skipping iteration: simulation-variable exception.", True)
                            break
                    is_var_sim_except = False
                    for var_sim_except in cfg.var_sim_excepts:
                        if var_sim_except in p_raw:
                            is_var_sim_except = True
                            utils.log("Skipping iteration: simulation exception.", True)
                            break
                    if is_sim_except or is_var_sim_except:
                        continue

                    # Paths and NetCDF files.
                    p_regrid     = p_raw.replace(cfg.cat_raw, cfg.cat_regrid)
                    p_qqmap      = p_raw.replace(cfg.cat_raw, cfg.cat_qqmap)
                    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + cfg.cat_qqmap + ".nc"
                    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + cfg.cat_qqmap + ".nc"
                    p_obs_fut    = cfg.get_p_obs(stn, var)
                    p_obs        = cfg.get_p_stn(var, stn)

                    # Figures.
                    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/wflow", var) +\
                            p_regrid_fut.split("/")[-1].replace("4qqmap.nc", "wflow.png")

                    # Step #3: Spatial and temporal extraction.
                    # Step #4: Grid transfer or interpolation.
                    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/ and
                    #              one .nc file in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
                    msg = "Step #3-4 Spatial & temporal extraction and grid transfer (or interpolation) is "
                    if cfg.opt_scen_extract and (not(os.path.isfile(p_raw)) or not(os.path.isfile(p_regrid))):
                        utils.log(msg + "running")
                        extract(var, p_obs, sim_hist, sim, p_raw, p_regrid)
                    else:
                        utils.log(msg + "not required")

                    # Adjusting the datasets associated with observations and simulated conditions
                    # (for the reference and future periods) to ensure that calendar is based on 365 days per year and
                    # that values are within boundaries (0-100%).
                    # This creates two .nc files in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
                    utils.log("-")
                    msg = "Step #4.5 Pre-processing is "
                    if cfg.opt_scen_preprocess and \
                        (not(os.path.isfile(p_regrid_ref)) or not(os.path.isfile(p_regrid_fut))):
                        utils.log(msg + "running")
                        preprocess(var, p_obs, p_obs_fut, p_regrid, p_regrid_ref, p_regrid_fut)
                    else:
                        utils.log(msg + "not required")

                    # Step #5: Post-processing.

                    # Step #5a: Calculate adjustment factors.
                    utils.log("-")
                    msg = "Step #5a  Calculating adjustment factors is "
                    if cfg.opt_calib:
                        utils.log(msg + "running")
                        scenarios_calib.bias_correction(stn, var, sim_name)
                    else:
                        utils.log(msg + "not required")
                    df_sel = cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                              (cfg.df_calib["stn"] == stn) &
                                              (cfg.df_calib["var"] == var)]
                    nq       = float(df_sel["nq"])
                    up_qmf   = float(df_sel["up_qmf"])
                    time_int = float(df_sel["time_int"])
                    bias_err = float(df_sel["bias_err"])

                    # Display calibration parameters.
                    msg = "Selected parameters: nq=" + str(nq) + ", up_qmf=" + str(up_qmf) + \
                          ", time_int=" + str(time_int) + ", bias_err=" + str(bias_err)
                    utils.log(msg, True)

                    # Step #5b: Statistical downscaling.
                    # Step #5c: Bias correction.
                    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/qqmap/<var>/.
                    # This creates one .png file in ~/sim_climat/<country>/<project>/<stn>/fig/postprocess/<var>/.
                    utils.log("-")
                    msg = "Step #5bc Statistical downscaling and bias adjustment is "
                    if cfg.opt_scen_postprocess and not(os.path.isfile(p_qqmap)):
                        utils.log(msg + "running")
                        postprocess(var, stn, int(nq), up_qmf, int(time_int), p_obs, p_regrid_ref,
                                    p_regrid_fut, p_qqmap)
                    else:
                        utils.log(msg + "not required")

                    # Generates extra plot.
                    # This creates one .png file in ~/sim_climat/<country>/<project>/<stn>/fig/wflow/<var>/.
                    if cfg.opt_plt_ref_fut:
                        plot.plot_ref_fut(var, int(nq), up_qmf, int(time_int), p_regrid_ref, p_regrid_fut, p_fig)


if __name__ == "__main__":
    run()
