# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Workflow functions.
#
# TODO.MAB: Build a function that verifies the amount of data that is available in a dataset using:
#           ds.notnull().groupby('time.year').sum('time') or
#           xclim.core.checks.missing_[pct|any|wmo]
# Authors:
# 1. rousseau.yannick@ouranos.ca
# 2. bourgault.marcandre@ouranos.ca (original)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import datetime
import functools
import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import plot
import rcm
import scenarios_calib
import utils
import xarray as xr
import xarray.core.variable as xcv
from qm import train, predict
from scipy.interpolate import griddata


def load_observations(var):

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

        obs  = pd.read_csv(p_stn_list[i], sep=cfg.file_sep)
        time = pd.to_datetime(
            obs["annees"].astype("str") + "-" + obs["mois"].astype("str") + "-" + obs["jours"].astype("str"))

        # Find longitude and latitude.
        lon_lat_data = pd.read_csv(p_stn_info[0], sep=cfg.file_sep)
        lon = float(lon_lat_data[lon_lat_data["station"] == stn][cfg.dim_lon])
        lat = float(lon_lat_data[lon_lat_data["station"] == stn][cfg.dim_lat])

        # Wind ---------------------------------------------------------------------------------------------------------

        if var in [cfg.var_cordex_uas, cfg.var_cordex_vas]:

            obs_dd = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("vv", axis=1)), index=time, columns=[stn])
            obs_vv = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("dd", axis=1)), index=time, columns=[stn])

            # Direction or DD.
            data_dd        = obs_dd[stn].values
            data_xarray_dd = np.expand_dims(np.expand_dims(data_dd, axis=1), axis=2)
            windfromdir    = xr.DataArray(data_xarray_dd,
                                          coords=[(cfg.dim_time, time), (cfg.dim_lon, [lon]), (cfg.dim_lat, [lat])])

            # Velocity or VV.
            data_vv        = obs_vv[stn].values
            data_xarray_vv = np.expand_dims(np.expand_dims(data_vv, axis=1), axis=2)
            wind           = xr.DataArray(data_xarray_vv,
                                          coords=[(cfg.dim_time, time), (cfg.dim_lon, [lon]), (cfg.dim_lat, [lat])])
            wind.attrs[cfg.attrs_units] = "m s-1"

            # Calculate wind components.
            uas, vas = utils.sfcwind_2_uas_vas(wind, windfromdir)

            if var == cfg.var_cordex_uas:
                data_xarray = uas
            else:
                data_xarray = vas
            da = xr.DataArray(data_xarray, coords=[(cfg.dim_time, time), (cfg.dim_lon, [lon]), (cfg.dim_lat, [lat])])
            da.name = var
            if var == cfg.var_cordex_uas:
                da.attrs[cfg.attrs_sname] = "eastward_wind"
                da.attrs[cfg.attrs_lname] = "Eastward near-surface wind"
            else:
                da.attrs[cfg.attrs_sname] = "northward_wind"
                da.attrs[cfg.attrs_lname] = "Northward near-surface wind"
            da.attrs[cfg.attrs_units] = "m s-1"
            da.attrs[cfg.attrs_gmap]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        # Temperature or precipitation ---------------------------------------------------------------------------------

        elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmax, cfg.var_cordex_tasmin, cfg.var_cordex_pr]:

            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])

            # Precipitation (involves converting from mm to kg m-2 s-1.
            if var == cfg.var_cordex_pr:

                data        = obs[stn].values / cfg.spd
                data_xarray = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
                da          = xr.DataArray(data_xarray,
                                           coords=[(cfg.dim_time, time), (cfg.dim_lon, [lon]), (cfg.dim_lat, [lat])])

                da.name = var
                da.attrs[cfg.attrs_sname] = "precipitation_flux"
                da.attrs[cfg.attrs_lname] = "Precipitation"
                da.attrs[cfg.attrs_units] = "kg m-2 s-1"
                da.attrs[cfg.attrs_gmap]  = "regular_lon_lat"
                da.attrs[cfg.attrs_comments] =\
                    "station data converted from Total Precip (mm) using a density of 1000 kg/m³"

            # Temperature.
            else:

                data        = obs[stn].values
                data_xarray = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
                da          = xr.DataArray(data_xarray,
                                           coords=[(cfg.dim_time, time), (cfg.dim_lon, [lon]), (cfg.dim_lat, [lat])])

                da.name = var
                da.attrs[cfg.attrs_sname] = "temperature"
                da.attrs[cfg.attrs_lname] = "temperature"
                da.attrs[cfg.attrs_units] = "degree_C"
                da.attrs[cfg.attrs_gmap]  = "regular_lon_lat"
                da.attrs[cfg.attrs_comments] = "station data converted from degree C"

            # Create dataset.
            ds = da.to_dataset()

        else:
            ds = None

        # Wind, temperature or precipitation ---------------------------------------------------------------------------

        # Add attributes to lon, lat, time, elevation, and the grid.
        da      = xr.DataArray(np.full(len(time), np.nan), [(cfg.dim_time, time)])
        da.name = "regular_lon_lat"
        da.attrs[cfg.attrs_gmapname] = "lonlat"

        # Create dataset and add attributes.
        ds["regular_lon_lat"] = da
        ds.lon.attrs[cfg.attrs_sname] = "longitude"
        ds.lon.attrs[cfg.attrs_lname] = "longitude"
        ds.lon.attrs[cfg.attrs_units] = "degrees_east"
        ds.lon.attrs[cfg.attrs_axis]  = "X"
        ds.lat.attrs[cfg.attrs_sname] = "latitude"
        ds.lat.attrs[cfg.attrs_lname] = "latitude"
        ds.lat.attrs[cfg.attrs_units] = "degrees_north"
        ds.lat.attrs[cfg.attrs_axis]  = "Y"
        ds.attrs[cfg.attrs_stn] = stn
        # ds.attrs["Province"]           = station["province"]
        # ds.attrs["Climate Identifier"] = station["ID"]
        # ds.attrs["WMO Identifier"]     = station["WMO_ID"]
        # ds.attrs["TC Identifier"]      = station["TC_ID"]
        # ds.attrs["Institution"]        = "Environment and Climate Change Canada"

        # Save data.
        p_stn = d_stn + var + "_" + ds.attrs[cfg.attrs_stn] + ".nc"
        desc = "/" + cfg.cat_obs + "/" + os.path.basename(p_stn)
        utils.save_netcdf(ds, p_stn, desc=desc)
        utils.close_netcdf(ds)
        utils.close_netcdf(da)


def load_reanalysis(var_ra):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine NetCDF files into a single file.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the call to the function
    utils.open_netcdf with a list of NetCDF files.

    Parameters
    ----------
    var_ra : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    var = cfg.convert_var_name(var_ra)

    # Paths.
    p_stn_list = list(glob.glob(cfg.d_ra_day + var_ra + "/*.nc"))
    p_stn = cfg.d_stn + var + "/" + var + "_" + cfg.obs_src + ".nc"
    d_stn = os.path.dirname(p_stn)
    if not (os.path.isdir(d_stn)):
        os.makedirs(d_stn)

    if not os.path.exists(p_stn):

        # Combine datasets.
        ds = utils.open_netcdf(p_stn_list, combine='by_coords', concat_dim=cfg.dim_time).copy()
        utils.close_netcdf(ds)

        # Rename variables.
        if var_ra == cfg.var_era5_t2mmin:
            var_ra = cfg.var_era5_t2m
        elif var_ra == cfg.var_era5_t2mmax:
            var_ra = cfg.var_era5_t2m
        ds[var] = ds[var_ra]
        del ds[var_ra]

        # Subset.
        ds = utils.subset_lon_lat(ds)

        # Set attributes.
        ds[var].attrs[cfg.attrs_gmap] = "regular_lon_lat"
        if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
            ds[var].attrs[cfg.attrs_sname] = "temperature"
            ds[var].attrs[cfg.attrs_lname] = "Temperature"
            ds[var].attrs[cfg.attrs_units] = "K"
        elif var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            if var == cfg.var_cordex_pr:
                ds[var].attrs[cfg.attrs_sname] = "precipitation_flux"
                ds[var].attrs[cfg.attrs_lname] = "Precipitation"
            elif var == cfg.var_cordex_evapsbl:
                ds[var].attrs[cfg.attrs_sname] = "evaporation_flux"
                ds[var].attrs[cfg.attrs_lname] = "Evaporation"
            elif var == cfg.var_cordex_evapsblpot:
                ds[var].attrs[cfg.attrs_sname] = "evapotranspiration_flux"
                ds[var].attrs[cfg.attrs_lname] = "Evapotranspiration"
            ds[var].attrs[cfg.attrs_units] = "kg m-2 s-1"
        elif var in [cfg.var_cordex_uas, cfg.var_cordex_vas]:
            if var == cfg.var_cordex_uas:
                ds[var].attrs[cfg.attrs_sname] = "eastward_wind"
                ds[var].attrs[cfg.attrs_lname] = "Eastward near-surface wind"
            else:
                ds[var].attrs[cfg.attrs_sname] = "northward_wind"
                ds[var].attrs[cfg.attrs_lname] = "Northward near-surface wind"
            ds[var].attrs[cfg.attrs_units] = "m s-1"
        elif var == cfg.var_cordex_rsds:
            ds[var].attrs[cfg.attrs_sname] = "surface_solar_radiation_downwards"
            ds[var].attrs[cfg.attrs_lname] = "Surface solar radiation downwards"
            ds[var].attrs[cfg.attrs_units] = "J m-2"
        elif var == cfg.var_cordex_huss:
            ds[var].attrs[cfg.attrs_sname] = "specific_humidity"
            ds[var].attrs[cfg.attrs_lname] = "Specific humidity"
            ds[var].attrs[cfg.attrs_units] = "1"

        # Save data.
        desc = "/" + cfg.cat_obs + "/" + os.path.basename(p_stn)
        utils.save_netcdf(ds, p_stn, desc=desc)


def extract(var, ds_stn, d_ref, d_fut, p_raw):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract data from NetCDF files.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the (indirect) call to the
    function utils.open_netcdf with a list of NetCDF files.

    Parameters
    ----------
    var : str
        Weather variable.
    ds_stn : xr.Dataset
        NetCDF file containing station data.
    d_ref : str
        Directory of NetCDF files containing simulations (reference period).
    d_fut : str
        Directory of NetCDF files containing simulations (future period).
    p_raw : str
        Path of the directory containing raw data.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Directories.
    d_raw = cfg.get_d_sim("", cfg.cat_raw, var)

    # Zone of interest -------------------------------------------------------------------------------------------------

    # Observations.
    if not cfg.opt_ra:

        # Assume a square around the location.
        lat_stn = round(float(ds_stn.lat.values), 1)
        lon_stn = round(float(ds_stn.lon.values), 1)
        lat_bnds = [lat_stn - cfg.radius, lat_stn + cfg.radius]
        lon_bnds = [lon_stn - cfg.radius, lon_stn + cfg.radius]

    # Reanalysis.
    # When using reanalysis data, need to include extra cells in case the resolution of the reanalysis dataset is lower
    # than the resolution of the projection dataset or if meshes are not overlapping perfectly.
    else:

        # Projections.
        # Must use xr.open_dataset here, otherwise there is a problem in parallel mode.
        p_proj = list(glob.glob(d_ref + var + "/*.nc"))[0]
        try:
            ds_proj = xr.open_dataset(p_proj)
        except xcv.MissingDimensionsError:
            ds_proj = xr.open_dataset(p_proj, drop_variables=["time_bnds"])
        res_proj_lat = abs(ds_proj.rlat.values[1] - ds_proj.rlat.values[0])
        res_proj_lon = abs(ds_proj.rlon.values[1] - ds_proj.rlon.values[0])

        # Reanalysis.
        res_ra_lat = abs(ds_stn.latitude.values[1] - ds_stn.latitude.values[0])
        res_ra_lon = abs(ds_stn.longitude.values[1] - ds_stn.longitude.values[0])

        # Calculate the number of extra space.
        n_lat_ext = float(max(1.0, math.ceil(res_proj_lat / res_ra_lat)))
        n_lon_ext = float(max(1.0, math.ceil(res_proj_lon / res_ra_lon)))

        # Calculate extended boundaries.
        lat_bnds = [cfg.lat_bnds[0] - n_lat_ext * res_ra_lat, cfg.lat_bnds[1] + n_lat_ext * res_ra_lat]
        lon_bnds = [cfg.lon_bnds[0] - n_lon_ext * res_ra_lon, cfg.lon_bnds[1] + n_lon_ext * res_ra_lon]

    # Data extraction --------------------------------------------------------------------------------------------------

    utils.log("Extracting data from NetCDF file", True)

    # The idea is to extract historical and projected data based on a range of longitude, latitude, years.
    ds_raw = rcm.extract_variable(d_ref, d_fut, var, lat_bnds, lon_bnds,
                                  priority_timestep=cfg.priority_timestep[cfg.variables_cordex.index(var)],
                                  tmpdir=d_raw)

    # Save NetCDF file (raw).
    desc = "/" + cfg.cat_raw + "/" + os.path.basename(p_raw)
    utils.save_netcdf(ds_raw, p_raw, desc=desc)


def interpolate(var, ds_stn, p_raw, p_regrid):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract data from NetCDF files.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the (indirect) call to the
    function utils.open_netcdf with a list of NetCDF files.
    TODO.MAB: Something could be done to define the search radius as a function of the occurrence (or not) of a pixel
              storm (data anomaly).

    Parameters
    ----------
    var : str
        Weather variable.
    ds_stn : xr.Dataset
        NetCDF file containing station data.
    p_raw : str
        Path of the directory containing raw data.
    p_regrid : str
        Path of the file containing regrid data.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Temporal interpolation -------------------------------------------------------------------------------------------

    utils.log("Loading data from NetCDF file (raw, w/o interpolation)", True)

    # Load dataset.
    ds_raw = utils.open_netcdf(p_raw).copy()
    utils.close_netcdf(ds_raw)

    msg = "Temporal interpolation is "

    # This checks if the data is daily. If it is sub-daily, resample to daily. This can take a while, but should
    # save us computation time later.
    if ds_raw.time.isel(time=[0, 1]).diff(dim=cfg.dim_time).values[0] <\
            np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0]:

        # Interpolate.
        msg = msg + "running"
        utils.log(msg, True)
        ds_raw = ds_raw.resample(time="1D").mean(dim=cfg.dim_time, keep_attrs=True)

        # Save NetCDF file (raw).
        desc = "/" + cfg.cat_raw + "/" + os.path.basename(p_raw)
        utils.save_netcdf(ds_raw, p_raw, desc=desc, std_save=True)
        ds_raw = utils.open_netcdf(p_raw)
    else:
        msg = msg + "not required (data is daily)"
        utils.log(msg, True)

    # Spatial interpolation --------------------------------------------------------------------------------------------

    msg = "Spatial interpolation is "

    if not os.path.exists(p_regrid):

        msg = msg + "running"
        utils.log(msg, True)

        # Method 1: Convert data to a new grid.
        if cfg.opt_ra:

            # Get longitude and latitude values.
            if cfg.dim_lon in ds_stn.dims:
                lon_vals = ds_stn.lon.values.ravel()
                lat_vals = ds_stn.lat.values.ravel()
                lon_shp = ds_stn.lon.shape[0]
                lat_shp = ds_stn.lat.shape[0]
            else:
                lon_vals = ds_stn.longitude.values.ravel()
                lat_vals = ds_stn.latitude.values.ravel()
                lon_shp = ds_stn.longitude.shape[0]
                lat_shp = ds_stn.latitude.shape[0]

            # Create new mesh.
            new_grid = np.meshgrid(lon_vals, lat_vals)
            if np.min(new_grid[0]) > 0:
                new_grid[0] -= 360
            t_len = len(ds_raw.time)
            arr_regrid = np.empty((t_len, lat_shp, lon_shp))
            for t in range(0, t_len):
                arr_regrid[t, :, :] = griddata(
                    (ds_raw.lon.values.ravel(), ds_raw.lat.values.ravel()),
                    ds_raw[var][t, :, :].values.ravel(),
                    (new_grid[0], new_grid[1]),
                    fill_value=np.nan, method="linear")

            # Create data array and dataset.
            if not cfg.opt_ra:
                da_regrid = xr.DataArray(arr_regrid,
                                         coords={cfg.dim_time: ds_raw.time[0:t_len],
                                                 cfg.dim_lat: float(lat_vals),
                                                 cfg.dim_lon: float(lon_vals)},
                                         dims=[cfg.dim_time, cfg.dim_rlat, cfg.dim_rlon], attrs=ds_raw.attrs)
            else:
                da_regrid = xr.DataArray(arr_regrid,
                                         coords=[(cfg.dim_time, ds_raw.time[0:t_len]),
                                                 (cfg.dim_lat, lat_vals),
                                                 (cfg.dim_lon, lon_vals)],
                                         dims=[cfg.dim_time, cfg.dim_rlat, cfg.dim_rlon], attrs=ds_raw.attrs)
            ds_regrid = da_regrid.to_dataset(name=var)
            ds_regrid[var].attrs[cfg.attrs_units] = ds_raw[var].attrs[cfg.attrs_units]

        # Method 2: Take nearest information.
        else:

            # Assume a square around the location.
            lat_stn = round(float(ds_stn.lat.values), 1)
            lon_stn = round(float(ds_stn.lon.values), 1)

            # Determine nearest points.
            ds_regrid = ds_raw.sel(rlat=lat_stn, rlon=lon_stn, method="nearest", tolerance=1)

        # Save NetCDF file (regrid).
        desc = "/" + cfg.cat_regrid + "/" + os.path.basename(p_regrid)
        utils.save_netcdf(ds_regrid, p_regrid, desc=desc)

    else:

        msg = msg + "not required"
        utils.log(msg, True)


def preprocess(var, ds_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs pre-processing.

    Parameters
    ----------
    var : str
        Variable.
    ds_stn : xr.Dataset
        NetCDF file containing station data.
    p_obs : str
        Path of NetCDF file containing observations.
    p_regrid : str
        Path of regrid NetCDF simulation file.
    p_regrid_ref : str
        Path of regrid NetCDF simulation file (reference period).
    p_regrid_fut : str
        Path of regrid NetCDF simulation file (future period).
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_fut = utils.open_netcdf(p_regrid)
    ds_regrid_ref = None
    ds_regrid_fut = None

    def close_netcdf():

        utils.close_netcdf(ds_stn)
        utils.close_netcdf(ds_fut)
        utils.close_netcdf(ds_regrid_ref)
        utils.close_netcdf(ds_regrid_fut)

    # Add small perturbation.
    def perturbate(ds, var_inner, d_val=1e-12):

        if not cfg.opt_ra:
            ds[var_inner].values = ds[var_inner].values + np.random.rand(ds[var_inner].values.shape[0]) * d_val
        else:
            ds[var_inner].values = ds[var_inner].values + np.random.rand(ds[var_inner].values.shape[0], 1, 1) * d_val

        return ds

    # Observations -----------------------------------------------------------------------------------------------------

    if not os.path.exists(p_obs):

        # Interpolate temporally by dropping February 29th and by sub-selecting between reference years.
        ds_obs = ds_stn.sel(time=~((ds_stn.time.dt.month == 2) & (ds_stn.time.dt.day == 29)))
        ds_obs = ds_obs.where(
            (ds_obs.time.dt.year >= cfg.per_ref[0]) & (ds_obs.time.dt.year <= cfg.per_ref[1]), drop=True)

        # Add small perturbation.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            perturbate(ds_obs, var)

        # Save NetCDF file.
        desc = "/" + cfg.cat_obs + "/" + os.path.basename(p_obs)
        utils.save_netcdf(ds_obs, p_obs, desc=desc)

    # Simulated climate (future period) --------------------------------------------------------------------------------

    if os.path.exists(p_regrid_fut):

        ds_regrid_fut = utils.open_netcdf(p_regrid_fut)

    else:

        # Drop February 29th.
        ds_regrid_fut = ds_fut.sel(time=~((ds_fut.time.dt.month == 2) & (ds_fut.time.dt.day == 29)))

        # Adjust values that do not make sense.
        # TODO.YR: Verify if positive or negative values need to be considered for cfg.var_cordex_evapsbl and
        #          cfg.var_cordex_evapsblpot.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot, cfg.var_cordex_clt]:
            ds_regrid_fut[var].values[ds_regrid_fut[var] < 0] = 0
            if var == cfg.var_cordex_clt:
                ds_regrid_fut[var].values[ds_regrid_fut[var] > 100] = 100

        # Add small perturbation.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            perturbate(ds_regrid_fut, var)

        # Convert to a 365-day calendar.
        ds_regrid_fut = utils.convert_to_365_calender(ds_regrid_fut)

        # Save dataset.
        desc = "/" + cfg.cat_regrid + "/" + os.path.basename(p_regrid_fut)
        utils.save_netcdf(ds_regrid_fut, p_regrid_fut, desc=desc)

    # Simulated climate (reference period) -----------------------------------------------------------------------------

    if not os.path.exists(p_regrid_ref):

        ds_regrid_ref = ds_regrid_fut.sel(time=slice(str(cfg.per_ref[0]), str(cfg.per_ref[1])))

        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot, cfg.var_cordex_clt]:
            pos = np.where(np.squeeze(ds_regrid_ref[var].values) > 0.01)[0]
            ds_regrid_ref[var][pos] = 1e-12

        # Save dataset.
        desc = "/" + cfg.cat_regrid + "/" + os.path.basename(p_regrid_ref)
        utils.save_netcdf(ds_regrid_ref, p_regrid_ref, desc=desc)

    close_netcdf()


def postprocess(var, nq, up_qmf, time_win, ds_stn, p_ref, p_fut, p_qqmap, p_qmf, title="", p_fig=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs post-processing.

    Parameters
    ----------
    var : str
        Weather variable.
    nq : int
        ...
    up_qmf : float
        ...
    time_win : int
        ...
    ds_stn : xr.Dataset
        NetCDF file containing station data.
    p_ref : str
        Path of NetCDF file containing simulation data (reference period).
    p_fut : str
        Path of NetCDF file containing simulation data (future period).
    p_qqmap : str
        Path of NetCDF file containing adjusted simulation data.
    p_qmf : str
        Path of NetCDF file containing quantile map function.
    title : str, optional
        Title of figure.
    p_fig : str, optimal
        Path of figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    # The files p_stn and p_ref cannot be opened using xr.open_mfdataset.
    da_stn = ds_stn[var]
    if cfg.dim_longitude in da_stn.dims:
        da_stn = da_stn.rename({cfg.dim_longitude: cfg.dim_rlon, cfg.dim_latitude: cfg.dim_rlat})
    ds_ref = utils.open_netcdf(p_ref)
    ds_fut = utils.open_netcdf(p_fut)
    da_ref = ds_ref[var]
    da_fut = ds_fut[var]
    ds_qmf = None
    ds_qqmap = None

    # The following two commented statements are similar to those in the initial code version They seem to have not
    # effect as a boundary box selection was already made earlier.
    # ds_ref = ds_ref.sel(rlon=slice(min(ds_stn[var].longitude), max(ds_stn[var].longitude)))
    # ds_fut = ds_fut.sel(rlon=slice(min(ds_stn[var].longitude), max(ds_stn[var].longitude)))

    def close_netcdf():

        utils.close_netcdf(da_ref)
        utils.close_netcdf(da_fut)
        utils.close_netcdf(ds_qmf)
        utils.close_netcdf(ds_qqmap)

    # Future -----------------------------------------------------------------------------------------------------------

    # Convert to a 365-day calendar.
    da_fut_365 = utils.convert_to_365_calender(ds_fut)[var]

    # Observation ------------------------------------------------------------------------------------------------------

    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
        kind = cfg.kind_mult
    elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        if not cfg.opt_ra:
            da_stn = da_stn + cfg.d_KC
        kind = cfg.kind_add
    else:
        kind = cfg.kind_add

    # Interpolate.
    if not cfg.opt_ra:
        da_stn = da_stn.interpolate_na(dim=cfg.dim_time)

    # Quantile Mapping Function ----------------------------------------------------------------------------------------

    # Load transfer function.
    if (p_qmf != "") and os.path.exists(p_qmf):
        ds_qmf = utils.open_netcdf(p_qmf)

    # Calculate transfer function.
    else:
        da_qmf = xr.DataArray(train(da_ref.squeeze(), da_stn.squeeze(), nq, cfg.group, kind, time_win,
                                    detrend_order=cfg.detrend_order))
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            da_qmf.values[da_qmf > up_qmf] = up_qmf
        ds_qmf = da_qmf.to_dataset(name=var)
        ds_qmf[var].attrs[cfg.attrs_group] = da_qmf.attrs[cfg.attrs_group]
        ds_qmf[var].attrs[cfg.attrs_kind] = da_qmf.attrs[cfg.attrs_kind]
        if p_qmf != "":
            desc = "/" + cfg.cat_qmf + "/" + os.path.basename(p_qmf)
            utils.save_netcdf(ds_qmf, p_qmf, desc=desc)
            ds_qmf = utils.open_netcdf(p_qmf)

    # Quantile Mapping -------------------------------------------------------------------------------------------------

    # Load quantile mapping.
    if (p_qqmap != "") and os.path.exists(p_qqmap):
        ds_qqmap = utils.open_netcdf(p_qqmap)

    # Apply transfer function.
    else:
        try:
            interp = (True if not cfg.opt_ra else False)
            da_qqmap = xr.DataArray(predict(da_fut_365.squeeze(), ds_qmf[var].squeeze(),
                                            interp=interp, detrend_order=cfg.detrend_order))
            ds_qqmap = da_qqmap.to_dataset(name=var)
            del ds_qqmap[var].attrs[cfg.attrs_bias]
            ds_qqmap[var].attrs[cfg.attrs_sname] = da_stn.attrs[cfg.attrs_sname]
            ds_qqmap[var].attrs[cfg.attrs_lname] = da_stn.attrs[cfg.attrs_lname]
            ds_qqmap[var].attrs[cfg.attrs_units] = da_stn.attrs[cfg.attrs_units]
            ds_qqmap[var].attrs[cfg.attrs_gmap]  = da_stn.attrs[cfg.attrs_gmap]
            if p_qqmap != "":
                desc = "/" + cfg.cat_qqmap + "/" + os.path.basename(p_qqmap)
                utils.save_netcdf(ds_qqmap, p_qqmap, desc=desc)
                ds_qqmap = utils.open_netcdf(p_qqmap)

        except ValueError as err:
            utils.log("Failed to create QQMAP NetCDF file.", True)
            utils.log(format(err), True)
            pass

    # Create plots -----------------------------------------------------------------------------------------------------

    if p_fig != "":

        # Extract QQMap for the reference period.
        ds_qqmap_ref = ds_qqmap.where((ds_qqmap.time.dt.year >= cfg.per_ref[0]) &
                                      (ds_qqmap.time.dt.year <= cfg.per_ref[1]), drop=True)

        # Convert to data arrays.
        da_qqmap_ref = ds_qqmap_ref[var]
        da_qqmap     = ds_qqmap[var]
        da_qmf       = ds_qmf[var]

        # Convert units.
        coef_1 = 1
        coef_2 = 1
        delta = 0
        if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
            delta = -cfg.d_KC
        elif var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            coef_1 = cfg.spd
            coef_2 = 365
        da_stn       = da_stn * coef_1 + delta
        da_ref       = da_ref * coef_1 + delta
        da_fut       = da_fut * coef_1 + delta
        da_qqmap     = da_qqmap * coef_1 + delta
        da_qqmap_ref = da_qqmap_ref * coef_1 + delta
        if not cfg.opt_ra:
            da_qmf = da_qmf + delta
        else:
            da_qmf = da_qmf * coef_2

        # Select center coordinates.
        da_stn_xy       = da_stn
        da_ref_xy       = da_ref
        da_fut_xy       = da_fut
        da_qqmap_ref_xy = da_qqmap_ref
        da_qqmap_xy     = da_qqmap
        da_qmf_xy       = da_qmf
        if cfg.opt_ra:
            da_stn_xy       = utils.subset_center(da_stn_xy)
            da_ref_xy       = utils.subset_center(da_ref_xy)
            da_fut_xy       = utils.subset_center(da_fut_xy)
            da_qqmap_xy     = utils.subset_center(da_qqmap_xy)
            da_qqmap_ref_xy = utils.subset_center(da_qqmap_ref_xy)
            da_qmf_xy       = utils.subset_center(da_qmf_xy)

        # Generate summary plot.
        if cfg.opt_plot:
            plot.plot_calib(da_qmf_xy, da_qqmap_ref_xy, da_stn_xy, da_ref_xy, da_fut_xy, da_qqmap_xy,
                            var, title, p_fig)

        # Generate time series only.
        if cfg.opt_plot:
            plot.plot_calib_ts(da_stn_xy, da_fut_xy, da_qqmap_xy, var, title, p_fig.replace(".png", "_ts.png"))

    close_netcdf()

    return ds_qqmap if (p_fig == "") else None


def generate():

    """
    --------------------------------------------------------------------------------------------------------------------
    Produce climate scenarios.

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

    # Step #2c: Convert observations from CSV to NetCDF files (for stations) or
    #           Combine all years in a single NetCDF file (for reanalysis).
    # This creates one .nc file per variable-station in ~/<country>/<project>/<stn>/obs/<source>/<var>/.
    utils.log("=")
    utils.log("Step #2c  Converting observations from CSV to NetCDF files")
    if not cfg.opt_ra:
        for var in cfg.variables_cordex:
            load_observations(var)
    else:
        for var_ra in cfg.variables_ra:
            load_reanalysis(var_ra)

    # Step #2d: List directories potentially containing CORDEX files (but not necessarily for all selected variables).
    utils.log("=")
    utils.log("Step #2d  Listing directories with CORDEX files")
    list_cordex = utils.list_cordex(cfg.d_proj, cfg.rcps)

    utils.log("=")
    utils.log("Step #3-5 Production of climate scenarios is running")

    # Loop through variables.
    for i_var in range(0, len(cfg.variables_cordex)):
        var = cfg.variables_cordex[i_var]

        # Select file names for observation (or reanalysis).
        p_stn = ""
        if not cfg.opt_ra:
            d_stn = cfg.get_d_stn(var)
            p_stn_list = glob.glob(d_stn + "*.nc")
            p_stn_list.sort()
        else:
            p_stn = cfg.d_stn + var + "/" + var + "_" + cfg.obs_src + ".nc"
            p_stn_list = [p_stn]

        # Loop through stations.
        for i_stn in range(0, len(p_stn_list)):

            # Station name.
            if not cfg.opt_ra:
                stn = os.path.basename(p_stn).replace(".nc", "").replace(var + "_", "")
                if not (stn in cfg.stns):
                    continue
            else:
                stn = cfg.obs_src

            # Directories.
            d_obs    = cfg.get_d_sim(stn, cfg.cat_obs, var)
            d_raw    = cfg.get_d_sim(stn, cfg.cat_scen + "/" + cfg.cat_raw, var)
            d_regrid = cfg.get_d_sim(stn, cfg.cat_scen + "/" + cfg.cat_regrid, var)
            d_qqmap  = cfg.get_d_sim(stn, cfg.cat_scen + "/" + cfg.cat_qqmap, var)
            d_qmf    = cfg.get_d_sim(stn, cfg.cat_scen + "/" + cfg.cat_qmf, var)
            d_fig_calibration = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cfg.cat_fig_calibration, var)
            d_fig_postprocess = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cfg.cat_fig_postprocess, var)
            d_fig_workflow    = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cfg.cat_fig_workflow, var)

            # Load station data.
            # This needs to be done to avoid competing processes (in parallel mode).
            p_stn = p_stn_list[i_stn]
            ds_stn = utils.open_netcdf(p_stn)

            # Create directories (required because of parallel processing).
            if not (os.path.isdir(d_obs)):
                os.makedirs(d_obs)
            if not (os.path.isdir(d_raw)):
                os.makedirs(d_raw)
            if not (os.path.isdir(d_regrid)):
                os.makedirs(d_regrid)
            if not (os.path.isdir(d_qmf)):
                os.makedirs(d_qmf)
            if not (os.path.isdir(d_qqmap)):
                os.makedirs(d_qqmap)
            if not (os.path.isdir(d_fig_calibration)):
                os.makedirs(d_fig_calibration)
            if not (os.path.isdir(d_fig_postprocess)):
                os.makedirs(d_fig_postprocess)
            if not (os.path.isdir(d_fig_workflow)):
                os.makedirs(d_fig_workflow)

            # Loop through RCPs.
            for rcp in cfg.rcps:

                # Extract and sort simulations lists.
                list_cordex_ref = list_cordex[rcp + "_historical"]
                list_cordex_fut = list_cordex[rcp]
                list_cordex_ref.sort()
                list_cordex_fut.sort()
                n_sim = len(list_cordex_ref)

                # Perform extraction.
                # A first call to generate_single is required for the extraction to be done in scalar mode (before
                # forking) because of the incompatibility of xr.open_mfdataset with parallel processing.
                if cfg.n_proc > 1:
                    for i_sim in range(n_sim):
                        generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, True, i_sim)

                # Scalar mode.
                if cfg.n_proc == 1:
                    for i_sim in range(n_sim):
                        generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, False, i_sim)

                # Parallel processing mode.
                else:

                    # Loop until all simulations have been processed.
                    while True:

                        # Calculate the number of simulations processed (before generation).
                        # This quick verification is based on the QQMAP NetCDF file, but there are several other
                        # files that are generated. The 'completeness' verification is more complete in scalar mode.
                        n_sim_proc_before = len(list(glob.glob(d_qqmap + "*.nc")))

                        # Scalar processing mode.
                        if cfg.n_proc == 1:
                            for i_sim in range(n_sim):
                                generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, False,
                                                i_sim)

                        # Parallel processing mode.
                        else:

                            try:
                                utils.log("Processing: '" + var + "', '" + stn + "', '" + rcp + "'", True)
                                utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                                pool = multiprocessing.Pool(processes=cfg.n_proc)
                                func = functools.partial(generate_single, list_cordex_ref, list_cordex_fut, ds_stn,
                                                         d_raw, var, stn, rcp, False)
                                pool.map(func, list(range(n_sim)))
                                pool.close()
                                pool.join()
                                utils.log("Fork ended.", True)
                            except Exception as e:
                                utils.log(str(e))
                                pass

                        # Calculate the number of simulations processed (after generation).
                        n_sim_proc_after = len(list(glob.glob(d_qqmap + "*.nc")))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_sim_proc_before == n_sim_proc_after):
                            break


def generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, extract_only, i_sim_proc):

    """
    --------------------------------------------------------------------------------------------------------------------
    Produce a single climate scenario.
    Caution: This function is only compatible with scalar processing (n_proc=1) when extract_only is False owing to the
    (indirect) call to the function utils.open_netcdf with a list of NetCDF files.

    Parameters
    ----------
    list_cordex_ref : [str]
        List of CORDEX files for the reference period.
    list_cordex_fut : [str]
        List of CORDEX files for the future period.
    ds_stn : xr.Dataset
        Station file.
    d_raw : str
        Directory containing raw NetCDF files.
    var : str
        Variable.
    stn : str
        Station.
    rcp : str
        RCP emission scenario.
    extract_only:
        If True, only extract.
    i_sim_proc : int
        Rank of simulation to process (in ascending order of raw NetCDF file names).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Directories containing simulations.
    d_sim_ref = list_cordex_ref[i_sim_proc]
    d_sim_fut = list_cordex_fut[i_sim_proc]

    # Get simulation name.
    c = d_sim_fut.split("/")
    sim_name = c[cfg.get_rank_inst()] + "_" + c[cfg.get_rank_gcm()]

    utils.log("=")
    utils.log("Variable   : " + var)
    utils.log("Station    : " + stn)
    utils.log("RCP        : " + rcp)
    utils.log("Simulation : " + sim_name)
    utils.log("=")

    # Skip iteration if the variable 'var' is not available in the current directory.
    p_sim_ref_list = list(glob.glob(d_sim_ref + "/" + var + "/*.nc"))
    p_sim_fut_list = list(glob.glob(d_sim_fut + "/" + var + "/*.nc"))

    if (len(p_sim_ref_list) == 0) or (len(p_sim_fut_list) == 0):
        utils.log("Skipping iteration: data not available for simulation-variable.", True)
        return

    # Files within CORDEX or CORDEX-NA.
    if "cordex" in d_sim_fut.lower():
        p_raw = d_raw + var + "_" + c[cfg.get_rank_inst()] + "_" +\
                c[cfg.get_rank_gcm()].replace("*", "_") + ".nc"
    elif len(d_sim_fut) == 3:
        p_raw = d_raw + var + "_Ouranos_" + d_sim_fut + ".nc"
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
        return

    # Paths and NetCDF files.
    p_regrid     = p_raw.replace(cfg.cat_raw, cfg.cat_regrid)
    p_qqmap      = p_raw.replace(cfg.cat_raw, cfg.cat_qqmap)
    p_qmf        = p_raw.replace(cfg.cat_raw, cfg.cat_qmf)
    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + cfg.cat_qqmap + ".nc"
    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + cfg.cat_qqmap + ".nc"
    p_obs        = cfg.get_p_obs(stn, var)

    # Step #3: Extraction.
    # This step only works in scalar mode.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/
    msg = "Step #3   Extraction is "
    if not os.path.isfile(p_raw):
        utils.log(msg + "running")
        extract(var, ds_stn, d_sim_ref, d_sim_fut, p_raw)
    else:
        utils.log(msg + "not required")

    # When running in parallel mode and only performing extraction, the remaining steps will be done in a second pass.
    if extract_only:
        return()

    # Step #4: Spatial and temporal interpolation.
    # This modifies one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/ and
    #      creates  one .nc file in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
    msg = "Step #4   Spatial and temporal interpolation is "
    if not os.path.isfile(p_regrid):
        utils.log(msg + "running")
        interpolate(var, ds_stn, p_raw, p_regrid)
    else:
        utils.log(msg + "not required")

    # Adjusting the datasets associated with observations and simulated conditions
    # (for the reference and future periods) to ensure that calendar is based on 365 days per year and
    # that values are within boundaries (0-100%).
    # This creates two .nc files in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
    utils.log("-")
    msg = "Step #4.5 Pre-processing is "
    if not(os.path.isfile(p_obs)) or not(os.path.isfile(p_regrid_ref)) or not(os.path.isfile(p_regrid_fut)):
        utils.log(msg + "running")
        preprocess(var, ds_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut)
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
    time_win = float(df_sel["time_win"])
    bias_err = float(df_sel["bias_err"])

    # Display calibration parameters.
    msg = "Selected parameters: nq=" + str(nq) + ", up_qmf=" + str(up_qmf) + \
          ", time_win=" + str(time_win) + ", bias_err=" + str(bias_err)
    utils.log(msg, True)

    # Step #5b: Statistical downscaling.
    # Step #5c: Bias correction.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/qqmap/<var>/.
    utils.log("-")
    msg = "Step #5bc Statistical downscaling and bias adjustment is "
    if not(os.path.isfile(p_qqmap)) or not(os.path.isfile(p_qmf)):
        utils.log(msg + "running")
        postprocess(var, int(nq), up_qmf, int(time_win), ds_stn, p_regrid_ref, p_regrid_fut,
                    p_qqmap, p_qmf)
    else:
        utils.log(msg + "not required")


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Generate climate scenarios.
    msg = "Step #2-5 Production of climate scenarios is "
    if cfg.opt_scen:

        # Uncomment the following line to calibrate for all stations and variables.
        # scen_calib.run()

        # Generate scenarios.
        generate()

    else:

        utils.log("=")
        utils.log(msg + "not required")

    # Generate time series.
    if cfg.opt_plot:

        # --------------------------------------------------------------------------------------------------------------

        utils.log("=")
        utils.log("Generating post-process and workflow diagrams.", True)

        # Loop through variables.
        for var in cfg.variables_cordex:

            # Loop through stations.
            stns = (cfg.stns if not cfg.opt_ra else [cfg.obs_src])
            for stn in stns:

                # Path ofo NetCDF file containing station data.
                p_stn = cfg.d_stn + var + "/" + var + "_" + stn + ".nc"

                # Loop through raw NetCDF files.
                p_raw_list = list(glob.glob(cfg.get_d_sim(stn, cfg.cat_raw, var) + "*.nc"))
                for p_raw in p_raw_list:

                    # Path of NetCDF files.
                    p_regrid     = p_raw.replace(cfg.cat_raw, cfg.cat_regrid)
                    p_qqmap      = p_raw.replace(cfg.cat_raw, cfg.cat_qqmap)
                    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + cfg.cat_qqmap + ".nc"
                    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + cfg.cat_qqmap + ".nc"

                    # Extract simulations name.
                    sim_name = os.path.basename(p_raw).replace(var + "_", "").replace(".nc", "")

                    # Calibration parameters.
                    df_sel = cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                              (cfg.df_calib["stn"] == stn) &
                                              (cfg.df_calib["var"] == var)]
                    nq = float(df_sel["nq"])
                    up_qmf = float(df_sel["up_qmf"])
                    time_win = float(df_sel["time_win"])

                    # This creates one .png file in ~/sim_climat/<country>/<project>/<stn>/fig/postprocess/<var>/.
                    fn_fig = p_regrid_fut.split("/")[-1].replace("_4qqmap.nc", "_" + cfg.cat_fig_postprocess + ".png")
                    title = fn_fig[:-4] + "_nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cfg.cat_fig_postprocess, var) + fn_fig
                    plot.plot_postprocess(p_stn, p_regrid_fut, p_qqmap, var, p_fig, title)

                    # This creates one .png file in ~/sim_climat/<country>/<project>/<stn>/fig/workflow/<var>/.
                    p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/" + cfg.cat_fig_workflow, var) + \
                        p_regrid_fut.split("/")[-1].replace("4qqmap.nc", cfg.cat_fig_workflow + ".png")
                    plot.plot_workflow(var, int(nq), up_qmf, int(time_win), p_regrid_ref, p_regrid_fut, p_fig)

        # --------------------------------------------------------------------------------------------------------------

        utils.log("=")
        utils.log("Generating time series.", True)

        for var in cfg.variables_cordex:
            plot.plot_ts(var)

        # --------------------------------------------------------------------------------------------------------------

        # Perform interpolation (requires multiples stations).
        # Heat maps are not generated:
        # - the result is not good with a limited number of stations;
        # - calculation is very slow (something is wrong).
        if cfg.opt_plot_heat and ((len(cfg.stns) > 1) or cfg.opt_ra):

            utils.log("=")
            utils.log("Generating heat maps.", True)

            for i in range(len(cfg.variables_cordex)):

                # Reference period.
                plot.plot_heatmap(cfg.variables_cordex[i], [], cfg.rcp_ref, [cfg.per_ref])

                # Future period.
                for rcp in cfg.rcps:
                    plot.plot_heatmap(cfg.variables_cordex[i], [], rcp, cfg.per_hors)


if __name__ == "__main__":
    run()
