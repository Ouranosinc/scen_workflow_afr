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
        p_obs = d_stn + var + "_" + ds.attrs[cfg.attrs_stn] + ".nc"
        utils.save_dataset(ds, p_obs)
        ds.close()
        da.close()


def combine_nc(var_ra):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine NetCDF files into a single file.

    Parameters
    ----------
    var_ra : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    var = cfg.convert_var_name(var_ra)

    # Paths.
    p_stn_format = cfg.d_ra_day + var_ra + "/*.nc"
    p_stn = cfg.d_stn + var + "/" + var + "_" + cfg.obs_src + ".nc"

    if not os.path.exists(p_stn):

        # Combine datasets.
        ds = xr.open_mfdataset(p_stn_format, combine='by_coords', concat_dim=cfg.dim_time)

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
        utils.save_dataset(ds, p_stn)


def extract(var, p_stn, d_ref, d_fut, p_raw, p_regrid):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extracts data.
    TODO.MAB: Something could be done to define the search radius as a function of the occurrence (or not) of a pixel
              storm (data anomaly).

    Parameters
    ----------
    var : str
        Weather variable.
    p_stn : str
        Path of the file containing station data.
    d_ref : str
        Directory of simulation files containing historical data.
    d_fut : str
        Directory of simulation files containing projected data.
    p_raw : str
        Path of the directory containing raw data.
    p_regrid : str
        Path of the file containing regrid data.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load observations.
    ds_stn = xr.open_dataset(p_stn)

    # Directories.
    d_raw = cfg.get_d_sim("", cfg.cat_raw, var)

    # Zone of interest -------------------------------------------------------------------------------------------------

    # Observations.
    lat_stn = None
    lon_stn = None
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
        p_proj = list(glob.glob(d_ref + var + "/*.nc"))[0]
        try:
            ds_proj = xr.open_dataset(p_proj)
        except Exception as e:
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

    # Data extraction and temporal interpolation -----------------------------------------------------------------------

    p_raw_generated = False
    if os.path.exists(p_raw):

        utils.log("Loading data from NetCDF file (raw)", True)

        ds_raw = xr.open_dataset(p_raw)

    else:

        utils.log("Extracting data from NetCDF file (raw)", True)

        # The idea is to extract historical and projected data based on a range of longitude, latitude, years.
        ds_raw = rcm.extract_variable(d_ref, d_fut, var, lat_bnds, lon_bnds,
                                  priority_timestep=cfg.priority_timestep[cfg.variables_cordex.index(var)],
                                  tmpdir=d_raw)

        if cfg.opt_scen_itp_time:

            msg = "Temporal interpolation is "

            # This checks if the data is daily. If it is sub-daily, resample to daily. This can take a while, but should
            # save us computation time later.
            if ds_raw.time.isel(time=[0, 1]).diff(dim=cfg.dim_time).values[0] <\
                    np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0]:
                msg = msg + "running"
                utils.log(msg, True)
                ds_raw = ds_raw.resample(time="1D").mean(dim=cfg.dim_time, keep_attrs=True)
            else:
                msg = msg + "not required (data is daily)"
                utils.log(msg, True)

        # Save NetCDF file (raw).
        utils.log("Writing NetCDF file (raw)", True)
        utils.save_dataset(ds_raw, p_raw)
        ds_raw = xr.open_dataset(p_raw)
        p_raw_generated = True

    # Spatial interpolation --------------------------------------------------------------------------------------------

    if cfg.opt_scen_itp_space or p_raw_generated:

        msg = "Spatial interpolation is "

        # Method 1: Convert data to a new grid.
        if cfg.opt_scen_regrid:

            msg = msg + "running"
            utils.log(msg, True)

            # Get longitude and latitude values.
            if not cfg.opt_ra:
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

            msg = msg + "not required"
            utils.log(msg, True)

            ds_regrid = ds_raw.sel(rlat=lat_stn, rlon=lon_stn, method="nearest", tolerance=1)

        # Save NetCDF file (regrid).
        utils.log("Writing NetCDF file (regrid)", True)
        utils.save_dataset(ds_regrid, p_regrid)


def preprocess(var, p_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs pre-processing.

    Parameters
    ----------
    var : str
        Variable.
    p_stn : str
        Path of file containing station data.
    p_obs : str
        Path of file containing observation data.
    p_regrid : str
        Path of directory containing regrid data.
    p_regrid_ref : str
        Path of directory containing regrid data for the reference period.
    p_regrid_fut : str
        Path of directory containing regrid data for the future period.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_stn = xr.open_dataset(p_stn)
    ds_fut = xr.open_dataset(p_regrid)

    # Add small perturbation.
    def perturbate(ds, var, d_val=1e-12):

        if not cfg.opt_ra:
            ds[var].values = ds[var].values + np.random.rand(ds[var].values.shape[0]) * d_val
        else:
            ds[var].values = ds[var].values + np.random.rand(ds[var].values.shape[0], 1, 1) * d_val

        return ds

    # Observations -----------------------------------------------------------------------------------------------------

    if not os.path.exists(p_obs):

        # Interpolate temporally by dropping February 29th and by sub-selecting between reference years.
        ds_stn = ds_stn.sel(time=~((ds_stn.time.dt.month == 2) & (ds_stn.time.dt.day == 29)))
        ds_stn = ds_stn.where(
            (ds_stn.time.dt.year >= cfg.per_ref[0]) & (ds_stn.time.dt.year <= cfg.per_ref[1]), drop=True)

        # Add small perturbation.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            perturbate(ds_stn, var)

        # Save NetCDF file.
        utils.save_dataset(ds_stn, p_obs)

    # Simulated climate (future period) --------------------------------------------------------------------------------

    if os.path.exists(p_regrid_fut):

        ds_fut_365 = xr.open_dataset(p_regrid_fut)

    else:

        # Drop February 29th.
        ds_fut = ds_fut.sel(time=~((ds_fut.time.dt.month == 2) & (ds_fut.time.dt.day == 29)))

        # Adjust values that do not make sense.
        # TODO.YR: Verify if positive or negative values need to be considered for cfg.var_cordex_evapsbl and
        #       cfg.var_cordex_evapsblpot.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot, cfg.var_cordex_clt]:
            ds_fut[var].values[ds_fut[var] < 0] = 0
            if var == cfg.var_cordex_clt:
                ds_fut[var].values[ds_fut[var] > 100] = 100

        # Add small perturbation.
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            perturbate(ds_fut, var)

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
    # DEBUG: if cfg.opt_plt_365vs360:
    # DEBUG:     plot.plot_360_vs_365(ds_fut, ds_fut_365, var)

    # Simulated climate (reference period) -----------------------------------------------------------------------------

    if not os.path.exists(p_regrid_ref):

        ds_ref = ds_fut_365.sel(time=slice(str(cfg.per_ref[0]), str(cfg.per_ref[1])))

        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot, cfg.var_cordex_clt]:
            pos = np.where(np.squeeze(ds_ref[var].values) > 0.01)[0]
            ds_ref[var][pos] = 1e-12

        # Save dataset.
        utils.save_dataset(ds_ref, p_regrid_ref)


def postprocess(var, nq, up_qmf, time_win, p_obs, p_ref, p_fut, p_qqmap, p_qmf, title="", p_fig=""):

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
    p_obs : str
        Path of NetCDF file of observations.
    p_ref : str
        Path of NetCDF file of simulation for the reference period.
    p_fut : str
        Path of NetCDF file of simulation for the future period.
    p_qqmap : str
        Path of NetCDF file of adjusted simulation.
    p_qmf : str
        Path of NetCDF file of quantile map function.
    title : str, optional
        Title of figure.
    p_fig : str, optimal
        Path of figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    ds_obs = xr.open_dataset(p_obs)[var]
    if cfg.dim_longitude in ds_obs.dims:
        ds_obs = ds_obs.rename({cfg.dim_longitude: cfg.dim_rlon, cfg.dim_latitude: cfg.dim_rlat})
    ds_fut = xr.open_dataset(p_fut)[var]
    ds_ref = xr.open_dataset(p_ref)[var]

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

    # DEBUG: Plot 365 versus 360 data.
    # DEBUG: if cfg.opt_plt_365vs360:
    # DEBUG:     plot.plot_360_vs_365(ds_fut, ds_fut_365)

    # Observation ------------------------------------------------------------------------------------------------------

    if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
        kind = cfg.kind_mult
    elif var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
        if not cfg.opt_ra:
            ds_obs = ds_obs + cfg.d_KC
        kind = cfg.kind_add
    else:
        kind = cfg.kind_add

    # Interpolate.
    if not cfg.opt_ra:
        ds_obs = ds_obs.interpolate_na(dim=cfg.dim_time)

    # Quantile Mapping Function ----------------------------------------------------------------------------------------

    # Load transfer function.
    if (p_qmf != "") and os.path.exists(p_qmf):
        ds_qmf = xr.open_dataset(p_qmf)

    # Calculate transfer function.
    else:
        if not cfg.opt_ra:
            ds_qmf = train(ds_ref.squeeze(), ds_obs.squeeze(), nq, cfg.group, kind, time_win,
                           detrend_order=cfg.detrend_order)
        else:
            ds_qmf = train(ds_ref, ds_obs, nq, cfg.group, kind, time_win, detrend_order=cfg.detrend_order)
        if var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            ds_qmf.values[ds_qmf > up_qmf] = up_qmf
        if p_qmf != "":
            utils.save_dataset(ds_qmf, p_qmf)

    # Quantile Mapping -------------------------------------------------------------------------------------------------

    # Load quantile mapping.
    ds_qqmap = None
    if (p_qqmap != "") and os.path.exists(p_qqmap):
        ds_qqmap = xr.open_dataset(p_qqmap)

    # Apply transfer function.
    else:
        try:
            if not cfg.opt_ra:
                ds_qqmap = predict(ds_fut_365.squeeze(), ds_qmf, interp=True, detrend_order=cfg.detrend_order)
            else:
                ds_qqmap = predict(ds_fut_365, ds_qmf, interp=False, detrend_order=cfg.detrend_order)
            del ds_qqmap.attrs[cfg.attrs_bias]
            ds_qqmap.attrs[cfg.attrs_sname] = ds_obs.attrs[cfg.attrs_sname]
            ds_qqmap.attrs[cfg.attrs_lname] = ds_obs.attrs[cfg.attrs_lname]
            ds_qqmap.attrs[cfg.attrs_units] = ds_obs.attrs[cfg.attrs_units]
            ds_qqmap.attrs[cfg.attrs_gmap]  = ds_obs.attrs[cfg.attrs_gmap]
            if p_qqmap != "":
                utils.save_dataset(ds_qqmap, p_qqmap)

        except ValueError as err:
            utils.log("Failed to create QQMAP NetCDF file.", True)
            utils.log(format(err), True)
            pass

    # Create plots -----------------------------------------------------------------------------------------------------

    if p_fig == "":

        return ds_qqmap

    else:

        # Extract QQMap for the reference period.
        ds_qqmap_ref = ds_qqmap.where((ds_qqmap.time.dt.year >= cfg.per_ref[0]) &
                                      (ds_qqmap.time.dt.year <= cfg.per_ref[1]), drop=True)

        # Convert units.
        coef_1 = 1
        coef_2 = 1
        delta = 0
        if var in [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax]:
            delta = -cfg.d_KC
        elif var in [cfg.var_cordex_pr, cfg.var_cordex_evapsbl, cfg.var_cordex_evapsblpot]:
            coef_1 = cfg.spd
            coef_2 = 365
        ds_qmf = ds_qmf * coef_2
        if not cfg.opt_ra:
            ds_qmf = ds_qmf + delta
        ds_qqmap_ref = ds_qqmap_ref * coef_1 + delta
        ds_obs       = ds_obs * coef_1 + delta
        ds_ref       = ds_ref * coef_1 + delta
        ds_fut       = ds_fut * coef_1 + delta
        ds_qqmap     = ds_qqmap * coef_1 + delta

        # Select center coordinates.
        ds_qmf_xy       = ds_qmf
        ds_qqmap_ref_xy = ds_qqmap_ref
        ds_obs_xy       = ds_obs
        ds_ref_xy       = ds_ref
        ds_fut_xy       = ds_fut
        ds_qqmap_xy     = ds_qqmap
        if cfg.opt_ra:
            ds_qmf_xy       = utils.subset_ctr_mass(ds_qmf_xy)
            ds_qqmap_ref_xy = utils.subset_ctr_mass(ds_qqmap_ref_xy)
            ds_obs_xy       = utils.subset_ctr_mass(ds_obs_xy)
            ds_ref_xy       = utils.subset_ctr_mass(ds_ref_xy)
            ds_fut_xy       = utils.subset_ctr_mass(ds_fut_xy)
            ds_qqmap_xy     = utils.subset_ctr_mass(ds_qqmap_xy)

        # Generate summary plot.
        if cfg.opt_plot:
            plot.plot_calib(ds_qmf_xy, ds_qqmap_ref_xy, ds_obs_xy, ds_ref_xy, ds_fut_xy, ds_qqmap_xy,
                            var, title, p_fig)

        # Generate time series only.
        if cfg.opt_plot:
            plot.plot_calib_ts(ds_obs_xy, ds_fut_xy, ds_qqmap_xy, var, title, p_fig.replace(".png", "_ts.png"))


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
    if cfg.opt_scen_load_obs:
        if not cfg.opt_ra:
            for var in cfg.variables_cordex:
                load_observations(var)
        else:
            for var_ra in cfg.variables_ra:
                combine_nc(var_ra)

    # Step #2d: List directories potentially containing CORDEX files (but not necessarily for all selected variables).
    utils.log("=")
    utils.log("Step #2d  Listing directories with CORDEX files")
    list_cordex = utils.list_cordex(cfg.d_proj, cfg.rcps)

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
                stn = os.path.basename(p_stn_list[i_stn]).replace(".nc", "").replace(var + "_", "")
                if not (stn in cfg.stns):
                    continue
            else:
                stn = cfg.obs_src

            # Create directories.
            d_obs    = cfg.get_d_sim(stn, cfg.cat_obs, var)
            d_raw    = cfg.get_d_sim(stn, cfg.cat_raw, var)
            d_fut    = cfg.get_d_sim(stn, cfg.cat_qqmap, var)
            d_regrid = cfg.get_d_sim(stn, cfg.cat_regrid, var)
            d_qqmap  = cfg.get_d_sim(stn, cfg.cat_qqmap, var)

            # Loop through RCPs.
            for rcp in cfg.rcps:

                # Extract and sort simulations lists.
                list_cordex_ref = list_cordex[rcp + "_historical"]
                list_cordex_fut = list_cordex[rcp]
                list_cordex_ref.sort()
                list_cordex_fut.sort()
                n_sim = len(list_cordex_ref)

                if not cfg.opt_ra:
                    p_stn = cfg.get_p_stn(var, stn)

                # Scalar processing mode.
                if cfg.n_proc == 1:

                    for i_sim in range(n_sim):
                        generate_single(list_cordex_ref, list_cordex_fut, p_stn, d_raw, var, stn, rcp, i_sim)

                # Parallel processing mode.
                else:

                    # Loop until all simulations have been processed.
                    n_sim_processed = 0
                    while n_sim_processed < n_sim:

                        # Calculate the number of simulations that were processed.
                        # This quick verification is based on the QQMAP NetCDF file, but there are several other
                        # files that are generated. The 'completeness' verification is more complete in scalar mode.
                        n_sim_processed = len(list(glob.glob(d_qqmap + "*.nc")))
                        if n_sim == n_sim_processed:
                            break

                        # Scalar processing mode.
                        if cfg.n_proc == 1:
                            for i_sim in range(n_sim):
                                generate_single(list_cordex_ref, list_cordex_fut, p_stn, d_raw, var, stn, rcp, i_sim)

                        # Parallel processing mode.
                        else:

                            try:
                                utils.log("Step #3-5 Production of climate scenarios.")
                                utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                                pool = multiprocessing.Pool(processes=cfg.n_proc)
                                func = functools.partial(generate_single, list_cordex_ref, list_cordex_fut, p_stn, d_raw,
                                                         var, stn, rcp)
                                pool.map(func, list(range(n_sim)))
                                pool.close()
                                pool.join()
                                utils.log("Parallel processing ended.", True)
                            except Exception as e:
                                pass


def generate_single(list_cordex_ref, list_cordex_fut, p_stn, d_raw, var, stn, rcp, i_sim_proc):

    """
    --------------------------------------------------------------------------------------------------------------------
    Produce a single climate scenario.

    Parameters
    ----------
    list_cordex_ref : [str]

    list_cordex_fut : [str]
    p_stn : str
        Path of station fille.
    d_raw : str
        Directory containing raw NetCDF files.
    var : str
        Variable.
    stn : str
        Station.
    rcp : str
        RCP emission scenario.
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
    p_sim_ref_list = glob.glob(d_sim_ref + "/" + var + "/*.nc")
    p_sim_fut_list = glob.glob(d_sim_fut + "/" + var + "/*.nc")
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

    # Step #3: Spatial and temporal extraction.
    # Step #4: Grid transfer or interpolation.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/ and
    #              one .nc file in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
    msg = "Step #3-4 Spatial & temporal extraction and grid transfer (or interpolation) is "
    if cfg.opt_scen_extract and (not(os.path.isfile(p_raw)) or not(os.path.isfile(p_regrid))):
        utils.log(msg + "running")
        extract(var, p_stn, d_sim_ref, d_sim_fut, p_raw, p_regrid)
    else:
        utils.log(msg + "not required")

    # Adjusting the datasets associated with observations and simulated conditions
    # (for the reference and future periods) to ensure that calendar is based on 365 days per year and
    # that values are within boundaries (0-100%).
    # This creates two .nc files in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
    utils.log("-")
    msg = "Step #4.5 Pre-processing is "
    if cfg.opt_scen_preprocess and \
       (not(os.path.isfile(p_obs)) or
        not(os.path.isfile(p_regrid_ref)) or
        not(os.path.isfile(p_regrid_fut))):
        utils.log(msg + "running")
        preprocess(var, p_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut)
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
    if cfg.opt_scen_postprocess or not(os.path.isfile(p_qqmap)) or not(os.path.isfile(p_qmf)):
        utils.log(msg + "running")
        postprocess(var, int(nq), up_qmf, int(time_win), p_stn, p_regrid_ref, p_regrid_fut,
                    p_qqmap, p_qmf)
    else:
        utils.log(msg + "not required")

    # Generate plots.
    if cfg.opt_plot:

        # This creates one .png file in ~/sim_climat/<country>/<project>/<stn>/fig/postprocess/<var>/.
        fn_fig = p_regrid_fut.split("/")[-1].replace("_4qqmap.nc", "_postprocess.png")
        title = fn_fig[:-4] + "_nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
        p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/postprocess", var) + fn_fig
        plot.plot_postprocess(p_stn, p_regrid_fut, p_qqmap, var, p_fig, title)

        # This creates one .png file in ~/sim_climat/<country>/<project>/<stn>/fig/workflow/<var>/.
        p_fig = cfg.get_d_sim(stn, cfg.cat_fig + "/workflow", var) +\
            p_regrid_fut.split("/")[-1].replace("4qqmap.nc", "workflow.png")
        plot.plot_workflow(var, int(nq), up_qmf, int(time_win), p_regrid_ref, p_regrid_fut, p_fig)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Generate climate scenarios.
    msg = "Step #3-5 Generation of climate scenarios is "
    if cfg.opt_scen:

        msg = msg + "running"
        utils.log(msg, True)

        # Uncomment the following line to calibrate for all stations and variables.
        # scen_calib.run()

        # Generate scenarios.
        generate()

    else:

        utils.log("=")
        msg = msg + "not required"
        utils.log(msg)

    # Generate time series.
    if cfg.opt_plot:

        utils.log("=")
        utils.log("Generating time series.", True)

        for var in cfg.variables_cordex:
            plot.plot_ts(var)

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
