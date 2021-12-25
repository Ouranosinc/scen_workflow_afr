# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Production of climate scenarios.
#
# This requires installing package SciTools (when xr.DataArray.time.dtype=const.dtype_obj).
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca (current)
# 2. marc-andre.bourgault@ggr.ulaval.ca (second)
# 3. rondeau-genesse.gabriel@ouranos.ca (original)
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import constants as const
import datetime
import file_utils as fu
import functools
import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import plot
import re
import statistics as stats
import utils
import xarray as xr
import xarray.core.variable as xcv
import warnings
from config import cfg
from quantile_mapping import train, predict
from scipy.interpolate import griddata

import sys
sys.path.append("dashboard")
from dashboard import def_stat, def_varidx as vi, def_view


def load_observations(
    var: str
):

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
    p_stn_info = glob.glob(d_stn + ".." + cfg.sep + "*" + fu.f_ext_csv)
    p_stn_l = glob.glob(d_stn + "*" + fu.f_ext_csv)
    p_stn_l.sort()

    # Compile data.
    for i in range(0, len(p_stn_l)):

        stn = os.path.basename(p_stn_l[i]).replace(fu.f_ext_nc, "").split("_")[1]

        if not(stn in cfg.stns):
            continue

        obs  = pd.read_csv(p_stn_l[i], sep=cfg.f_sep)
        time = pd.to_datetime(
            obs["annees"].astype("str") + "-" + obs["mois"].astype("str") + "-" + obs["jours"].astype("str"))

        # Find longitude and latitude.
        lon_lat_data = pd.read_csv(p_stn_info[0], sep=cfg.f_sep)
        lon = float(lon_lat_data[lon_lat_data["station"] == stn][const.dim_lon])
        lat = float(lon_lat_data[lon_lat_data["station"] == stn][const.dim_lat])

        # Temperature --------------------------------------------------------------------------------------------------

        if var in [vi.v_tas, vi.v_tasmax, vi.v_tasmin]:

            # Extract temperature.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values, axis=1), axis=2)

            # Create DataArray.
            da = xr.DataArray(arr, coords=[(const.dim_time, time), (const.dim_lon, [lon]), (const.dim_lat, [lat])])
            da.name = var
            da.attrs[const.attrs_sname] = "temperature"
            da.attrs[const.attrs_lname] = "temperature"
            da.attrs[const.attrs_units] = const.unit_C
            da.attrs[const.attrs_gmap] = "regular_lon_lat"
            da.attrs[const.attrs_comments] = "station data converted from degree C"

            # Create dataset.
            ds = da.to_dataset()

        # Precipitation, evaporation, evapotranspiration ---------------------------------------------------------------

        elif var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:

            # Extract variable and convert from mm to kg m-2 s-1.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values / const.spd, axis=1), axis=2)

            # Create DataArray.
            da = xr.DataArray(arr, coords=[(const.dim_time, time), (const.dim_lon, [lon]), (const.dim_lat, [lat])])
            da.name = var
            da.attrs[const.attrs_sname] = "precipitation_flux"
            da.attrs[const.attrs_lname] = "Precipitation"
            da.attrs[const.attrs_units] = const.unit_kg_m2s1
            da.attrs[const.attrs_gmap] = "regular_lon_lat"
            da.attrs[const.attrs_comments] =\
                "station data converted from Total Precip (mm) using a density of 1000 kg/m³"

            # Create dataset.
            ds = da.to_dataset()

        # Wind ---------------------------------------------------------------------------------------------------------

        elif var in [vi.v_uas, vi.v_vas]:

            # Extract wind direction (dd).
            obs_dd = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("vv", axis=1)), index=time, columns=[stn])
            arr_dd = np.expand_dims(np.expand_dims(obs_dd[stn].values, axis=1), axis=2)
            da_dd = xr.DataArray(arr_dd,
                                 coords=[(const.dim_time, time), (const.dim_lon, [lon]), (const.dim_lat, [lat])])

            # Extract wind velocity (vv).
            obs_vv = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("dd", axis=1)), index=time, columns=[stn])
            arr_vv = np.expand_dims(np.expand_dims(obs_vv[stn].values, axis=1), axis=2)
            da_vv = xr.DataArray(arr_vv,
                                 coords=[(const.dim_time, time), (const.dim_lon, [lon]), (const.dim_lat, [lat])])
            da_vv.attrs[const.attrs_units] = const.unit_m_s

            # Calculate wind components.
            da_uas, da_vas = utils.sfcwind_2_uas_vas(da_vv, da_dd)

            # Create DataArray.
            da = da_uas if var == vi.v_uas else da_vas
            da = xr.DataArray(da, coords=[(const.dim_time, time), (const.dim_lon, [lon]), (const.dim_lat, [lat])])
            da.name = var
            if var == vi.v_uas:
                da.attrs[const.attrs_sname] = "eastward_wind"
                da.attrs[const.attrs_lname] = "Eastward near-surface wind"
            else:
                da.attrs[const.attrs_sname] = "northward_wind"
                da.attrs[const.attrs_lname] = "Northward near-surface wind"
            da.attrs[const.attrs_units] = const.unit_m_s
            da.attrs[const.attrs_gmap]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        elif var == vi.v_sfcwindmax:

            # Extract wind velocity.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values, axis=1), axis=2)
            da = xr.DataArray(arr, coords=[(const.dim_time, time), (const.dim_lon, [lon]), (const.dim_lat, [lat])])

            # Create DataArray.
            da.name = var
            da.attrs[const.attrs_sname] = "wind"
            da.attrs[const.attrs_lname] = "near-surface wind"
            da.attrs[const.attrs_units] = const.unit_m_s
            da.attrs[const.attrs_gmap]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        else:
            ds = None

        # Wind, temperature or precipitation ---------------------------------------------------------------------------

        # Add attributes to lon, lat, time, elevation, and the grid.
        da      = xr.DataArray(np.full(len(time), np.nan), [(const.dim_time, time)])
        da.name = "regular_lon_lat"
        da.attrs[const.attrs_gmapname] = "lonlat"

        # Create dataset and add attributes.
        ds["regular_lon_lat"] = da
        ds.lon.attrs[const.attrs_sname] = const.dim_longitude
        ds.lon.attrs[const.attrs_lname] = const.dim_longitude
        ds.lon.attrs[const.attrs_units] = "degrees_east"
        ds.lon.attrs[const.attrs_axis]  = "X"
        ds.lat.attrs[const.attrs_sname] = const.dim_latitude
        ds.lat.attrs[const.attrs_lname] = const.dim_latitude
        ds.lat.attrs[const.attrs_units] = "degrees_north"
        ds.lat.attrs[const.attrs_axis]  = "Y"
        ds.attrs[const.attrs_stn] = stn

        # Save data.
        p_stn = d_stn + var + "_" + ds.attrs[const.attrs_stn] + fu.f_ext_nc
        desc = cfg.sep + const.cat_obs + cfg.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def preload_reanalysis(
    var_ra: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine daily NetCDF files (one day per file) into annual single file (all days of one year per file).

    Parameters
    ----------
    var_ra : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List NetCDF files.
    p_l = list(glob.glob(cfg.d_ra_day + var_ra + cfg.sep + "daily" + cfg.sep + "*" + fu.f_ext_nc))

    # Determine which token corresponds to the date (based on the name of the first file).
    id_token = -1
    if len(p_l) > 0:
        tokens = re.split(r"[_|.]", os.path.basename(p_l[0]))
        for i in range(len(tokens)):
            if (len(tokens[i]) == 8) and tokens[i].isnumeric():
                id_token = i
                break
    if id_token == -1:
        fu.log("Unable to locate date within file name.")
        return

    # Extract years.
    year_l = []
    for p in p_l:
        year = int(re.split(r"[_|.]", os.path.basename(p))[id_token][0:4])
        if year not in year_l:
            year_l.append(year)

    # Add time dimension (if not there).
    for p in p_l:
        ds = fu.open_netcdf(p)
        if const.dim_time not in ds.dims:
            date_str = re.split(r"[_|.]", os.path.basename(p))[id_token][0:8]
            time = pd.to_datetime(date_str[0:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
            da_time = xr.DataArray(time)
            ds[const.dim_time] = da_time
            ds = ds.expand_dims(const.dim_time)
            fu.save_netcdf(ds, p)

    # Combine files.
    for year in year_l:

        # Paths.
        p_pattern = cfg.d_ra_day + var_ra + cfg.sep + "daily" + cfg.sep + "*" + str(year) + "*" + fu.f_ext_nc
        p_l = list(glob.glob(p_pattern))
        p_l.sort()
        p = cfg.d_ra_day + var_ra + cfg.sep + var_ra + "_" + cfg.obs_src + "_day_" + str(year) + fu.f_ext_nc

        if (not os.path.exists(p)) or cfg.opt_force_overwrite:

            # Combine NetCDF files.
            ds = fu.open_netcdf(p_l, combine="nested", concat_dim=const.dim_time).load()

            # Rename dimensions
            ds = ds.rename_dims({"Lon": const.dim_longitude, "Lat": const.dim_latitude})
            ds[const.dim_longitude] = ds["Lon"]
            ds[const.dim_latitude] = ds["Lat"]
            ds = ds.drop_vars(["Lon", "Lat"])
            ds[const.dim_longitude].attrs["long_name"] = const.dim_longitude
            ds[const.dim_latitude].attrs["long_name"] = const.dim_latitude
            ds[const.dim_longitude].attrs["units"] = "degrees_east"
            ds[const.dim_latitude].attrs["units"] = "degrees_north"
            if var_ra not in ds.variables:
                if var_ra == vi.v_enacts_rr:
                    da_name = "precip"
                else:
                    da_name = "temp"
                ds[var_ra] = ds[da_name]
                ds = ds.drop_vars([da_name])

            # Adjust units.
            if (var_ra in [vi.v_enacts_tmin, vi.v_enacts_tmin]) and (const.unit_C in ds[var_ra].attrs["units"]):
                ds[var_ra] = ds[var_ra] + 273.0
                ds[var_ra].attrs["units"] = const.unit_K
            elif (var_ra in [vi.v_enacts_rr, vi.v_enacts_pet]) and (const.unit_mm in ds[var_ra].attrs["units"]):
                ds[var_ra] = ds[var_ra] / const.spd
                ds[var_ra].attrs["units"] = const.unit_kg_m2s1

            # Save combined datasets.
            fu.save_netcdf(ds, p, os.path.basename(p))


def load_reanalysis(
    var_ra: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine NetCDF files into a single file.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the call to fu.open_netcdf
    with a list of NetCDF files.

    Parameters
    ----------
    var_ra : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    var = vi.VarIdx(var_ra).convert_name(vi.ens_cordex)

    # Paths.
    p_stn_l = list(glob.glob(cfg.d_ra_day + var_ra + cfg.sep + "*" + fu.f_ext_nc))
    p_stn = cfg.d_stn + var + cfg.sep + var + "_" + cfg.obs_src + fu.f_ext_nc
    d_stn = os.path.dirname(p_stn)
    if not (os.path.isdir(d_stn)):
        os.makedirs(d_stn)

    if (not os.path.exists(p_stn)) or cfg.opt_force_overwrite:

        # Combine datasets (the 'load' is necessary to apply the mask later).
        # ds = fu.open_netcdf(p_stn_l, combine="by_coords", concat_dim=const.dim_time).load()
        ds = fu.open_netcdf(p_stn_l, combine="nested", concat_dim=const.dim_time).load()

        # Rename variables.
        if var_ra in [vi.v_era5_t2mmin, vi.v_era5_t2mmax]:
            var_ra = vi.v_era5_t2m
        elif var_ra in [vi.v_era5_u10min, vi.v_era5_u10max]:
            var_ra = vi.v_era5_u10
        elif var_ra in [vi.v_era5_v10min, vi.v_era5_v10max]:
            var_ra = vi.v_era5_v10
        elif var_ra == vi.v_era5_uv10max:
            var_ra = vi.v_era5_uv10
        ds[var] = ds[var_ra]
        del ds[var_ra]

        # Subset.
        ds = utils.subset_lon_lat_time(ds, var, cfg.lon_bnds, cfg.lat_bnds)

        # Apply and create mask.
        if (cfg.obs_src == vi.ens_era5_land) and (var not in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]):
            da_mask = fu.create_mask()
            da = utils.apply_mask(ds[var], da_mask)
            ds = da.to_dataset(name=var)

        # Set attributes.
        ds[var].attrs[const.attrs_gmap] = "regular_lon_lat"
        if var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
            ds[var].attrs[const.attrs_sname] = "temperature"
            ds[var].attrs[const.attrs_lname] = "Temperature"
            ds[var].attrs[const.attrs_units] = const.unit_K
        elif var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
            if (cfg.obs_src == vi.ens_era5) or (cfg.obs_src == vi.ens_era5_land):
                ds[var] = ds[var] * 1000 / const.spd
            if var == vi.v_pr:
                ds[var].attrs[const.attrs_sname] = "precipitation_flux"
                ds[var].attrs[const.attrs_lname] = "Precipitation"
            elif var == vi.v_evspsbl:
                ds[var].attrs[const.attrs_sname] = "evaporation_flux"
                ds[var].attrs[const.attrs_lname] = "Evaporation"
            elif var == vi.v_evspsblpot:
                ds[var].attrs[const.attrs_sname] = "evapotranspiration_flux"
                ds[var].attrs[const.attrs_lname] = "Evapotranspiration"
            ds[var].attrs[const.attrs_units] = const.unit_kg_m2s1
        elif var in [vi.v_uas, vi.v_vas, vi.v_sfcwindmax]:
            if var == vi.v_uas:
                ds[var].attrs[const.attrs_sname] = "eastward_wind"
                ds[var].attrs[const.attrs_lname] = "Eastward near-surface wind"
            elif var == vi.v_vas:
                ds[var].attrs[const.attrs_sname] = "northward_wind"
                ds[var].attrs[const.attrs_lname] = "Northward near-surface wind"
            else:
                ds[var].attrs[const.attrs_sname] = "wind"
                ds[var].attrs[const.attrs_lname] = "near-surface wind"
            ds[var].attrs[const.attrs_units] = const.unit_m_s
        elif var == vi.v_rsds:
            ds[var].attrs[const.attrs_sname] = "surface_solar_radiation_downwards"
            ds[var].attrs[const.attrs_lname] = "Surface solar radiation downwards"
            ds[var].attrs[const.attrs_units] = const.unit_J_m2
        elif var == vi.v_huss:
            ds[var].attrs[const.attrs_sname] = "specific_humidity"
            ds[var].attrs[const.attrs_lname] = "Specific humidity"
            ds[var].attrs[const.attrs_units] = const.unit_1

        # Change sign to have the same meaning between projections and reanalysis.
        # A positive sign for the following variable means that the transfer direction is from the surface toward the
        # atmosphere. A negative sign means that there is condensation.
        if (var in [vi.v_evspsbl, vi.v_evspsblpot]) and (cfg.obs_src in [vi.ens_era5, vi.ens_era5_land]):
            ds[var] = -ds[var]

        # Save data.
        desc = cfg.sep + const.cat_obs + cfg.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def extract(
    var: str,
    ds_stn: xr.Dataset,
    d_ref: str,
    d_fut: str,
    p_raw: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract data from NetCDF files.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the (indirect) call to
    fu.open_netcdf with a list of NetCDF files.

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

    # Zone of interest -------------------------------------------------------------------------------------------------

    # Observations.
    if not cfg.opt_ra:

        # Assume a square around the location.
        lat_stn = round(float(ds_stn.lat.values), 1)
        lon_stn = round(float(ds_stn.lon.values), 1)
        lat_l = [lat_stn - cfg.radius, lat_stn + cfg.radius]
        lon_l = [lon_stn - cfg.radius, lon_stn + cfg.radius]

    # Reanalysis.
    # When using reanalysis data, need to include extra cells in case the resolution of the reanalysis dataset is lower
    # than the resolution of the projection dataset or if meshes are not overlapping perfectly.
    else:

        # Projections.
        # Must use xr.open_dataset here, otherwise there is a problem in parallel mode.
        p_proj = list(glob.glob(d_ref + var + cfg.sep + "*" + fu.f_ext_nc))[0]
        try:
            ds_proj = xr.open_dataset(p_proj).load()
        except xcv.MissingDimensionsError:
            ds_proj = xr.open_dataset(p_proj, drop_variables=["time_bnds"]).load()
        fu.close_netcdf(ds_proj)
        res_proj_lat = abs(ds_proj.rlat.values[1] - ds_proj.rlat.values[0])
        res_proj_lon = abs(ds_proj.rlon.values[1] - ds_proj.rlon.values[0])

        # Reanalysis.
        res_ra_lat = abs(ds_stn.latitude.values[1] - ds_stn.latitude.values[0])
        res_ra_lon = abs(ds_stn.longitude.values[1] - ds_stn.longitude.values[0])

        # Calculate the number of extra space.
        n_lat_ext = float(max(1.0, math.ceil(res_proj_lat / res_ra_lat)))
        n_lon_ext = float(max(1.0, math.ceil(res_proj_lon / res_ra_lon)))

        # Calculate extended boundaries.
        lat_l = [cfg.lat_bnds[0] - n_lat_ext * res_ra_lat, cfg.lat_bnds[1] + n_lat_ext * res_ra_lat]
        lon_l = [cfg.lon_bnds[0] - n_lon_ext * res_ra_lon, cfg.lon_bnds[1] + n_lon_ext * res_ra_lon]

    # Data extraction --------------------------------------------------------------------------------------------------

    # Patch: Create and discard an empty file to avoid stalling when writing the NetCDF file.
    #        It seems to wake up the hard disk.
    p_inc = p_raw.replace(fu.f_ext_nc, ".incomplete")
    if not os.path.exists(p_inc):
        open(p_inc, 'a').close()
    if os.path.exists(p_inc):
        os.remove(p_inc)

    # List years.
    year_l = [min(min(cfg.per_ref), min(cfg.per_fut)), max(max(cfg.per_ref), max(cfg.per_fut))]

    # Find the data at the requested timestep.
    p_ref = d_ref.replace(cfg.sep + "*" + cfg.sep, cfg.sep + "day" + cfg.sep) + var + cfg.sep + "*" + fu.f_ext_nc
    p_fut = d_fut.replace(cfg.sep + "*" + cfg.sep, cfg.sep + "day" + cfg.sep) + var + cfg.sep + "*" + fu.f_ext_nc
    p_l = sorted(glob.glob(p_ref)) + sorted(glob.glob(p_fut))

    # Open NetCDF.
    # Note: subset is not working with chunks.
    ds = fu.open_netcdf(p_l, chunks={const.dim_time: 365}, drop_variables=["time_vectors", "ts", "time_bnds"],
                        combine="by_coords")

    # Subset.
    ds_subset = utils.subset_lon_lat_time(ds, var, lon_l, lat_l, year_l)

    # Save NetCDF file.
    desc = cfg.sep + const.cat_raw + cfg.sep + os.path.basename(p_raw)
    fu.save_netcdf(ds_subset, p_raw, desc=desc)


def interpolate(
    var: str,
    ds_stn: xr.Dataset,
    p_raw: str,
    p_regrid: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Extract data from NetCDF files.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the (indirect) call to
    fu.open_netcdf with a list of NetCDF files.
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

    fu.log("Loading data from NetCDF file (raw, w/o interpolation)", True)

    # Load dataset.
    ds_raw = fu.open_netcdf(p_raw)

    msg = "Interpolating (time)"

    # This checks if the data is daily. If it is sub-daily, resample to daily. This can take a while, but should
    # save us computation time later.
    if ds_raw.time.isel(time=[0, 1]).diff(dim=const.dim_time).values[0] <\
            np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0]:

        # Interpolate.
        fu.log(msg, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            ds_raw = ds_raw.resample(time="1D").mean(dim=const.dim_time, keep_attrs=True)

        # Save NetCDF file (raw).
        desc = cfg.sep + const.cat_raw + cfg.sep + os.path.basename(p_raw)
        fu.save_netcdf(ds_raw, p_raw, desc=desc)
        ds_raw = fu.open_netcdf(p_raw)
    else:
        fu.log(msg + " (not required; daily frequency)", True)

    # Spatial interpolation --------------------------------------------------------------------------------------------

    msg = "Interpolating (space)"

    if (not os.path.exists(p_regrid)) or cfg.opt_force_overwrite:

        fu.log(msg, True)

        # Method 1: Convert data to a new grid.
        if cfg.opt_ra:

            ds_regrid = regrid(ds_raw, ds_stn, var)

        # Method 2: Take nearest information.
        else:

            # Assume a square around the location.
            lat_stn = round(float(ds_stn.lat.values), 1)
            lon_stn = round(float(ds_stn.lon.values), 1)

            # Determine nearest points.
            ds_regrid = ds_raw.sel(rlat=lat_stn, rlon=lon_stn, method="nearest", tolerance=1)

        # Save NetCDF file (regrid).
        desc = cfg.sep + const.cat_regrid + cfg.sep + os.path.basename(p_regrid)
        fu.save_netcdf(ds_regrid, p_regrid, desc=desc)

    else:

        fu.log(msg + " (not required)", True)


def regrid(
    ds_data: xr.Dataset,
    ds_grid: xr.Dataset,
    var: str
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Perform grid change.
    TODO: Make this function more universal and more efficient.

    Parameters
    ----------
    ds_data: xr.Dataset
        Dataset containing data.
    ds_grid: xr.Dataset
        Dataset containing grid
    var: str
        Climate variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get longitude and latitude values (grid).
    # This is not working.
    # if const.dim_rlon in ds_grid.dims:
    #     grid_lon = ds_grid.rlon.values
    #     grid_lat = ds_grid.rlat.values
    # elif const.dim_lon in ds_grid.variables:
    #     grid_lon = ds_grid.lon.values[1]
    #     grid_lat = ds_grid.lat.values[0]
    # else:
    #     grid_lon = ds_grid.longitude.values
    #     grid_lat = ds_grid.latitude.values
    if const.dim_lon in ds_grid.dims:
        grid_lon = ds_grid.lon.values.ravel()
        grid_lat = ds_grid.lat.values.ravel()
    else:
        grid_lon = ds_grid.longitude.values.ravel()
        grid_lat = ds_grid.latitude.values.ravel()

    # Get longitude and latitude values (data).
    # This is not working.
    # if const.dim_rlon in ds_data.dims:
    #     data_lon = np.array(list(ds_data.rlon.values) * len(ds_data.rlat.values))
    #     data_lat = np.array(list(ds_data.rlat.values) * len(ds_data.rlon.values))
    # elif const.dim_lon in ds_data.variables:
    #     data_lon = ds_data.lon.values.ravel()
    #     data_lat = ds_data.lat.values.ravel()
    # else:
    #     data_lon = ds_data.longitude.values
    #     data_lat = ds_data.latitude.values

    # Create new mesh.
    new_grid = np.meshgrid(grid_lon, grid_lat)
    if np.min(new_grid[0]) > 0:
        new_grid[0] -= 360
    t_len = len(ds_data.time)
    arr_regrid = np.empty((t_len, len(grid_lat), len(grid_lon)))
    for t in range(0, t_len):
        arr_regrid[t, :, :] = griddata(
            (ds_data.lon.values.ravel(), ds_data.lat.values.ravel()),
            ds_data[var][t, :, :].values.ravel(),
            (new_grid[0], new_grid[1]), fill_value=np.nan, method="linear"
        )

    # Create data array.
    # Using xarray v0.20.2, the following line crashes with the following error:
    # {TypeError} Using a DataArray object to construct a variable is ambiguous, please extract the data using the .data
    # property.
    # It runs fine with v0.18.2.
    da_regrid = xr.DataArray(
        arr_regrid,
        coords=[(const.dim_time, ds_data.time[0:t_len]), (const.dim_lat, grid_lat), (const.dim_lon, grid_lon)],
        dims=[const.dim_time, const.dim_rlat, const.dim_rlon],
        attrs=ds_data.attrs
    )

    # Apply and create mask.
    if (cfg.obs_src == vi.ens_era5_land) and (var not in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]):
        ds_regrid = da_regrid.to_dataset(name=var)
        da_mask = fu.create_mask()
        da_regrid = utils.apply_mask(ds_regrid[var], da_mask)

    # Create dataset.
    ds_regrid = da_regrid.to_dataset(name=var)
    ds_regrid[var].attrs[const.attrs_units] = ds_data[var].attrs[const.attrs_units]

    return ds_regrid


def perturbate(
    ds: xr.Dataset,
    var: str
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Add small perturbation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset.
    var: str
        Variable.

    Returns
    -------
    xr.Dataset
        Perturbed dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get perturbation value.
    d_val = 1e-12
    if cfg.opt_calib_perturb is not None:
        for i in range(len(cfg.opt_calib_perturb)):
            if var == cfg.opt_calib_perturb[i][0]:
                d_val = float(cfg.opt_calib_perturb[i][1])
                break

    # Data array has a single dimension.
    if len(ds[var].dims) == 1:
        ds[var].values = ds[var].values + np.random.rand(ds[var].values.shape[0]) * d_val

    # Data array has 3 dimensions, with unknown ranking.
    else:
        vals = ds[var].values.shape[0]
        i_time = list(ds[var].dims).index(const.dim_time)
        ds[var].values = ds[var].values +\
            np.random.rand(vals if i_time == 0 else 1, vals if i_time == 1 else 1, vals if i_time == 2 else 1) * d_val

    return ds


def preprocess(
    var: str,
    ds_stn: xr.Dataset,
    p_obs: str,
    p_regrid: str,
    p_regrid_ref: str,
    p_regrid_fut: str
):

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

    ds_fut = fu.open_netcdf(p_regrid)

    # Observations -----------------------------------------------------------------------------------------------------

    if (not os.path.exists(p_obs)) or cfg.opt_force_overwrite:

        # Drop February 29th and select reference period.
        ds_obs = utils.remove_feb29(ds_stn)
        ds_obs = utils.sel_period(ds_obs, cfg.per_ref)

        # Save NetCDF file.
        desc = cfg.sep + const.cat_obs + cfg.sep + os.path.basename(p_obs)
        fu.save_netcdf(ds_obs, p_obs, desc=desc)

    # Simulated climate (future period) --------------------------------------------------------------------------------

    if os.path.exists(p_regrid_fut) and (not cfg.opt_force_overwrite):

        ds_regrid_fut = fu.open_netcdf(p_regrid_fut)

    else:

        # Drop February 29th.
        ds_regrid_fut = utils.remove_feb29(ds_fut)

        # Adjust values that do not make sense.
        if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot, vi.v_clt]:
            ds_regrid_fut[var].values[ds_regrid_fut[var] < 0] = 0
            if var == vi.v_clt:
                ds_regrid_fut[var].values[ds_regrid_fut[var] > 100] = 100

        # Add small perturbation.
        if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
            perturbate(ds_regrid_fut, var)

        # Convert to a 365-day calendar.
        ds_regrid_fut = utils.convert_to_365_calender(ds_regrid_fut)

        # Save dataset.
        desc = cfg.sep + const.cat_regrid + cfg.sep + os.path.basename(p_regrid_fut)
        fu.save_netcdf(ds_regrid_fut, p_regrid_fut, desc=desc)

    # Simulated climate (reference period) -----------------------------------------------------------------------------

    if (not os.path.exists(p_regrid_ref)) or cfg.opt_force_overwrite:

        # Select reference period.
        ds_regrid_ref = utils.sel_period(ds_regrid_fut, cfg.per_ref)

        if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot, vi.v_clt]:
            pos = np.where(np.squeeze(ds_regrid_ref[var].values) > 0.01)[0]
            ds_regrid_ref[var][pos] = 1e-12

        # Save dataset.
        desc = cfg.sep + const.cat_regrid + cfg.sep + os.path.basename(p_regrid_ref)
        fu.save_netcdf(ds_regrid_ref, p_regrid_ref, desc=desc)


def postprocess(
    var: str,
    nq: int,
    up_qmf: float,
    time_win: int,
    ds_stn: xr.Dataset,
    p_ref: str,
    p_fut: str,
    p_qqmap: str,
    p_qmf: str,
    title: str = "",
    p_fig: str = ""
):

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
    if not cfg.opt_ra:
        if var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
            da_stn_attrs = ds_stn[var].attrs
            ds_stn[var] = ds_stn[var] + const.d_KC
            ds_stn[var].attrs = da_stn_attrs
            ds_stn[var].attrs[const.attrs_units] = const.unit_K
        da_stn = ds_stn[var][:, 0, 0]
    else:
        da_stn = ds_stn[var]
    if const.dim_longitude in da_stn.dims:
        da_stn = da_stn.rename({const.dim_longitude: const.dim_rlon, const.dim_latitude: const.dim_rlat})
    ds_ref = fu.open_netcdf(p_ref)
    ds_fut = fu.open_netcdf(p_fut)
    da_ref = ds_ref[var]
    da_fut = ds_fut[var]
    ds_qqmap = None

    # Future -----------------------------------------------------------------------------------------------------------

    # Convert to a 365-day calendar.
    da_fut_365 = utils.convert_to_365_calender(ds_fut)[var]

    # Observation ------------------------------------------------------------------------------------------------------

    if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
        kind = const.kind_mult
    elif var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
        kind = const.kind_add
    else:
        kind = const.kind_add

    # Interpolate.
    if not cfg.opt_ra:
        da_stn = da_stn.interpolate_na(dim=const.dim_time)

    # Quantile Mapping Function ----------------------------------------------------------------------------------------

    # Load transfer function.
    if (p_qmf != "") and os.path.exists(p_qmf) and (not cfg.opt_force_overwrite):
        chunks = {const.dim_time: 1} if cfg.use_chunks else None
        ds_qmf = fu.open_netcdf(p_qmf, chunks=chunks)

    # Calculate transfer function.
    else:
        da_qmf = xr.DataArray(train(da_ref.squeeze(), da_stn.squeeze(), nq, const.group, kind, time_win,
                                    detrend_order=const.detrend_order))
        if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
            da_qmf.values[da_qmf > up_qmf] = up_qmf
            da_qmf.values[da_qmf < -up_qmf] = -up_qmf
        ds_qmf = da_qmf.to_dataset(name=var)
        ds_qmf[var].attrs[const.attrs_group] = da_qmf.attrs[const.attrs_group]
        ds_qmf[var].attrs[const.attrs_kind] = da_qmf.attrs[const.attrs_kind]
        ds_qmf[var].attrs[const.attrs_units] = da_ref.attrs[const.attrs_units]
        if p_qmf != "":
            desc = cfg.sep + const.cat_qmf + cfg.sep + os.path.basename(p_qmf)
            fu.save_netcdf(ds_qmf, p_qmf, desc=desc)
            ds_qmf = fu.open_netcdf(p_qmf)

    # Quantile Mapping -------------------------------------------------------------------------------------------------

    # Load quantile mapping.
    if (p_qqmap != "") and os.path.exists(p_qqmap) and (not cfg.opt_force_overwrite):
        chunks = {const.dim_time: 1} if cfg.use_chunks else None
        ds_qqmap = fu.open_netcdf(p_qqmap, chunks=chunks)

    # Apply transfer function.
    else:
        try:
            interp = (True if not cfg.opt_ra else False)
            da_qqmap = xr.DataArray(predict(da_fut_365.squeeze(), ds_qmf[var].squeeze(),
                                            interp=interp, detrend_order=const.detrend_order))
            ds_qqmap = da_qqmap.to_dataset(name=var)
            del ds_qqmap[var].attrs[const.attrs_bias]
            ds_qqmap[var].attrs[const.attrs_sname] = da_stn.attrs[const.attrs_sname]
            ds_qqmap[var].attrs[const.attrs_lname] = da_stn.attrs[const.attrs_lname]
            ds_qqmap[var].attrs[const.attrs_units] = da_stn.attrs[const.attrs_units]
            if const.attrs_gmap in da_stn.attrs:
                ds_qqmap[var].attrs[const.attrs_gmap]  = da_stn.attrs[const.attrs_gmap]
            if p_qqmap != "":
                desc = cfg.sep + const.cat_qqmap + cfg.sep + os.path.basename(p_qqmap)
                fu.save_netcdf(ds_qqmap, p_qqmap, desc=desc)
                ds_qqmap = fu.open_netcdf(p_qqmap)

        except ValueError as err:
            fu.log("Failed to create QQMAP NetCDF file.", True)
            fu.log(format(err), True)
            pass

    # Create plots -----------------------------------------------------------------------------------------------------

    if p_fig != "":

        # Select reference period.
        ds_qqmap_ref = utils.sel_period(ds_qqmap, cfg.per_ref)

        # Convert to data arrays.
        da_qqmap_ref = ds_qqmap_ref[var]
        da_qqmap     = ds_qqmap[var]
        da_qmf       = ds_qmf[var]

        def convert_units(da: xr.DataArray, units: str) -> xr.DataArray:
            if (da.units == const.unit_kg_m2s1) and (units == const.unit_mm):
                da = da * const.spd
                da.attrs[const.attrs_units] = units
            elif (da.units == const.unit_K) and (units == const.unit_C):
                da = da - const.d_KC
                da[const.attrs_units] = units
            return da

        # Convert units.
        if var in [vi.v_pr, vi.v_evspsbl, vi.v_evspsblpot]:
            da_stn       = convert_units(da_stn, const.unit_mm)
            da_ref       = convert_units(da_ref, const.unit_mm)
            da_fut       = convert_units(da_fut, const.unit_mm)
            da_qqmap     = convert_units(da_qqmap, const.unit_mm)
            da_qqmap_ref = convert_units(da_qqmap_ref, const.unit_mm)
            da_qmf       = convert_units(da_qmf, const.unit_mm)
        elif var in [vi.v_tas, vi.v_tasmin, vi.v_tasmax]:
            da_stn       = convert_units(da_stn, const.unit_C)
            da_ref       = convert_units(da_ref, const.unit_C)
            da_fut       = convert_units(da_fut, const.unit_C)
            da_qqmap     = convert_units(da_qqmap, const.unit_C)
            da_qqmap_ref = convert_units(da_qqmap_ref, const.unit_C)

        # Select center coordinates.
        da_stn_xy       = da_stn
        da_ref_xy       = da_ref
        da_fut_xy       = da_fut
        da_qqmap_ref_xy = da_qqmap_ref
        da_qqmap_xy     = da_qqmap
        da_qmf_xy       = da_qmf
        if cfg.opt_ra:
            if cfg.p_bounds == "":
                da_stn_xy       = utils.subset_ctrl_pt(da_stn_xy)
                da_ref_xy       = utils.subset_ctrl_pt(da_ref_xy)
                da_fut_xy       = utils.subset_ctrl_pt(da_fut_xy)
                da_qqmap_xy     = utils.subset_ctrl_pt(da_qqmap_xy)
                da_qqmap_ref_xy = utils.subset_ctrl_pt(da_qqmap_ref_xy)
                da_qmf_xy       = utils.subset_ctrl_pt(da_qmf_xy)
            else:
                da_stn_xy       = utils.squeeze_lon_lat(da_stn_xy)
                da_ref_xy       = utils.squeeze_lon_lat(da_ref_xy)
                da_fut_xy       = utils.squeeze_lon_lat(da_fut_xy)
                da_qqmap_xy     = utils.squeeze_lon_lat(da_qqmap_xy)
                da_qqmap_ref_xy = utils.squeeze_lon_lat(da_qqmap_ref_xy)
                da_qmf_xy       = utils.squeeze_lon_lat(da_qmf_xy)

        if cfg.opt_diagnostic[0]:

            # Generate summary plot.
            plot.plot_calib(da_stn_xy, da_ref_xy, da_fut_xy, da_qqmap_xy, da_qqmap_ref_xy, da_qmf_xy,
                            var, title, p_fig)

            # Generate time series only.
            plot.plot_calib_ts(da_stn_xy, da_fut_xy, da_qqmap_xy, var, title,
                               p_fig.replace(fu.f_ext_png, "_ts" + fu.f_ext_png))

    return ds_qqmap if (p_fig == "") else None


def bias_adj(
    stn: str,
    vi_name: str,
    sim_name: str = "",
    calc_err: bool = False
):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias adjustment.

    Parameters
    ----------
    stn : str
        Station name.
    vi_name : str
        Climate variable name.
    sim_name : str
        Simulation name.
    calc_err : bool
        If True, only calculate the error (will not work properly in parallel mode).
    -------------------------------------------------------------------------------------------------------------------
    """

    # List regrid files.
    d_regrid = cfg.get_d_scen(stn, const.cat_regrid, vi_name)
    p_regrid_l = fu.list_files(d_regrid)
    if p_regrid_l is None:
        fu.log("The required files are not available.", True)
        return
    p_regrid_l = [i for i in p_regrid_l if "_4qqmap" not in i]
    p_regrid_l.sort()

    # Loop through simulation sets.
    for i in range(len(p_regrid_l)):
        p_regrid_tokens = p_regrid_l[i].split(cfg.sep)
        sim_name_i = p_regrid_tokens[len(p_regrid_tokens) - 1].replace(vi_name + "_", "").replace(fu.f_ext_nc, "")

        # Skip iteration if it does not correspond to the specified simulation name.
        if (sim_name != "") and (sim_name != sim_name_i):
            continue

        # Best parameter set.
        bias_err_best = -1

        # Loop through combinations of nq values, up_qmf and time_win values.
        for nq in cfg.nq_calib:
            for up_qmf in cfg.up_qmf_calib:
                for time_win in cfg.time_win_calib:

                    # NetCDF files.
                    p_stn        = cfg.get_p_stn(vi_name, stn)
                    p_regrid     = p_regrid_l[i]
                    p_regrid_ref = p_regrid.replace(fu.f_ext_nc, "_ref_4qqmap" + fu.f_ext_nc)
                    p_regrid_fut = p_regrid.replace(fu.f_ext_nc, "_4qqmap" + fu.f_ext_nc)

                    # If there is a single combination of calibration parameters, NetCDF files can be saved, so that
                    # they don't have to be created during post-process.
                    p_qqmap = ""
                    p_qmf   = ""
                    if (len(cfg.nq_calib) == 1) and (len(cfg.up_qmf_calib) == 1) and (len(cfg.time_win_calib) == 1):
                        p_qqmap = p_regrid.replace(const.cat_regrid, const.cat_qqmap)
                        p_qmf   = p_regrid.replace(const.cat_regrid, const.cat_qmf)

                    # Verify if required files exist.
                    msg = "File missing: "
                    if ((not os.path.exists(p_stn)) or (not os.path.exists(p_regrid_ref)) or
                        (not os.path.exists(p_regrid_fut))) and\
                       (not cfg.opt_force_overwrite):
                        if not(os.path.exists(p_stn)):
                            fu.log(msg + p_stn, True)
                        if not(os.path.exists(p_regrid_ref)):
                            fu.log(msg + p_regrid_ref, True)
                        if not(os.path.exists(p_regrid_fut)):
                            fu.log(msg + p_regrid_fut, True)
                        continue

                    # Load station data, drop February 29th, and select reference period.
                    ds_stn = fu.open_netcdf(p_stn)
                    ds_stn = utils.remove_feb29(ds_stn)
                    ds_stn = utils.sel_period(ds_stn, cfg.per_ref)

                    # Path and title of calibration figure and csv file.
                    fn_fig = vi_name + "_" + sim_name_i + "_" + const.cat_fig_calibration + fu.f_ext_png
                    comb = "nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                    title = sim_name_i + "_" + comb
                    d = const.cat_scen + cfg.sep + const.cat_fig + cfg.sep + const.cat_fig_calibration
                    p_fig = cfg.get_d_scen(stn, d, vi_name) + comb + cfg.sep + fn_fig
                    p_fig_csv =\
                        p_fig.replace(cfg.sep + vi_name + cfg.sep, cfg.sep + vi_name + "_" + fu.f_csv + cfg.sep).\
                        replace(fu.f_ext_png, "_" + def_stat.code_mean + fu.f_ext_csv)
                    p_fig_ts = p_fig.replace(fu.f_ext_png, "_ts" + fu.f_ext_png)
                    p_fig_ts_csv =\
                        p_fig_ts.replace(cfg.sep + vi_name + cfg.sep, cfg.sep + vi_name + "_" + fu.f_csv + cfg.sep).\
                        replace(fu.f_ext_png, fu.f_ext_csv)

                    # Bias adjustment ----------------------------------------------------------------------------------

                    if not calc_err:

                        msg = "nq=" + str(nq) + ", up_qmf=" + str(up_qmf) + ", time_win=" + str(time_win)
                        if not(os.path.exists(p_fig)) or \
                           (not(os.path.exists(p_fig_csv)) and cfg.opt_save_csv[0]) or\
                           not(os.path.exists(p_fig_ts)) or \
                           (not (os.path.exists(p_fig_ts_csv)) and cfg.opt_save_csv[0]) or \
                           not(os.path.exists(p_qqmap)) or not(os.path.exists(p_qmf)) or cfg.opt_force_overwrite:
                            fu.log(msg, True)

                            # Calculate QQ and generate calibration plots.
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                postprocess(vi_name, nq, up_qmf, time_win, ds_stn, p_regrid_ref, p_regrid_fut,
                                            p_qqmap, p_qmf, title, p_fig)
                        else:
                            fu.log(msg + " (not required)", True)

                    # Bias error ---------------------------------------------------------------------------------------

                    elif os.path.exists(p_qqmap):

                        # Extract the reference period from the adjusted simulation.
                        ds_qqmap_ref = fu.open_netcdf(p_qqmap)
                        ds_qqmap_ref = utils.remove_feb29(ds_qqmap_ref)
                        ds_qqmap_ref = utils.sel_period(ds_qqmap_ref, cfg.per_ref)

                        # Calculate the error between observations and simulation for the reference period.
                        bias_err_current = float(round(utils.calc_error(ds_stn[vi_name], ds_qqmap_ref[vi_name]), 4))

                        # Set calibration parameters (nq, up_qmf and time_win) and calculate error according to the
                        # selected method.
                        if (not cfg.opt_calib_auto) or\
                           (cfg.opt_calib_auto and ((bias_err_best < 0) or (bias_err_current < bias_err_best))):
                            col_names = ["nq", "up_qmf", "time_win", "bias_err"]
                            col_values = [float(nq), up_qmf, float(time_win), bias_err_current]
                            calib_row = (cfg.df_calib["sim_name"] == sim_name) &\
                                        (cfg.df_calib["stn"] == stn) &\
                                        (cfg.df_calib["var"] == vi_name)
                            cfg.df_calib.loc[calib_row, col_names] = col_values
                            fu.save_csv(cfg.df_calib, cfg.p_calib)


def init_calib_params():

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize calibration parameters.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Simulations, stations and variables.
    sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l = [], [], [], [], [], [], []

    # Function used to build the dataframe.
    def build_df(_sim_name_l: [str], _stn_l: [str], _vi_code_l: [str], _nq_l: [int],
                 _up_qmf_l: [float], _time_win_l: [int], _bias_err_l: [float]) -> pd.DataFrame:

        dict_pd = {
            "sim_name": _sim_name_l,
            "stn": _stn_l,
            "var": _vi_code_l,
            "nq": _nq_l,
            "up_qmf": _up_qmf_l,
            "time_win": _time_win_l,
            "bias_err": _bias_err_l}
        return pd.DataFrame(dict_pd)

    # Attempt loading a calibration file.
    if os.path.exists(cfg.p_calib):
        cfg.df_calib = pd.read_csv(cfg.p_calib)
        if len(cfg.df_calib) > 0:
            sim_name_l = list(cfg.df_calib["sim_name"])
            stn_l      = list(cfg.df_calib["stn"])
            vi_code_l  = list(cfg.df_calib["var"])
            nq_l       = list(cfg.df_calib["nq"])
            up_qmf_l   = list(cfg.df_calib["up_qmf"])
            time_win_l = list(cfg.df_calib["time_win"])
            bias_err_l = list(cfg.df_calib["bias_err"])
        fu.log("Calibration file loaded.", True)

    # List CORDEX files.
    list_cordex = fu.list_cordex(cfg.d_proj, cfg.rcps)

    # Stations.
    stns = cfg.stns
    if cfg.opt_ra:
        stns = [cfg.obs_src]

    # List simulation names, stations and variables.
    for i_rcp in range(len(cfg.rcps)):
        rcp = cfg.rcps[i_rcp]
        sim_l = list_cordex[rcp]
        sim_l.sort()
        for i_sim in range(0, len(sim_l)):
            list_i = list_cordex[rcp][i_sim].split(cfg.sep)
            sim_name = list_i[cfg.get_rank_inst()] + "_" + list_i[cfg.get_rank_inst() + 1]
            for stn in stns:
                for vi_name in cfg.variables:

                    # Add the combination if it does not already exist.
                    if (cfg.df_calib is None) or\
                       (len(cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                             (cfg.df_calib["stn"] == stn) &
                                             (cfg.df_calib["var"] == vi_name)]) == 0):
                        sim_name_l.append(sim_name)
                        stn_l.append(stn)
                        vi_code_l.append(vi_name)
                        nq_l.append(cfg.nq_default)
                        up_qmf_l.append(cfg.up_qmf_default)
                        time_win_l.append(cfg.time_win_default)
                        bias_err_l.append(cfg.bias_err_default)

                        # Update dataframe.
                        cfg.df_calib = build_df(sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Build dataframe.
    cfg.df_calib = build_df(sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Save calibration parameters to a CSV file.
    if cfg.p_calib != "":

        # Create directory if it does not already exist.
        d = os.path.dirname(cfg.p_calib)
        if not (os.path.isdir(d)):
            os.makedirs(d)

        # Save file.
        fu.save_csv(cfg.df_calib, cfg.p_calib)
        if os.path.exists(cfg.p_calib):
            fu.log("Calibration file created or updated.", True)


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
    d_exec = cfg.get_d_scen("", "", "")
    if not(os.path.isdir(d_exec)):
        os.makedirs(d_exec)

    # Step #2: Data selection.

    # Step #2c: Convert observations from CSV to NetCDF files (for stations) or
    #           Combine all years in a single NetCDF file (for reanalysis).
    # This creates one .nc file per variable-station in ~/<country>/<project>/<stn>/obs/<source>/<var>/.
    fu.log("=")

    if not cfg.opt_ra:
        fu.log("Step #2c  Converting observations from CSV to NetCDF files")
        for var in cfg.variables:
            load_observations(var)
    else:
        fu.log("Step #2c  Merging reanalysis NetCDF files.")
        for var_ra in cfg.variables_ra:
            if cfg.obs_src == vi.ens_enacts:
                preload_reanalysis(var_ra)
            load_reanalysis(var_ra)

    # Step #2d: List directories potentially containing CORDEX files (but not necessarily for all selected variables).
    fu.log("=")
    fu.log("Step #2d  Listing directories with CORDEX files")
    list_cordex = fu.list_cordex(cfg.d_proj, cfg.rcps)

    fu.log("=")
    fu.log("Step #3-5 Producing climate scenarios")

    # Loop through variables.
    for i_var in range(0, len(cfg.variables)):
        var = cfg.variables[i_var]

        # Select file names for observation (or reanalysis).
        if not cfg.opt_ra:
            d_stn = cfg.get_d_stn(var)
            p_stn_l = glob.glob(d_stn + "*" + fu.f_ext_nc)
            p_stn_l.sort()
        else:
            p_stn_l = [cfg.d_stn + var + cfg.sep + var + "_" + cfg.obs_src + fu.f_ext_nc]

        # Loop through stations.
        for i_stn in range(0, len(p_stn_l)):

            # Station name.
            p_stn = p_stn_l[i_stn]
            if not cfg.opt_ra:
                stn = os.path.basename(p_stn).replace(fu.f_ext_nc, "").replace(var + "_", "")
                if not (stn in cfg.stns):
                    continue
            else:
                stn = cfg.obs_src

            # Directories.
            d_stn    = cfg.get_d_stn(var)
            d_obs    = cfg.get_d_scen(stn, const.cat_obs, var)
            d_raw    = cfg.get_d_scen(stn, const.cat_scen + cfg.sep + const.cat_raw, var)
            d_regrid = cfg.get_d_scen(stn, const.cat_scen + cfg.sep + const.cat_regrid, var)
            d_qqmap  = cfg.get_d_scen(stn, const.cat_scen + cfg.sep + const.cat_qqmap, var)
            d_qmf    = cfg.get_d_scen(stn, const.cat_scen + cfg.sep + const.cat_qmf, var)
            d_fig_calibration = cfg.get_d_scen(stn, const.cat_fig + cfg.sep + const.cat_fig_calibration, var)
            d_fig_postprocess = cfg.get_d_scen(stn, const.cat_fig + cfg.sep + const.cat_fig_postprocess, var)
            d_fig_workflow    = cfg.get_d_scen(stn, const.cat_fig + cfg.sep + const.cat_fig_workflow, var)

            # Load station data, drop February 29th and select reference period.
            ds_stn = fu.open_netcdf(p_stn)
            ds_stn = utils.remove_feb29(ds_stn)
            ds_stn = utils.sel_period(ds_stn, cfg.per_ref)

            # Create directories (required because of parallel processing).
            if not (os.path.isdir(d_stn)):
                os.makedirs(d_stn)
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

                fu.log("Processing: " + var + ", " + stn + ", " + rcp, True)

                # Scalar mode.
                if cfg.n_proc == 1:
                    for i_sim in range(n_sim):
                        generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, False, i_sim)

                # Parallel processing mode.
                else:

                    # Perform extraction.
                    # A first call to generate_single is required for the extraction to be done in scalar mode (before
                    # forking) because of the incompatibility of xr.open_mfdataset with parallel processing.
                    for i_sim in range(n_sim):
                        generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, True, i_sim)

                    # Loop until all simulations have been processed.
                    while True:

                        # Calculate the number of processed files (before generation).
                        # This quick verification is based on the QQMAP NetCDF file, but there are several other
                        # files that are generated. The 'completeness' verification is more complete in scalar mode.
                        n_sim_proc_before = len(list(glob.glob(d_qqmap + "*" + fu.f_ext_nc)))

                        # Scalar processing mode.
                        if cfg.n_proc == 1:
                            for i_sim in range(n_sim):
                                generate_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, False,
                                                i_sim)

                        # Parallel processing mode.
                        else:

                            try:
                                fu.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                                pool = multiprocessing.Pool(processes=min(cfg.n_proc, len(list_cordex_ref)))
                                func = functools.partial(generate_single, list_cordex_ref, list_cordex_fut, ds_stn,
                                                         d_raw, var, stn, rcp, False)
                                pool.map(func, list(range(n_sim)))
                                pool.close()
                                pool.join()
                                fu.log("Fork ended.", True)
                            except Exception as e:
                                fu.log(str(e))
                                pass

                        # Calculate the number of processed files (after generation).
                        n_sim_proc_after = len(list(glob.glob(d_qqmap + "*" + fu.f_ext_nc)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cfg.n_proc == 1) or (n_sim_proc_before == n_sim_proc_after):
                            break

                # Calculate bias adjustment errors.
                for i_sim in range(n_sim):
                    tokens = list_cordex_fut[i_sim].split(cfg.sep)
                    sim_name = tokens[cfg.get_rank_inst()] + "_" + tokens[cfg.get_rank_gcm()]
                    bias_adj(stn, var, sim_name, True)


def generate_single(
    list_cordex_ref: [str],
    list_cordex_fut: [str],
    ds_stn: xr.Dataset,
    d_raw: str,
    var: str,
    stn: str,
    rcp: str,
    extract_only: bool,
    i_sim_proc: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Produce a single climate scenario.
    Caution: This function is only compatible with scalar processing (n_proc=1) when extract_only is False owing to the
    (indirect) call to fu.open_netcdf with a list of NetCDF files.

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
    tokens = d_sim_fut.split(cfg.sep)
    sim_name = tokens[cfg.get_rank_inst()] + "_" + tokens[cfg.get_rank_gcm()]

    fu.log("=")
    fu.log("Variable   : " + var)
    fu.log("Station    : " + stn)
    fu.log("RCP        : " + rcp)
    fu.log("Simulation : " + sim_name)
    fu.log("=")

    # Skip iteration if the variable 'var' is not available in the current directory.
    p_sim_ref_l = list(glob.glob(d_sim_ref + cfg.sep + var + cfg.sep + "*" + fu.f_ext_nc))
    p_sim_fut_l = list(glob.glob(d_sim_fut + cfg.sep + var + cfg.sep + "*" + fu.f_ext_nc))

    if (len(p_sim_ref_l) == 0) or (len(p_sim_fut_l) == 0):
        fu.log("Skipping iteration: data not available for simulation-variable.", True)
        if cfg.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Files within CORDEX or CORDEX-NA.
    if "cordex" in d_sim_fut.lower():
        p_raw = d_raw + var + "_" + tokens[cfg.get_rank_inst()] + "_" +\
                tokens[cfg.get_rank_gcm()].replace("*", "_") + fu.f_ext_nc
    elif len(d_sim_fut) == 3:
        p_raw = d_raw + var + "_Ouranos_" + d_sim_fut + fu.f_ext_nc
    else:
        p_raw = None

    # Skip iteration if the simulation or simulation-variable is in the exception list.
    is_sim_except = False
    for sim_except in cfg.sim_excepts:
        if sim_except in p_raw:
            is_sim_except = True
            fu.log("Skipping iteration: simulation-variable exception.", True)
            break
    is_var_sim_except = False
    for var_sim_except in cfg.var_sim_excepts:
        if var_sim_except in p_raw:
            is_var_sim_except = True
            fu.log("Skipping iteration: simulation exception.", True)
            break
    if is_sim_except or is_var_sim_except:
        if cfg.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Paths and NetCDF files.
    p_regrid     = p_raw.replace(const.cat_raw, const.cat_regrid)
    p_qqmap      = p_raw.replace(const.cat_raw, const.cat_qqmap)
    p_qmf        = p_raw.replace(const.cat_raw, const.cat_qmf)
    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + const.cat_qqmap + fu.f_ext_nc
    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + const.cat_qqmap + fu.f_ext_nc
    p_obs        = cfg.get_p_obs(stn, var)

    # Step #3: Extraction.
    # This step only works in scalar mode.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/
    msg = "Step #3   Extracting projections"
    if (not os.path.isfile(p_raw)) or cfg.opt_force_overwrite:
        fu.log(msg)
        extract(var, ds_stn, d_sim_ref, d_sim_fut, p_raw)
    else:
        fu.log(msg + " (not required)")

    # When running in parallel mode and only performing extraction, the remaining steps will be done in a second pass.
    if extract_only:
        return

    # Step #4: Spatial and temporal interpolation.
    # This modifies one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/ and
    #      creates  one .nc file in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
    msg = "Step #4   Interpolating (space and time)"
    if (not os.path.isfile(p_regrid)) or cfg.opt_force_overwrite:
        fu.log(msg)
        interpolate(var, ds_stn, p_raw, p_regrid)
    else:
        fu.log(msg + " (not required)")

    # Adjusting the datasets associated with observations and simulated conditions
    # (for the reference and future periods) to ensure that calendar is based on 365 days per year and
    # that values are within boundaries (0-100%).
    # This creates two .nc files in ~/sim_climat/<country>/<project>/<stn>/regrid/<var>/.
    fu.log("-")
    msg = "Step #4.5 Pre-processing"
    if (not(os.path.isfile(p_obs)) or not(os.path.isfile(p_regrid_ref)) or not(os.path.isfile(p_regrid_fut))) or\
       cfg.opt_force_overwrite:
        fu.log(msg)
        preprocess(var, ds_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut)
    else:
        fu.log(msg + " (not required)")

    # Step #5: Post-processing.

    # Step #5a: Calculate adjustment factors.
    fu.log("-")
    msg = "Step #5a  Calculating adjustment factors"
    if cfg.opt_calib:
        fu.log(msg)
        bias_adj(stn, var, sim_name)
    else:
        fu.log(msg + " (not required)")
    df_sel = cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                              (cfg.df_calib["stn"] == stn) &
                              (cfg.df_calib["var"] == var)]
    if df_sel is not None:
        nq       = float(df_sel["nq"])
        up_qmf   = float(df_sel["up_qmf"])
        time_win = float(df_sel["time_win"])
        bias_err = float(df_sel["bias_err"])
    else:
        nq       = float(cfg.nq_default)
        up_qmf   = float(cfg.up_qmf_default)
        time_win = float(cfg.time_win_default)
        bias_err = float(cfg.bias_err_default)

    # Display calibration parameters.
    msg = "Selected parameters: nq=" + str(nq) + ", up_qmf=" + str(up_qmf) + \
          ", time_win=" + str(time_win) + ", bias_err=" + str(bias_err)
    fu.log(msg, True)

    # Step #5b: Statistical downscaling.
    # Step #5c: Bias correction.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/qqmap/<var>/.
    fu.log("-")
    msg = "Step #5bc Statistical downscaling and adjusting bias"
    if (not(os.path.isfile(p_qqmap)) or not(os.path.isfile(p_qmf))) or cfg.opt_force_overwrite:
        fu.log(msg)
        postprocess(var, int(nq), up_qmf, int(time_win), ds_stn, p_regrid_ref, p_regrid_fut, p_qqmap, p_qmf)
    else:
        fu.log(msg + " (not required)")

    if cfg.n_proc > 1:
        fu.log("Work done!", True)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    not_req = " (not required)"

    # Scenarios --------------------------------------------------------------------------------------------------------

    # Generate climate scenarios.
    msg = "Step #2-5 Calculating scenarios"
    if cfg.opt_scen:
        fu.log(msg)
        generate()
    else:
        fu.log(msg + not_req)

    # Statistics -------------------------------------------------------------------------------------------------------

    fu.log("=")
    msg = "Step #7   Exporting results (scenarios)"
    if cfg.opt_stat[0] or cfg.opt_save_csv[0]:
        fu.log(msg)
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #7a  Calculating statistics (scenarios)"
    if cfg.opt_stat[0]:
        fu.log(msg)
        stats.calc_stats(const.cat_scen)
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #7b  Converting NetCDF to CSV files (scenarios)"
    if cfg.opt_save_csv[0] and not cfg.opt_ra:
        fu.log(msg)
        fu.log("-")
        stats.conv_nc_csv(const.cat_scen)
    else:
        fu.log(msg + not_req)

    # Plots ------------------------------------------------------------------------------------------------------------

    fu.log("=")
    msg = "Step #8   Generating plots and maps (scenarios)"
    if cfg.opt_diagnostic[0] or cfg.opt_map[0]:
        fu.log(msg)
    else:
        fu.log(msg + not_req)

    # Generate diagnostic and bias plots (daily and monthly).
    if cfg.opt_diagnostic[0] or cfg.opt_cycle[0]:

        fu.log("-")
        fu.log("Step #8a  Generating post-process, workflow, daily and monthly plots (scenarios)")

        # Loop through variables.
        for var in cfg.variables:

            # Loop through stations.
            stns = (cfg.stns if not cfg.opt_ra else [cfg.obs_src])
            for stn in stns:

                fu.log("Processing: " + var + ", " + stn, True)

                # Path ofo NetCDF file containing station data.
                p_obs = cfg.get_p_obs(stn, var)

                # Loop through raw NetCDF files.
                p_raw_l = list(glob.glob(cfg.get_d_scen(stn, const.cat_raw, var) + "*" + fu.f_ext_nc))
                for i in range(len(p_raw_l)):
                    p_raw = p_raw_l[i]

                    # Path of NetCDF files.
                    p_regrid     = p_raw.replace(const.cat_raw, const.cat_regrid)
                    p_qqmap      = p_raw.replace(const.cat_raw, const.cat_qqmap)
                    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + const.cat_qqmap + fu.f_ext_nc
                    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + const.cat_qqmap + fu.f_ext_nc

                    # Calibration parameters.
                    sim_name = os.path.basename(p_raw).replace(var + "_", "").replace(fu.f_ext_nc, "")
                    df_sel = cfg.df_calib.loc[(cfg.df_calib["sim_name"] == sim_name) &
                                              (cfg.df_calib["stn"] == stn) &
                                              (cfg.df_calib["var"] == var)]
                    nq = float(df_sel["nq"])
                    up_qmf = float(df_sel["up_qmf"])
                    time_win = float(df_sel["time_win"])

                    # File name.
                    fn_fig = p_regrid_fut.split(cfg.sep)[-1]. \
                        replace("_4qqmap" + fu.f_ext_nc, "_" + const.cat_fig_postprocess + fu.f_ext_png)

                    # Generate diagnostic plots.
                    if cfg.opt_diagnostic[0]:

                        # This creates one file:
                        #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/postprocess/<var>/*.png
                        title = fn_fig[:-4] + "_nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                        d = const.cat_fig + cfg.sep + const.cat_scen + cfg.sep + const.cat_fig_postprocess
                        p_fig = cfg.get_d_scen(stn, d, var) + fn_fig
                        if (not os.path.exists(p_fig)) or cfg.opt_force_overwrite:
                            plot.plot_postprocess(p_obs, p_regrid_fut, p_qqmap, var, p_fig, title)

                        # This creates one file:
                        #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/workflow/<var>/*.png
                        d = const.cat_fig + cfg.sep + const.cat_scen + cfg.sep + const.cat_fig_workflow
                        p_fig = cfg.get_d_scen(stn, d, var) + \
                            p_regrid_fut.split(cfg.sep)[-1].replace("4qqmap" + fu.f_ext_nc,
                                                                    const.cat_fig_workflow + fu.f_ext_png)
                        if (not os.path.exists(p_fig)) or cfg.opt_force_overwrite:
                            plot.plot_workflow(var, int(nq), up_qmf, int(time_win), p_regrid_ref, p_regrid_fut, p_fig)

                    # Generate monthly and daily plots.
                    if cfg.opt_cycle[0]:

                        ds_qqmap = fu.open_netcdf(p_qqmap)
                        for per in cfg.per_hors:
                            per_str = str(per[0]) + "_" + str(per[1])

                            # This creates 2 files:
                            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>/*.png
                            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>_csv/*.csv
                            title = fn_fig[:-4].replace(const.cat_fig_postprocess, per_str + "_" +
                                                        const.cat_fig_cycle_ms)
                            stats.calc_cycle(ds_qqmap, stn, var, per, const.freq_MS, title)

                            # This creates 2 files:
                            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>/*.png
                            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>_csv/*.csv
                            title = fn_fig[:-4].replace(const.cat_fig_postprocess, per_str + "_" +
                                                        const.cat_fig_cycle_d)
                            stats.calc_cycle(ds_qqmap, stn, var, per, const.freq_D, title)

                if os.path.exists(p_obs) and cfg.opt_cycle[0]:

                    ds_obs = fu.open_netcdf(p_obs)
                    per_str = str(cfg.per_ref[0]) + "_" + str(cfg.per_ref[1])

                    # This creates 2 files:
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>/*.png
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>_csv/*.csv
                    title = var + "_" + per_str + "_" + const.cat_fig_cycle_ms
                    stats.calc_cycle(ds_obs, stn, var, cfg.per_ref, const.freq_MS, title)

                    # This creates 2 files:
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>/*.png
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen_cycle_d/<var>_csv/*.csv
                    title = var + "_" + per_str + "_" + const.cat_fig_cycle_d
                    stats.calc_cycle(ds_obs, stn, var, cfg.per_ref, const.freq_D, title)

    fu.log("-")
    msg = "Step #8b  Generating time series (scenarios, bias-adjusted values)"
    if cfg.opt_ts[0]:
        fu.log(msg)
        stats.calc_ts(const.cat_scen, def_view.code_ts)
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #8c  Generating time series (scenarios, raw values and bias)"
    if cfg.opt_ts_bias[0]:
        fu.log(msg)
        stats.calc_ts(const.cat_scen, def_view.code_ts_bias)
    else:
        fu.log(msg + not_req)

    # Heat maps --------------------------------------------------------------------------------------------------------

    # Generate maps.
    # Heat maps are not generated from data at stations:
    # - the result is not good with a limited number of stations;
    # - calculation is very slow (something is wrong).
    fu.log("-")
    msg = "Step #8d  Generating heat maps (scenarios)"
    if cfg.opt_ra and cfg.opt_map[0]:
        fu.log(msg)

        # Loop through variables.
        for i in range(len(cfg.variables)):

            # Generate maps.
            stats.calc_map(cfg.variables[i])

    else:
        fu.log(msg + " (not required)")


def run_calib():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through combinations of stations and variables.
    for stn in cfg.stns:
        for vi_code in cfg.variables:

            fu.log("-", True)
            fu.log("Station  : " + stn, True)
            fu.log("Variable : " + vi_code, True)
            fu.log("-", True)

            # Perform bias correction.
            bias_adj(stn, vi_code)


if __name__ == "__main__":
    run()
