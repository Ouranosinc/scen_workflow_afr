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
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import datetime
import functools
import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import re
import sys
import xarray as xr
import xarray.core.variable as xcv
import warnings
from scipy.interpolate import griddata
from typing import List, Optional

# Workflow libraries.
import file_utils as fu
import plot
import statistics as stats
import utils
from def_constant import const as c
from def_context import cntx
from quantile_mapping import train, predict

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import def_varidx as vi


def load_observations(
    var: vi.VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Load obervations.

    Parameters
    ----------
    var: vi.VarIdx
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Station list file and station files.
    d_stn = cntx.d_stn(var.name)
    p_stn_info = glob.glob(d_stn + ".." + cntx.sep + "*" + c.f_ext_csv)
    p_stn_l = glob.glob(d_stn + "*" + c.f_ext_csv)
    p_stn_l.sort()

    # Compile data.
    for i in range(0, len(p_stn_l)):

        stn = os.path.basename(p_stn_l[i]).replace(c.f_ext_nc, "").split("_")[1]

        if not(stn in cntx.stns):
            continue

        obs  = pd.read_csv(p_stn_l[i], sep=cntx.f_sep)
        time = pd.to_datetime(
            obs["annees"].astype("str") + "-" + obs["mois"].astype("str") + "-" + obs["jours"].astype("str"))

        # Find longitude and latitude.
        lon_lat_data = pd.read_csv(p_stn_info[0], sep=cntx.f_sep)
        lon = float(lon_lat_data[lon_lat_data["station"] == stn][c.dim_lon])
        lat = float(lon_lat_data[lon_lat_data["station"] == stn][c.dim_lat])

        # Temperature --------------------------------------------------------------------------------------------------

        if var.name in [c.v_tas, c.v_tasmax, c.v_tasmin]:

            # Extract temperature.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values, axis=1), axis=2)

            # Create DataArray.
            da = xr.DataArray(arr, coords=[(c.dim_time, time), (c.dim_lon, [lon]), (c.dim_lat, [lat])])
            da.name = var.name
            da.attrs[c.attrs_sname] = "temperature"
            da.attrs[c.attrs_lname] = "temperature"
            da.attrs[c.attrs_units] = c.unit_C
            da.attrs[c.attrs_gmap] = "regular_lon_lat"
            da.attrs[c.attrs_comments] = "station data converted from degree C"

            # Create dataset.
            ds = da.to_dataset()

        # Precipitation, evaporation, evapotranspiration ---------------------------------------------------------------

        elif var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:

            # Extract variable and convert from mm to kg m-2 s-1.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values / c.spd, axis=1), axis=2)

            # Create DataArray.
            da = xr.DataArray(arr, coords=[(c.dim_time, time), (c.dim_lon, [lon]), (c.dim_lat, [lat])])
            da.name = var.name
            da.attrs[c.attrs_sname] = "precipitation_flux"
            da.attrs[c.attrs_lname] = "Precipitation"
            da.attrs[c.attrs_units] = c.unit_kg_m2s1
            da.attrs[c.attrs_gmap] = "regular_lon_lat"
            da.attrs[c.attrs_comments] =\
                "station data converted from Total Precip (mm) using a density of 1000 kg/m³"

            # Create dataset.
            ds = da.to_dataset()

        # Wind ---------------------------------------------------------------------------------------------------------

        elif var.name in [c.v_uas, c.v_vas]:

            # Extract wind direction (dd).
            obs_dd = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("vv", axis=1)), index=time, columns=[stn])
            arr_dd = np.expand_dims(np.expand_dims(obs_dd[stn].values, axis=1), axis=2)
            da_dd = xr.DataArray(arr_dd, coords=[(c.dim_time, time), (c.dim_lon, [lon]), (c.dim_lat, [lat])])

            # Extract wind velocity (vv).
            obs_vv = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("dd", axis=1)), index=time, columns=[stn])
            arr_vv = np.expand_dims(np.expand_dims(obs_vv[stn].values, axis=1), axis=2)
            da_vv = xr.DataArray(arr_vv, coords=[(c.dim_time, time), (c.dim_lon, [lon]), (c.dim_lat, [lat])])
            da_vv.attrs[c.attrs_units] = c.unit_m_s

            # Calculate wind components.
            da_uas, da_vas = utils.sfcwind_2_uas_vas(da_vv, da_dd)

            # Create DataArray.
            da = da_uas if var.name == c.v_uas else da_vas
            da = xr.DataArray(da, coords=[(c.dim_time, time), (c.dim_lon, [lon]), (c.dim_lat, [lat])])
            da.name = var.name
            if var.name == c.v_uas:
                da.attrs[c.attrs_sname] = "eastward_wind"
                da.attrs[c.attrs_lname] = "Eastward near-surface wind"
            else:
                da.attrs[c.attrs_sname] = "northward_wind"
                da.attrs[c.attrs_lname] = "Northward near-surface wind"
            da.attrs[c.attrs_units] = c.unit_m_s
            da.attrs[c.attrs_gmap]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        elif var.name == c.v_sfcwindmax:

            # Extract wind velocity.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values, axis=1), axis=2)
            da = xr.DataArray(arr, coords=[(c.dim_time, time), (c.dim_lon, [lon]), (c.dim_lat, [lat])])

            # Create DataArray.
            da.name = var.name
            da.attrs[c.attrs_sname] = "wind"
            da.attrs[c.attrs_lname] = "near-surface wind"
            da.attrs[c.attrs_units] = c.unit_m_s
            da.attrs[c.attrs_gmap]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        else:
            ds = None

        # Wind, temperature or precipitation ---------------------------------------------------------------------------

        # Add attributes to lon, lat, time, elevation, and the grid.
        da      = xr.DataArray(np.full(len(time), np.nan), [(c.dim_time, time)])
        da.name = "regular_lon_lat"
        da.attrs[c.attrs_gmapname] = "lonlat"

        # Create dataset and add attributes.
        ds["regular_lon_lat"] = da
        ds.lon.attrs[c.attrs_sname] = c.dim_longitude
        ds.lon.attrs[c.attrs_lname] = c.dim_longitude
        ds.lon.attrs[c.attrs_units] = "degrees_east"
        ds.lon.attrs[c.attrs_axis]  = "X"
        ds.lat.attrs[c.attrs_sname] = c.dim_latitude
        ds.lat.attrs[c.attrs_lname] = c.dim_latitude
        ds.lat.attrs[c.attrs_units] = "degrees_north"
        ds.lat.attrs[c.attrs_axis]  = "Y"
        ds.attrs[c.attrs_stn] = stn

        # Save data.
        p_stn = d_stn + var.name + "_" + ds.attrs[c.attrs_stn] + c.f_ext_nc
        desc = cntx.sep + c.cat_obs + cntx.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def preload_reanalysis(
    var_ra: vi.VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine daily NetCDF files (one day per file) into annual single file (all days of one year per file).

    Parameters
    ----------
    var_ra: vi.VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    # List NetCDF files.
    p_l = list(glob.glob(cntx.d_ra_day + var_ra.name + cntx.sep + "daily" + cntx.sep + "*" + c.f_ext_nc))

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
        if c.dim_time not in ds.dims:
            date_str = re.split(r"[_|.]", os.path.basename(p))[id_token][0:8]
            time = pd.to_datetime(date_str[0:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
            da_time = xr.DataArray(time)
            ds[c.dim_time] = da_time
            ds = ds.expand_dims(c.dim_time)
            fu.save_netcdf(ds, p)

    # Combine files.
    for year in year_l:

        # Paths.
        p_pattern = cntx.d_ra_day + var_ra.name + cntx.sep + "daily" + cntx.sep + "*" + str(year) + "*" + c.f_ext_nc
        p_l = list(glob.glob(p_pattern))
        p_l.sort()
        p = cntx.d_ra_day + var_ra.name + cntx.sep + var_ra.name + "_" + cntx.obs_src + "_day_" + str(year) + c.f_ext_nc

        if (not os.path.exists(p)) or cntx.opt_force_overwrite:

            # Combine NetCDF files.
            ds = fu.open_netcdf(p_l, combine="nested", concat_dim=c.dim_time).load()

            # Rename dimensions
            ds = ds.rename_dims({"Lon": c.dim_longitude, "Lat": c.dim_latitude})
            ds[c.dim_longitude] = ds["Lon"]
            ds[c.dim_latitude] = ds["Lat"]
            ds = ds.drop_vars(["Lon", "Lat"])
            ds[c.dim_longitude].attrs["long_name"] = c.dim_longitude
            ds[c.dim_latitude].attrs["long_name"] = c.dim_latitude
            ds[c.dim_longitude].attrs["units"] = "degrees_east"
            ds[c.dim_latitude].attrs["units"] = "degrees_north"
            if var_ra.name not in ds.variables:
                if var_ra.name == c.v_enacts_rr:
                    da_name = "precip"
                else:
                    da_name = "temp"
                ds[var_ra.name] = ds[da_name]
                ds = ds.drop_vars([da_name])

            # Adjust units.
            if (var_ra.name in [c.v_enacts_tmin, c.v_enacts_tmin]) and (c.unit_C in ds[var_ra.name].attrs["units"]):
                ds[var_ra.name] = ds[var_ra.name] + 273.0
                ds[var_ra.name].attrs["units"] = c.unit_K
            elif (var_ra.name in [c.v_enacts_rr, c.v_enacts_pet]) and (c.unit_mm in ds[var_ra.name].attrs["units"]):
                ds[var_ra.name] = ds[var_ra.name] / c.spd
                ds[var_ra.name].attrs["units"] = c.unit_kg_m2s1

            # Save combined datasets.
            fu.save_netcdf(ds, p, os.path.basename(p))


def load_reanalysis(
    var_ra: vi.VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine NetCDF files into a single file.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the call to fu.open_netcdf
    with a list of NetCDF files.

    Parameters
    ----------
    var_ra: vi.VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    var_name = vi.VarIdx(var_ra.name).convert_name(c.ens_cordex)

    # Paths.
    p_stn_l = list(glob.glob(cntx.d_ra_day + var_ra.name + cntx.sep + "*" + c.f_ext_nc))
    p_stn = cntx.d_stn(var_name) + var_name + "_" + cntx.obs_src + c.f_ext_nc
    d_stn = os.path.dirname(p_stn)
    if not (os.path.isdir(d_stn)):
        os.makedirs(d_stn)

    if (not os.path.exists(p_stn)) or cntx.opt_force_overwrite:

        # Combine datasets (the 'load' is necessary to apply the mask later).
        # ds = fu.open_netcdf(p_stn_l, combine="by_coords", concat_dim=const.dim_time).load()
        ds = fu.open_netcdf(p_stn_l, combine="nested", concat_dim=c.dim_time).load()

        # Rename variables.
        if var_ra.name in [c.v_era5_t2mmin, c.v_era5_t2mmax]:
            var_ra_name = c.v_era5_t2m
        elif var_ra.name in [c.v_era5_u10min, c.v_era5_u10max]:
            var_ra_name = c.v_era5_u10
        elif var_ra.name in [c.v_era5_v10min, c.v_era5_v10max]:
            var_ra_name = c.v_era5_v10
        elif var_ra.name == c.v_era5_uv10max:
            var_ra_name = c.v_era5_uv10
        else:
            var_ra_name = var_ra.name
        if var_ra_name != var_ra.name:
            ds[var_name] = ds[var_ra_name]
            del ds[var_ra.name]

        # Subset.
        ds = utils.subset_lon_lat_time(ds, var_name, cntx.lon_bnds, cntx.lat_bnds)

        # Apply and create mask.
        if (cntx.obs_src == c.ens_era5_land) and (var_name not in [c.v_tas, c.v_tasmin, c.v_tasmax]):
            da_mask = fu.create_mask()
            da = utils.apply_mask(ds[var_name], da_mask)
            ds = da.to_dataset(name=var_name)

        # Set attributes.
        ds[var_name].attrs[c.attrs_gmap] = "regular_lon_lat"
        if var_name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
            ds[var_name].attrs[c.attrs_sname] = "temperature"
            ds[var_name].attrs[c.attrs_lname] = "Temperature"
            ds[var_name].attrs[c.attrs_units] = c.unit_K
        elif var_name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            if (cntx.obs_src == c.ens_era5) or (cntx.obs_src == c.ens_era5_land):
                ds[var_name] = ds[var_name] * 1000 / c.spd
            if var_name == c.v_pr:
                ds[var_name].attrs[c.attrs_sname] = "precipitation_flux"
                ds[var_name].attrs[c.attrs_lname] = "Precipitation"
            elif var_name == c.v_evspsbl:
                ds[var_name].attrs[c.attrs_sname] = "evaporation_flux"
                ds[var_name].attrs[c.attrs_lname] = "Evaporation"
            elif var_name == c.v_evspsblpot:
                ds[var_name].attrs[c.attrs_sname] = "evapotranspiration_flux"
                ds[var_name].attrs[c.attrs_lname] = "Evapotranspiration"
            ds[var_name].attrs[c.attrs_units] = c.unit_kg_m2s1
        elif var_name in [c.v_uas, c.v_vas, c.v_sfcwindmax]:
            if var_name == c.v_uas:
                ds[var_name].attrs[c.attrs_sname] = "eastward_wind"
                ds[var_name].attrs[c.attrs_lname] = "Eastward near-surface wind"
            elif var_name == c.v_vas:
                ds[var_name].attrs[c.attrs_sname] = "northward_wind"
                ds[var_name].attrs[c.attrs_lname] = "Northward near-surface wind"
            else:
                ds[var_name].attrs[c.attrs_sname] = "wind"
                ds[var_name].attrs[c.attrs_lname] = "near-surface wind"
            ds[var_name].attrs[c.attrs_units] = c.unit_m_s
        elif var_name == c.v_rsds:
            ds[var_name].attrs[c.attrs_sname] = "surface_solar_radiation_downwards"
            ds[var_name].attrs[c.attrs_lname] = "Surface solar radiation downwards"
            ds[var_name].attrs[c.attrs_units] = c.unit_J_m2
        elif var_name == c.v_huss:
            ds[var_name].attrs[c.attrs_sname] = "specific_humidity"
            ds[var_name].attrs[c.attrs_lname] = "Specific humidity"
            ds[var_name].attrs[c.attrs_units] = c.unit_1

        # Change sign to have the same meaning between projections and reanalysis.
        # A positive sign for the following variable means that the transfer direction is from the surface toward the
        # atmosphere. A negative sign means that there is condensation.
        if (var_name in [c.v_evspsbl, c.v_evspsblpot]) and (cntx.obs_src in [c.ens_era5, c.ens_era5_land]):
            ds[var_name] = -ds[var_name]

        # Save data.
        desc = cntx.sep + c.cat_obs + cntx.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def extract(
    var: vi.VarIdx,
    ds_stn: xr.Dataset,
    d_obs: str,
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
    var: vi.VarIdx
        Variable.
    ds_stn: xr.Dataset
        NetCDF file containing station data.
    d_obs: str
        Directory of NetCDF files containing observations (reference period).
    d_fut: str
        Directory of NetCDF files containing simulations (future period).
    p_raw: str
        Path of the directory containing raw data.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Zone of interest -------------------------------------------------------------------------------------------------

    # Observations.
    if not cntx.opt_ra:

        # Assume a square around the location.
        lat_stn = round(float(ds_stn.lat.values), 1)
        lon_stn = round(float(ds_stn.lon.values), 1)
        lat_l = [lat_stn - cntx.radius, lat_stn + cntx.radius]
        lon_l = [lon_stn - cntx.radius, lon_stn + cntx.radius]

    # Reanalysis.
    # When using reanalysis data, need to include extra cells in case the resolution of the reanalysis dataset is lower
    # than the resolution of the projection dataset or if meshes are not overlapping perfectly.
    else:

        # Projections.
        # Must use xr.open_dataset here, otherwise there is a problem in parallel mode.
        p_proj = list(glob.glob(d_obs + var.name + cntx.sep + "*" + c.f_ext_nc))[0]
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
        lat_l = [cntx.lat_bnds[0] - n_lat_ext * res_ra_lat, cntx.lat_bnds[1] + n_lat_ext * res_ra_lat]
        lon_l = [cntx.lon_bnds[0] - n_lon_ext * res_ra_lon, cntx.lon_bnds[1] + n_lon_ext * res_ra_lon]

    # Data extraction --------------------------------------------------------------------------------------------------

    # Patch: Create and discard an empty file to avoid stalling when writing the NetCDF file.
    #        It seems to wake up the hard disk.
    p_inc = p_raw.replace(c.f_ext_nc, ".incomplete")
    if not os.path.exists(p_inc):
        open(p_inc, 'a').close()
    if os.path.exists(p_inc):
        os.remove(p_inc)

    # List years.
    year_l = [min(min(cntx.per_ref), min(cntx.per_fut)), max(max(cntx.per_ref), max(cntx.per_fut))]

    # Find the data at the requested timestep.
    p_obs = d_obs.replace(cntx.sep + "*" + cntx.sep, cntx.sep + "day" + cntx.sep) +\
        var.name + cntx.sep + "*" + c.f_ext_nc
    p_fut = d_fut.replace(cntx.sep + "*" + cntx.sep, cntx.sep + "day" + cntx.sep) +\
        var.name + cntx.sep + "*" + c.f_ext_nc
    p_l = sorted(glob.glob(p_obs)) + sorted(glob.glob(p_fut))

    # Open NetCDF.
    # Note: subset is not working with chunks.
    ds = fu.open_netcdf(p_l, chunks={c.dim_time: 365}, drop_variables=["time_vectors", "ts", "time_bnds"],
                        combine="by_coords")

    # Subset.
    ds_subset = utils.subset_lon_lat_time(ds, var.name, lon_l, lat_l, year_l)

    # Save NetCDF file.
    desc = cntx.sep + c.cat_raw + cntx.sep + os.path.basename(p_raw)
    fu.save_netcdf(ds_subset, p_raw, desc=desc)


def interpolate(
    var: vi.VarIdx,
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
    var: vi.VarIdx
        Variable.
    ds_stn: xr.Dataset
        NetCDF file containing station data.
    p_raw: str
        Path of the directory containing raw data.
    p_regrid: str
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
    if ds_raw.time.isel(time=[0, 1]).diff(dim=c.dim_time).values[0] <\
            np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0]:

        # Interpolate.
        fu.log(msg, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            ds_raw = ds_raw.resample(time="1D").mean(dim=c.dim_time, keep_attrs=True)

        # Save NetCDF file (raw).
        desc = cntx.sep + c.cat_raw + cntx.sep + os.path.basename(p_raw)
        fu.save_netcdf(ds_raw, p_raw, desc=desc)
        ds_raw = fu.open_netcdf(p_raw)
    else:
        fu.log(msg + " (not required; daily frequency)", True)

    # Spatial interpolation --------------------------------------------------------------------------------------------

    msg = "Interpolating (space)"

    if (not os.path.exists(p_regrid)) or cntx.opt_force_overwrite:

        fu.log(msg, True)

        # Method 1: Convert data to a new grid.
        if cntx.opt_ra:

            ds_regrid = regrid(ds_raw, ds_stn, var)

        # Method 2: Take the nearest information.
        else:

            # Assume a square around the location.
            lat_stn = round(float(ds_stn.lat.values), 1)
            lon_stn = round(float(ds_stn.lon.values), 1)

            # Determine nearest points.
            ds_regrid = ds_raw.sel(rlat=lat_stn, rlon=lon_stn, method="nearest", tolerance=1)

        # Save NetCDF file (regrid).
        desc = cntx.sep + c.cat_regrid + cntx.sep + os.path.basename(p_regrid)
        fu.save_netcdf(ds_regrid, p_regrid, desc=desc)

    else:

        fu.log(msg + " (not required)", True)


def regrid(
    ds_data: xr.Dataset,
    ds_grid: xr.Dataset,
    var: vi.VarIdx
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
    var: vi.VarIdx
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get longitude and latitude values (grid).
    if c.dim_lon in ds_grid.dims:
        grid_lon = ds_grid.lon.values.ravel()
        grid_lat = ds_grid.lat.values.ravel()
    else:
        grid_lon = ds_grid.longitude.values.ravel()
        grid_lat = ds_grid.latitude.values.ravel()

    # Create new mesh.
    new_grid = np.meshgrid(grid_lon, grid_lat)
    if np.min(new_grid[0]) > 0:
        new_grid[0] -= 360
    t_len = len(ds_data.time)
    arr_regrid = np.empty((t_len, len(grid_lat), len(grid_lon)))
    for t in range(0, t_len):
        arr_regrid[t, :, :] = griddata(
            (ds_data.lon.values.ravel(), ds_data.lat.values.ravel()),
            ds_data[var.name][t, :, :].values.ravel(),
            (new_grid[0], new_grid[1]), fill_value=np.nan, method="linear"
        )

    # Create data array.
    # Using xarray v0.20.2, the following line crashes with the following error:
    # {TypeError} Using a DataArray object to construct a variable is ambiguous, please extract the data using the .data
    # property.
    # It runs fine with v0.18.2.
    da_regrid = xr.DataArray(
        arr_regrid,
        coords=[(c.dim_time, ds_data.time[0:t_len]), (c.dim_lat, grid_lat), (c.dim_lon, grid_lon)],
        dims=[c.dim_time, c.dim_rlat, c.dim_rlon],
        attrs=ds_data.attrs
    )

    # Apply and create mask.
    if (cntx.obs_src == c.ens_era5_land) and (var.name not in [c.v_tas, c.v_tasmin, c.v_tasmax]):
        ds_regrid = da_regrid.to_dataset(name=var.name)
        da_mask = fu.create_mask()
        da_regrid = utils.apply_mask(ds_regrid[var.name], da_mask)

    # Create dataset.
    ds_regrid = da_regrid.to_dataset(name=var.name)
    ds_regrid[var.name].attrs[c.attrs_units] = ds_data[var.name].attrs[c.attrs_units]

    return ds_regrid


def perturbate(
    ds: xr.Dataset,
    var: vi.VarIdx
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Add small perturbation.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    var: vi.VarIdx
        Variable.

    Returns
    -------
    xr.Dataset
        Perturbed dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get perturbation value.
    d_val = 1e-12
    if cntx.opt_calib_perturb is not None:
        for i in range(len(cntx.opt_calib_perturb)):
            if var.name == cntx.opt_calib_perturb[i][0]:
                d_val = float(cntx.opt_calib_perturb[i][1])
                break

    # Data array has a single dimension.
    if len(ds[var.name].dims) == 1:
        ds[var.name].values = ds[var.name].values + np.random.rand(ds[var.name].values.shape[0]) * d_val

    # Data array has 3 dimensions, with unknown ranking.
    else:
        vals = ds[var.name].values.shape[0]
        i_time = list(ds[var.name].dims).index(c.dim_time)
        ds[var.name].values = ds[var.name].values +\
            np.random.rand(vals if i_time == 0 else 1, vals if i_time == 1 else 1, vals if i_time == 2 else 1) * d_val

    return ds


def preprocess(
    var: vi.VarIdx,
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
    var: vi.VarIdx
        Variable.
    ds_stn: xr.Dataset
        NetCDF file containing station data.
    p_obs: str
        Path of NetCDF file containing observed data.
    p_regrid: str
        Path of regrid NetCDF simulation file.
    p_regrid_ref: str
        Path of regrid NetCDF simulation file (reference period).
    p_regrid_fut: str
        Path of regrid NetCDF simulation file (future period).
    --------------------------------------------------------------------------------------------------------------------
    """

    ds_fut = fu.open_netcdf(p_regrid)

    # Observations -----------------------------------------------------------------------------------------------------

    if (not os.path.exists(p_obs)) or cntx.opt_force_overwrite:

        # Drop February 29th and select reference period.
        ds_obs = utils.remove_feb29(ds_stn)
        ds_obs = utils.sel_period(ds_obs, cntx.per_ref)

        # Save NetCDF file.
        desc = cntx.sep + c.cat_obs + cntx.sep + os.path.basename(p_obs)
        fu.save_netcdf(ds_obs, p_obs, desc=desc)

    # Simulated climate (future period) --------------------------------------------------------------------------------

    if os.path.exists(p_regrid_fut) and (not cntx.opt_force_overwrite):

        ds_regrid_fut = fu.open_netcdf(p_regrid_fut)

    else:

        # Drop February 29th.
        ds_regrid_fut = utils.remove_feb29(ds_fut)

        # Adjust values that do not make sense.
        if var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot, c.v_clt]:
            ds_regrid_fut[var.name].values[ds_regrid_fut[var.name] < 0] = 0
            if var.name == c.v_clt:
                ds_regrid_fut[var.name].values[ds_regrid_fut[var.name] > 100] = 100

        # Add small perturbation.
        if var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            perturbate(ds_regrid_fut, var)

        # Convert to a 365-day calendar.
        ds_regrid_fut = utils.convert_to_365_calender(ds_regrid_fut)

        # Save dataset.
        desc = cntx.sep + c.cat_regrid + cntx.sep + os.path.basename(p_regrid_fut)
        fu.save_netcdf(ds_regrid_fut, p_regrid_fut, desc=desc)

    # Simulated climate (reference period) -----------------------------------------------------------------------------

    if (not os.path.exists(p_regrid_ref)) or cntx.opt_force_overwrite:

        # Select reference period.
        ds_regrid_ref = utils.sel_period(ds_regrid_fut, cntx.per_ref)

        if var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot, c.v_clt]:
            pos = np.where(np.squeeze(ds_regrid_ref[var.name].values) > 0.01)[0]
            ds_regrid_ref[var.name][pos] = 1e-12

        # Save dataset.
        desc = cntx.sep + c.cat_regrid + cntx.sep + os.path.basename(p_regrid_ref)
        fu.save_netcdf(ds_regrid_ref, p_regrid_ref, desc=desc)


def postprocess(
    var: vi.VarIdx,
    nq: int,
    up_qmf: float,
    time_win: int,
    ds_stn: xr.Dataset,
    p_obs: str,
    p_sim: str,
    p_sim_adj: str,
    p_qmf: str,
    title: str = "",
    p_fig: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs post-processing.

    Parameters
    ----------
    var: vi.VarIdx
        Variable.
    nq: int
        Number of quantiles.
    up_qmf: float
        Upper limit of quantile mapping function.
    time_win: int
        Window size.
    ds_stn: xr.Dataset
        NetCDF file containing station data.
    p_obs: str
        Path of NetCDF file containing observed data (reference period).
    p_sim: str
        Path of NetCDF file containing simulation data.
    p_sim_adj: str
        Path of NetCDF file containing adjusted simulation data.
    p_qmf: str
        Path of NetCDF file containing quantile map function.
    title: str, optional
        Title of figure.
    p_fig: str, optimal
        Path of figure.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    # 'p_stn' and 'p_obs' cannot be opened using xr.open_mfdataset.
    if not cntx.opt_ra:
        if var.name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
            da_stn_attrs = ds_stn[var.name].attrs
            ds_stn[var.name] = ds_stn[var.name] + c.d_KC
            ds_stn[var.name].attrs = da_stn_attrs
            ds_stn[var.name].attrs[c.attrs_units] = c.unit_K
        da_stn = ds_stn[var.name][:, 0, 0]
    else:
        da_stn = ds_stn[var.name]
    if c.dim_longitude in da_stn.dims:
        da_stn = da_stn.rename({c.dim_longitude: c.dim_rlon, c.dim_latitude: c.dim_rlat})
    ds_obs = fu.open_netcdf(p_obs)
    ds_sim = fu.open_netcdf(p_sim)
    da_obs = ds_obs[var.name]
    da_sim = ds_sim[var.name]
    ds_sim_adj = None

    # Simulation -------------------------------------------------------------------------------------------------------

    # Convert to a 365-day calendar.
    da_sim_365 = utils.convert_to_365_calender(ds_sim)[var.name]

    # Observation ------------------------------------------------------------------------------------------------------

    if var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
        kind = c.kind_mult
    elif var.name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
        kind = c.kind_add
    else:
        kind = c.kind_add

    # Interpolate.
    if not cntx.opt_ra:
        da_stn = da_stn.interpolate_na(dim=c.dim_time)

    # Quantile Mapping Function ----------------------------------------------------------------------------------------

    # Load transfer function.
    if (p_qmf != "") and os.path.exists(p_qmf) and (not cntx.opt_force_overwrite):
        chunks = {c.dim_time: 1} if cntx.use_chunks else None
        ds_qmf = fu.open_netcdf(p_qmf, chunks=chunks)

    # Calculate transfer function.
    else:
        da_qmf = xr.DataArray(train(da_obs.squeeze(), da_stn.squeeze(), nq, c.group, kind, time_win,
                                    detrend_order=c.detrend_order))
        if var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            da_qmf.values[da_qmf > up_qmf] = up_qmf
            da_qmf.values[da_qmf < -up_qmf] = -up_qmf
        ds_qmf = da_qmf.to_dataset(name=var.name)
        ds_qmf[var.name].attrs[c.attrs_group] = da_qmf.attrs[c.attrs_group]
        ds_qmf[var.name].attrs[c.attrs_kind] = da_qmf.attrs[c.attrs_kind]
        ds_qmf[var.name].attrs[c.attrs_units] = da_obs.attrs[c.attrs_units]
        if p_qmf != "":
            desc = cntx.sep + c.cat_qmf + cntx.sep + os.path.basename(p_qmf)
            fu.save_netcdf(ds_qmf, p_qmf, desc=desc)
            ds_qmf = fu.open_netcdf(p_qmf)

    # Quantile Mapping -------------------------------------------------------------------------------------------------

    # Load quantile mapping.
    if (p_sim_adj != "") and os.path.exists(p_sim_adj) and (not cntx.opt_force_overwrite):
        chunks = {c.dim_time: 1} if cntx.use_chunks else None
        ds_sim_adj = fu.open_netcdf(p_sim_adj, chunks=chunks)

    # Apply transfer function.
    else:
        try:
            interp = (True if not cntx.opt_ra else False)
            da_sim_adj = xr.DataArray(predict(da_sim_365.squeeze(), ds_qmf[var.name].squeeze(),
                                      interp=interp, detrend_order=c.detrend_order))
            ds_sim_adj = da_sim_adj.to_dataset(name=var.name)
            del ds_sim_adj[var.name].attrs[c.attrs_bias]
            ds_sim_adj[var.name].attrs[c.attrs_sname] = da_stn.attrs[c.attrs_sname]
            ds_sim_adj[var.name].attrs[c.attrs_lname] = da_stn.attrs[c.attrs_lname]
            ds_sim_adj[var.name].attrs[c.attrs_units] = da_stn.attrs[c.attrs_units]
            if c.attrs_gmap in da_stn.attrs:
                ds_sim_adj[var.name].attrs[c.attrs_gmap]  = da_stn.attrs[c.attrs_gmap]
            if p_sim_adj != "":
                desc = cntx.sep + c.cat_qqmap + cntx.sep + os.path.basename(p_sim_adj)
                fu.save_netcdf(ds_sim_adj, p_sim_adj, desc=desc)
                ds_sim_adj = fu.open_netcdf(p_sim_adj)

        except ValueError as err:
            fu.log("Failed to create QQMAP NetCDF file.", True)
            fu.log(format(err), True)
            pass

    # Create plots -----------------------------------------------------------------------------------------------------

    if p_fig != "":

        # Select reference period.
        ds_sim_adj_ref = utils.sel_period(ds_sim_adj, cntx.per_ref)

        # Convert to data arrays.
        da_sim_adj_ref = ds_sim_adj_ref[var.name]
        da_sim_adj     = ds_sim_adj[var.name]
        da_qmf         = ds_qmf[var.name]

        def convert_units(da: xr.DataArray, units: str) -> xr.DataArray:
            if (da.units == c.unit_kg_m2s1) and (units == c.unit_mm):
                da = da * c.spd
                da.attrs[c.attrs_units] = units
            elif (da.units == c.unit_K) and (units == c.unit_C):
                da = da - c.d_KC
                da[c.attrs_units] = units
            return da

        # Convert units.
        if var.name in [c.v_pr, c.v_evspsbl, c.v_evspsblpot]:
            da_stn         = convert_units(da_stn, c.unit_mm)
            da_obs         = convert_units(da_obs, c.unit_mm)
            da_sim         = convert_units(da_sim, c.unit_mm)
            da_sim_adj     = convert_units(da_sim_adj, c.unit_mm)
            da_sim_adj_ref = convert_units(da_sim_adj_ref, c.unit_mm)
            da_qmf       = convert_units(da_qmf, c.unit_mm)
        elif var.name in [c.v_tas, c.v_tasmin, c.v_tasmax]:
            da_stn         = convert_units(da_stn, c.unit_C)
            da_obs         = convert_units(da_obs, c.unit_C)
            da_sim         = convert_units(da_sim, c.unit_C)
            da_sim_adj     = convert_units(da_sim_adj, c.unit_C)
            da_sim_adj_ref = convert_units(da_sim_adj_ref, c.unit_C)

        # Select center coordinates.
        da_stn_xy         = da_stn
        da_obs_xy         = da_obs
        da_sim_xy         = da_sim
        da_sim_adj_ref_xy = da_sim_adj_ref
        da_sim_adj_xy     = da_sim_adj
        da_qmf_xy         = da_qmf
        if cntx.opt_ra:
            if cntx.p_bounds == "":
                da_stn_xy       = utils.subset_ctrl_pt(da_stn_xy)
                da_obs_xy       = utils.subset_ctrl_pt(da_obs_xy)
                da_sim_xy       = utils.subset_ctrl_pt(da_sim_xy)
                da_sim_adj_xy     = utils.subset_ctrl_pt(da_sim_adj_xy)
                da_sim_adj_ref_xy = utils.subset_ctrl_pt(da_sim_adj_ref_xy)
                da_qmf_xy       = utils.subset_ctrl_pt(da_qmf_xy)
            else:
                da_stn_xy       = utils.squeeze_lon_lat(da_stn_xy)
                da_obs_xy       = utils.squeeze_lon_lat(da_obs_xy)
                da_sim_xy       = utils.squeeze_lon_lat(da_sim_xy)
                da_sim_adj_xy     = utils.squeeze_lon_lat(da_sim_adj_xy)
                da_sim_adj_ref_xy = utils.squeeze_lon_lat(da_sim_adj_ref_xy)
                da_qmf_xy       = utils.squeeze_lon_lat(da_qmf_xy)

        if cntx.opt_diagnostic:

            # Generate summary plot.
            plot.plot_calib(da_stn_xy, da_obs_xy, da_sim_xy, da_sim_adj_xy, da_sim_adj_ref_xy, da_qmf_xy,
                            var, title, p_fig)

            # Generate time series only.
            plot.plot_calib_ts(da_stn_xy, da_sim_xy, da_sim_adj_xy, var, title,
                               p_fig.replace(c.f_ext_png, "_ts" + c.f_ext_png))

    return ds_sim_adj if (p_fig == "") else None


def bias_adj(
    stn: str,
    var: vi.VarIdx,
    sim_name: str = "",
    calc_err: bool = False
):

    """
    -------------------------------------------------------------------------------------------------------------------
    Performs bias adjustment.

    Parameters
    ----------
    stn: str
        Station name.
    var: vi.VarIdx
        Variable.
    sim_name: str
        Simulation name.
    calc_err: bool
        If True, only calculate the error (will not work properly in parallel mode).
    -------------------------------------------------------------------------------------------------------------------
    """

    # List regrid files.
    d_regrid = cntx.d_scen(stn, c.cat_regrid, var.name)
    p_regrid_l = fu.list_files(d_regrid)
    if p_regrid_l is None:
        fu.log("The required files are not available.", True)
        return
    p_regrid_l = [i for i in p_regrid_l if "_4qqmap" not in i]
    p_regrid_l.sort()

    # Loop through simulation sets.
    for i in range(len(p_regrid_l)):
        p_regrid_tokens = p_regrid_l[i].split(cntx.sep)
        sim_name_i = p_regrid_tokens[len(p_regrid_tokens) - 1].replace(var.name + "_", "").replace(c.f_ext_nc, "")

        # Skip iteration if it does not correspond to the specified simulation name.
        if (sim_name != "") and (sim_name != sim_name_i):
            continue

        # Best parameter set.
        bias_err_best = -1

        # Loop through combinations of nq values, up_qmf and time_win values.
        for nq in cntx.nq_calib:
            for up_qmf in cntx.up_qmf_calib:
                for time_win in cntx.time_win_calib:

                    # NetCDF files.
                    p_stn        = cntx.p_stn(var.name, stn)
                    p_regrid     = p_regrid_l[i]
                    p_regrid_ref = p_regrid.replace(c.f_ext_nc, "_ref_4qqmap" + c.f_ext_nc)
                    p_regrid_fut = p_regrid.replace(c.f_ext_nc, "_4qqmap" + c.f_ext_nc)

                    # If there is a single combination of calibration parameters, NetCDF files can be saved, so that
                    # they don't have to be created during post-process.
                    p_qqmap = ""
                    p_qmf   = ""
                    if (len(cntx.nq_calib) == 1) and (len(cntx.up_qmf_calib) == 1) and (len(cntx.time_win_calib) == 1):
                        p_qqmap = p_regrid.replace(c.cat_regrid, c.cat_qqmap)
                        p_qmf   = p_regrid.replace(c.cat_regrid, c.cat_qmf)

                    # Verify if required files exist.
                    msg = "File missing: "
                    if ((not os.path.exists(p_stn)) or (not os.path.exists(p_regrid_ref)) or
                        (not os.path.exists(p_regrid_fut))) and\
                       (not cntx.opt_force_overwrite):
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
                    ds_stn = utils.sel_period(ds_stn, cntx.per_ref)

                    # Path and title of calibration figure and csv file.
                    fn_fig = var.name + "_" + sim_name_i + "_" + c.cat_fig_calibration + c.f_ext_png
                    comb = "nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                    title = sim_name_i + "_" + comb
                    d = c.cat_scen + cntx.sep + c.cat_fig + cntx.sep + c.cat_fig_calibration
                    p_fig = cntx.d_scen(stn, d, var.name) + comb + cntx.sep + fn_fig
                    p_fig_csv =\
                        p_fig.replace(cntx.sep + var.name + cntx.sep,
                                      cntx.sep + var.name + "_" + c.f_csv + cntx.sep).\
                        replace(c.f_ext_png, "_" + c.stat_mean + c.f_ext_csv)
                    p_fig_ts = p_fig.replace(c.f_ext_png, "_ts" + c.f_ext_png)
                    p_fig_ts_csv = p_fig_ts.replace(cntx.sep + var.name + cntx.sep,
                                                    cntx.sep + var.name + "_" + c.f_csv + cntx.sep).\
                        replace(c.f_ext_png, c.f_ext_csv)

                    # Bias adjustment ----------------------------------------------------------------------------------

                    if not calc_err:

                        msg = "nq=" + str(nq) + ", up_qmf=" + str(up_qmf) + ", time_win=" + str(time_win)
                        if not(os.path.exists(p_fig)) or \
                           (not(os.path.exists(p_fig_csv)) and (c.f_csv in cntx.opt_diagnostic_format)) or\
                           not(os.path.exists(p_fig_ts)) or \
                           (not (os.path.exists(p_fig_ts_csv)) and (c.f_csv in cntx.opt_diagnostic_format)) or \
                           not(os.path.exists(p_qqmap)) or not(os.path.exists(p_qmf)) or cntx.opt_force_overwrite:
                            fu.log(msg, True)

                            # Calculate QQ and generate calibration plots.
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                postprocess(var, nq, up_qmf, time_win, ds_stn, p_regrid_ref, p_regrid_fut, p_qqmap,
                                            p_qmf, title, p_fig)
                        else:
                            fu.log(msg + " (not required)", True)

                    # Bias error ---------------------------------------------------------------------------------------

                    elif os.path.exists(p_qqmap):

                        # Extract the reference period from the adjusted simulation.
                        ds_qqmap_ref = fu.open_netcdf(p_qqmap)
                        ds_qqmap_ref = utils.remove_feb29(ds_qqmap_ref)
                        ds_qqmap_ref = utils.sel_period(ds_qqmap_ref, cntx.per_ref)

                        # Calculate the error between observations and simulation for the reference period.
                        bias_err_current = float(round(utils.calc_error(ds_stn[var.name], ds_qqmap_ref[var.name]), 4))

                        # Set calibration parameters (nq, up_qmf and time_win) and calculate error according to the
                        # selected method.
                        if (not cntx.opt_calib_auto) or\
                           (cntx.opt_calib_auto and ((bias_err_best < 0) or (bias_err_current < bias_err_best))):
                            col_names = ["nq", "up_qmf", "time_win", "bias_err"]
                            col_values = [float(nq), up_qmf, float(time_win), bias_err_current]
                            calib_row = (cntx.df_calib["sim_name"] == sim_name) &\
                                        (cntx.df_calib["stn"] == stn) &\
                                        (cntx.df_calib["var"] == var.name)
                            cntx.df_calib.loc[calib_row, col_names] = col_values
                            fu.save_csv(cntx.df_calib, cntx.p_calib)


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
    if os.path.exists(cntx.p_calib):
        cntx.df_calib = pd.read_csv(cntx.p_calib)
        if len(cntx.df_calib) > 0:
            sim_name_l = list(cntx.df_calib["sim_name"])
            stn_l      = list(cntx.df_calib["stn"])
            vi_code_l  = list(cntx.df_calib["var"])
            nq_l       = list(cntx.df_calib["nq"])
            up_qmf_l   = list(cntx.df_calib["up_qmf"])
            time_win_l = list(cntx.df_calib["time_win"])
            bias_err_l = list(cntx.df_calib["bias_err"])
        fu.log("Calibration file loaded.", True)

    # List CORDEX files.
    list_cordex = fu.list_cordex(cntx.d_proj, cntx.rcps)

    # Stations.
    stns = cntx.stns
    if cntx.opt_ra:
        stns = [cntx.obs_src]

    # List simulation names, stations and variables.
    for i_rcp in range(len(cntx.rcps)):
        rcp = cntx.rcps[i_rcp]
        sim_l = list_cordex[rcp]
        sim_l.sort()
        for i_sim in range(0, len(sim_l)):
            list_i = list_cordex[rcp][i_sim].split(cntx.sep)
            sim_name = list_i[cntx.rank_inst()] + "_" + list_i[cntx.rank_inst() + 1]
            for stn in stns:
                for var in cntx.vars.items:

                    # Add the combination if it does not already exist.
                    if (cntx.df_calib is None) or\
                       (len(cntx.df_calib.loc[(cntx.df_calib["sim_name"] == sim_name) &
                                              (cntx.df_calib["stn"] == stn) &
                                              (cntx.df_calib["var"] == var.name)]) == 0):
                        sim_name_l.append(sim_name)
                        stn_l.append(stn)
                        vi_code_l.append(var.name)
                        nq_l.append(cntx.nq_default)
                        up_qmf_l.append(cntx.up_qmf_default)
                        time_win_l.append(cntx.time_win_default)
                        bias_err_l.append(cntx.bias_err_default)

                        # Update dataframe.
                        cntx.df_calib = build_df(sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Build dataframe.
    cntx.df_calib = build_df(sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Save calibration parameters to a CSV file.
    if cntx.p_calib != "":

        # Create directory if it does not already exist.
        d = os.path.dirname(cntx.p_calib)
        if not (os.path.isdir(d)):
            os.makedirs(d)

        # Save file.
        fu.save_csv(cntx.df_calib, cntx.p_calib)
        if os.path.exists(cntx.p_calib):
            fu.log("Calibration file created or updated.", True)


def gen():

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
    d_exec = cntx.d_scen("", "", "")
    if not(os.path.isdir(d_exec)):
        os.makedirs(d_exec)

    # Step #2: Data selection.

    # Step #2c: Convert observations from CSV to NetCDF files (for stations) or
    #           Combine all years in a single NetCDF file (for reanalysis).
    # This creates one .nc file per variable-station in ~/<country>/<project>/<stn>/obs/<source>/<var>/.
    fu.log("=")

    if not cntx.opt_ra:
        fu.log("Step #2c  Converting observations from CSV to NetCDF files")
        for var in cntx.vars.items:
            load_observations(var)
    else:
        fu.log("Step #2c  Merging reanalysis NetCDF files.")
        for var_ra in cntx.vars_ra.items:
            if cntx.obs_src == c.ens_enacts:
                preload_reanalysis(var_ra)
            load_reanalysis(var_ra)

    # Step #2d: List directories potentially containing CORDEX files (but not necessarily for all selected variables).
    fu.log("=")
    fu.log("Step #2d  Listing directories with CORDEX files")
    list_cordex = fu.list_cordex(cntx.d_proj, cntx.rcps)

    fu.log("=")
    fu.log("Step #3-5 Producing climate scenarios")

    # Loop through variables.
    for var in cntx.vars.items:

        # Select file names for observation (or reanalysis).
        if not cntx.opt_ra:
            d_stn = cntx.d_stn(var.name)
            p_stn_l = glob.glob(d_stn + "*" + c.f_ext_nc)
            p_stn_l.sort()
        else:
            p_stn_l = [cntx.d_stn(var.name) + var.name + "_" + cntx.obs_src + c.f_ext_nc]

        # Loop through stations.
        for i_stn in range(0, len(p_stn_l)):

            # Station name.
            p_stn = p_stn_l[i_stn]
            if not cntx.opt_ra:
                stn = os.path.basename(p_stn).replace(c.f_ext_nc, "").replace(var.name + "_", "")
                if not (stn in cntx.stns):
                    continue
            else:
                stn = cntx.obs_src

            # Directories.
            d_stn    = cntx.d_stn(var.name)
            d_obs    = cntx.d_scen(stn, c.cat_obs, var.name)
            d_raw    = cntx.d_scen(stn, c.cat_scen + cntx.sep + c.cat_raw, var.name)
            d_regrid = cntx.d_scen(stn, c.cat_scen + cntx.sep + c.cat_regrid, var.name)
            d_qqmap  = cntx.d_scen(stn, c.cat_scen + cntx.sep + c.cat_qqmap, var.name)
            d_qmf    = cntx.d_scen(stn, c.cat_scen + cntx.sep + c.cat_qmf, var.name)
            d_fig_calibration = cntx.d_scen(stn, c.cat_fig + cntx.sep + c.cat_fig_calibration, var.name)
            d_fig_postprocess = cntx.d_scen(stn, c.cat_fig + cntx.sep + c.cat_fig_postprocess, var.name)
            d_fig_workflow    = cntx.d_scen(stn, c.cat_fig + cntx.sep + c.cat_fig_workflow, var.name)

            # Load station data, drop February 29th and select reference period.
            ds_stn = fu.open_netcdf(p_stn)
            ds_stn = utils.remove_feb29(ds_stn)
            ds_stn = utils.sel_period(ds_stn, cntx.per_ref)

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
            for rcp in cntx.rcps:

                # Extract and sort simulations lists.
                list_cordex_ref = list_cordex[rcp + "_historical"]
                list_cordex_fut = list_cordex[rcp]
                list_cordex_ref.sort()
                list_cordex_fut.sort()
                n_sim = len(list_cordex_ref)

                fu.log("Processing: " + var.name + ", " + stn + ", " + rcp, True)

                # Scalar mode.
                if cntx.n_proc == 1:
                    for i_sim in range(n_sim):
                        gen_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, False, i_sim)

                # Parallel processing mode.
                else:

                    # Perform extraction.
                    # A first call to gen_single is required for the extraction to be done in scalar mode (before
                    # forking) because of the incompatibility of xr.open_mfdataset with parallel processing.
                    for i_sim in range(n_sim):
                        gen_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, True, i_sim)

                    # Loop until all simulations have been processed.
                    while True:

                        # Calculate the number of processed files (before generation).
                        # This quick verification is based on the QQMAP NetCDF file, but there are several other
                        # files that are generated. The 'completeness' verification is more complete in scalar mode.
                        n_sim_proc_before = len(list(glob.glob(d_qqmap + "*" + c.f_ext_nc)))

                        # Scalar processing mode.
                        if cntx.n_proc == 1:
                            for i_sim in range(n_sim):
                                gen_single(list_cordex_ref, list_cordex_fut, ds_stn, d_raw, var, stn, rcp, False,
                                           i_sim)

                        # Parallel processing mode.
                        else:

                            try:
                                fu.log("Splitting work between " + str(cntx.n_proc) + " threads.", True)
                                pool = multiprocessing.Pool(processes=min(cntx.n_proc, len(list_cordex_ref)))
                                func = functools.partial(gen_single, list_cordex_ref, list_cordex_fut, ds_stn,
                                                         d_raw, var, stn, rcp, False)
                                pool.map(func, list(range(n_sim)))
                                pool.close()
                                pool.join()
                                fu.log("Fork ended.", True)
                            except Exception as e:
                                fu.log(str(e))
                                pass

                        # Calculate the number of processed files (after generation).
                        n_sim_proc_after = len(list(glob.glob(d_qqmap + "*" + c.f_ext_nc)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cntx.n_proc == 1) or (n_sim_proc_before == n_sim_proc_after):
                            break

                # Calculate bias adjustment errors.
                for i_sim in range(n_sim):
                    tokens = list_cordex_fut[i_sim].split(cntx.sep)
                    sim_name = tokens[cntx.rank_inst()] + "_" + tokens[cntx.rank_gcm()]
                    bias_adj(stn, var, sim_name, True)


def gen_single(
    list_cordex_ref: [str],
    list_cordex_fut: [str],
    ds_stn: xr.Dataset,
    d_raw: str,
    var: vi.VarIdx,
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
    list_cordex_ref: [str]
        List of CORDEX files for the reference period.
    list_cordex_fut: [str]
        List of CORDEX files for the future period.
    ds_stn: xr.Dataset
        Station file.
    d_raw: str
        Directory containing raw NetCDF files.
    var: vi.VarIdx
        Variable.
    stn: str
        Station.
    rcp: str
        RCP emission scenario.
    extract_only:
        If True, only extract.
    i_sim_proc: int
        Rank of simulation to process (in ascending order of raw NetCDF file names).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Directories containing simulations.
    d_sim_ref = list_cordex_ref[i_sim_proc]
    d_sim_fut = list_cordex_fut[i_sim_proc]

    # Get simulation name.
    tokens = d_sim_fut.split(cntx.sep)
    sim_name = tokens[cntx.rank_inst()] + "_" + tokens[cntx.rank_gcm()]

    fu.log("=")
    fu.log("Variable   : " + var.name)
    fu.log("Station    : " + stn)
    fu.log("RCP        : " + rcp)
    fu.log("Simulation : " + sim_name)
    fu.log("=")

    # Skip iteration if the variable 'var' is not available in the current directory.
    p_sim_ref_l = list(glob.glob(d_sim_ref + cntx.sep + var.name + cntx.sep + "*" + c.f_ext_nc))
    p_sim_fut_l = list(glob.glob(d_sim_fut + cntx.sep + var.name + cntx.sep + "*" + c.f_ext_nc))

    if (len(p_sim_ref_l) == 0) or (len(p_sim_fut_l) == 0):
        fu.log("Skipping iteration: data not available for simulation-variable.", True)
        if cntx.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Files within CORDEX or CORDEX-NA.
    if "cordex" in d_sim_fut.lower():
        p_raw = d_raw + var.name + "_" + tokens[cntx.rank_inst()] + "_" +\
                tokens[cntx.rank_gcm()].replace("*", "_") + c.f_ext_nc
    elif len(d_sim_fut) == 3:
        p_raw = d_raw + var.name + "_Ouranos_" + d_sim_fut + c.f_ext_nc
    else:
        p_raw = None

    # Skip iteration if the simulation or simulation-variable is in the exception list.
    is_sim_except = False
    for sim_except in cntx.sim_excepts:
        if sim_except in p_raw:
            is_sim_except = True
            fu.log("Skipping iteration: simulation-variable exception.", True)
            break
    is_var_sim_except = False
    for var_sim_except in cntx.var_sim_excepts:
        if var_sim_except in p_raw:
            is_var_sim_except = True
            fu.log("Skipping iteration: simulation exception.", True)
            break
    if is_sim_except or is_var_sim_except:
        if cntx.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Paths and NetCDF files.
    p_regrid     = p_raw.replace(c.cat_raw, c.cat_regrid)
    p_qqmap      = p_raw.replace(c.cat_raw, c.cat_qqmap)
    p_qmf        = p_raw.replace(c.cat_raw, c.cat_qmf)
    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + c.cat_qqmap + c.f_ext_nc
    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + c.cat_qqmap + c.f_ext_nc
    p_obs        = cntx.p_obs(stn, var.name)

    # Step #3: Extraction.
    # This step only works in scalar mode.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/raw/<var>/
    msg = "Step #3   Extracting projections"
    if (not os.path.isfile(p_raw)) or cntx.opt_force_overwrite:
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
    if (not os.path.isfile(p_regrid)) or cntx.opt_force_overwrite:
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
       cntx.opt_force_overwrite:
        fu.log(msg)
        preprocess(var, ds_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut)
    else:
        fu.log(msg + " (not required)")

    # Step #5: Post-processing.

    # Step #5a: Calculate adjustment factors.
    fu.log("-")
    msg = "Step #5a  Calculating adjustment factors"
    if cntx.opt_calib:
        fu.log(msg)
        bias_adj(stn, var, sim_name)
    else:
        fu.log(msg + " (not required)")
    df_sel = cntx.df_calib.loc[(cntx.df_calib["sim_name"] == sim_name) &
                               (cntx.df_calib["stn"] == stn) &
                               (cntx.df_calib["var"] == var.name)]
    if df_sel is not None:
        nq       = float(df_sel["nq"])
        up_qmf   = float(df_sel["up_qmf"])
        time_win = float(df_sel["time_win"])
        bias_err = float(df_sel["bias_err"])
    else:
        nq       = float(cntx.nq_default)
        up_qmf   = float(cntx.up_qmf_default)
        time_win = float(cntx.time_win_default)
        bias_err = float(cntx.bias_err_default)

    # Display calibration parameters.
    msg = "Selected parameters: nq=" + str(nq) + ", up_qmf=" + str(up_qmf) + \
          ", time_win=" + str(time_win) + ", bias_err=" + str(bias_err)
    fu.log(msg, True)

    # Step #5b: Statistical downscaling.
    # Step #5c: Bias correction.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/qqmap/<var>/.
    fu.log("-")
    msg = "Step #5bc Statistical downscaling and adjusting bias"
    if (not(os.path.isfile(p_qqmap)) or not(os.path.isfile(p_qmf))) or cntx.opt_force_overwrite:
        fu.log(msg)
        postprocess(var, int(nq), up_qmf, int(time_win), ds_stn, p_regrid_ref, p_regrid_fut, p_qqmap, p_qmf)
    else:
        fu.log(msg + " (not required)")

    if cntx.n_proc > 1:
        fu.log("Work done!", True)


def gen_per_var(
    func_name: str,
    view_code: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate diagnostic and cycle plots.

    func_name: str
        Name of function to be called.
    view_code: Optional[str]
        View code.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Number of variables to process.
    n_var = len(cntx.vars.items)

    # Scalar mode.
    if cntx.n_proc == 1:
        for i_var in range(n_var):
            if func_name == "calc_diag_cycle":
                calc_diag_cycle(cntx.vars.code_l, i_var)
            elif func_name == "stats.calc_map":
                stats.calc_map(cntx.vars.code_l, i_var)
            elif func_name == "stats.calc_ts":
                stats.calc_ts(view_code, cntx.vars.code_l, i_var)
            else:
                stats.calc_stat_tbl(cntx.vars.code_l, i_var)

    # Parallel processing mode.
    else:

        for i in range(math.ceil(n_var / cntx.n_proc)):

            # Select variables to process in the current loop.
            i_first = i * cntx.n_proc
            i_last = min((i + 1) * cntx.n_proc, n_var - 1)
            n_proc = i_last - i_first + 1
            var_name_l = cntx.vars.code_l[i_first, i_last]

            try:
                fu.log("Splitting work between " + str(n_proc) + " threads.", True)
                pool = multiprocessing.Pool(processes=n_proc)
                if func_name == "calc_diag_cycle":
                    func = functools.partial(calc_diag_cycle, var_name_l)
                elif func_name == "stats.calc_map":
                    func = functools.partial(stats.calc_map, var_name_l)
                elif func_name == "stats.calc_ts":
                    func = functools.partial(stats.calc_ts, view_code, var_name_l)
                else:
                    func = functools.partial(stats.calc_stat_tbl, var_name_l)
                pool.map(func, list(range(var_name_l)))
                pool.close()
                pool.join()
                fu.log("Fork ended.", True)

            except Exception as e:
                fu.log(str(e))
                pass


def calc_diag_cycle(
    variables: List[str],
    i_var_proc: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Generate diagnostic and cycle plots (single variable).

    variables: List[str],
        Variables.
    i_var_proc: int
        Rank of variable to process.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get variable.
    var_name = variables[i_var_proc]
    var = vi.VarIdx(var_name)

    # Loop through stations.
    stns = (cntx.stns if not cntx.opt_ra else [cntx.obs_src])
    for stn in stns:

        fu.log("Processing: " + var_name + ", " + stn, True)

        # Path ofo NetCDF file containing station data.
        p_obs = cntx.p_obs(stn, var_name)

        # Loop through raw NetCDF files.
        p_raw_l = list(glob.glob(cntx.d_scen(stn, c.cat_raw, var_name) + "*" + c.f_ext_nc))
        for i in range(len(p_raw_l)):
            p_raw = p_raw_l[i]

            # Path of NetCDF files.
            p_regrid = p_raw.replace(c.cat_raw, c.cat_regrid)
            p_qqmap = p_raw.replace(c.cat_raw, c.cat_qqmap)
            p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + c.cat_qqmap + c.f_ext_nc
            p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + c.cat_qqmap + c.f_ext_nc

            # Calibration parameters.
            sim_name = os.path.basename(p_raw).replace(var_name + "_", "").replace(c.f_ext_nc, "")
            df_sel = cntx.df_calib.loc[(cntx.df_calib["sim_name"] == sim_name) &
                                       (cntx.df_calib["stn"] == stn) &
                                       (cntx.df_calib["var"] == var_name)]
            nq = float(df_sel["nq"])
            up_qmf = float(df_sel["up_qmf"])
            time_win = float(df_sel["time_win"])

            # File name.
            fn_fig = p_regrid_fut.split(cntx.sep)[-1]. \
                replace("_4qqmap" + c.f_ext_nc, "_" + c.cat_fig_postprocess + c.f_ext_png)

            # Generate diagnostic plots.
            if cntx.opt_diagnostic and (len(cntx.opt_diagnostic_format) > 0):

                # This creates one file:
                #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/postprocess/<var>/*.png
                title = fn_fig[:-4] + "_nq_" + str(nq) + "_upqmf_" + str(up_qmf) + "_timewin_" + str(time_win)
                d = c.cat_fig + cntx.sep + c.cat_scen + cntx.sep + c.cat_fig_postprocess
                p_fig = cntx.d_scen(stn, d, var_name) + fn_fig
                stats.calc_postprocess(p_obs, p_regrid_fut, p_qqmap, var, p_fig, title)

                # This creates one file:
                #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/workflow/<var>/*.png
                d = c.cat_fig + cntx.sep + c.cat_scen + cntx.sep + c.cat_fig_workflow
                p_fig = cntx.d_scen(stn, d, var_name) + \
                    p_regrid_fut.split(cntx.sep)[-1].replace("4qqmap" + c.f_ext_nc, c.cat_fig_workflow + c.f_ext_png)
                stats.calc_workflow(var, int(nq), up_qmf, int(time_win), p_regrid_ref, p_regrid_fut, p_fig)

            # Generate monthly and daily plots.
            if cntx.opt_cycle and (len(cntx.opt_cycle_format) > 0):

                ds_qqmap = fu.open_netcdf(p_qqmap)
                for per in cntx.per_hors:
                    per_str = str(per[0]) + "_" + str(per[1])

                    # This creates 2 files:
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>/*.png
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>_csv/*.csv
                    title = fn_fig[:-4].replace(c.cat_fig_postprocess, per_str + "_" + c.view_cycle_ms)
                    stats.calc_cycle(ds_qqmap, stn, var, per, c.freq_MS, title)

                    # This creates 2 files:
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>/*.png
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>_csv/*.csv
                    title = fn_fig[:-4].replace(c.cat_fig_postprocess, per_str + "_" + c.view_cycle_d)
                    stats.calc_cycle(ds_qqmap, stn, var, per, c.freq_D, title)

        if os.path.exists(p_obs) and cntx.opt_cycle and (len(cntx.opt_cycle_format) > 0):
            ds_obs = fu.open_netcdf(p_obs)
            per_str = str(cntx.per_ref[0]) + "_" + str(cntx.per_ref[1])

            # This creates 2 files:
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>/*.png
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>_csv/*.csv
            title = var_name + "_" + per_str + "_" + c.view_cycle_ms
            stats.calc_cycle(ds_obs, stn, var, cntx.per_ref, c.freq_MS, title)

            # This creates 2 files:
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>/*.png
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen_cycle_d/<var>_csv/*.csv
            title = var_name + "_" + per_str + "_" + c.view_cycle_d
            stats.calc_cycle(ds_obs, stn, var, cntx.per_ref, c.freq_D, title)


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
    if cntx.opt_scen:
        fu.log(msg)
        gen()
    else:
        fu.log(msg + not_req)

    # Statistics -------------------------------------------------------------------------------------------------------

    fu.log("=")
    msg = "Step #7a  Calculating statistics (scenarios)"
    if cntx.opt_stat[0]:
        fu.log(msg)
        gen_per_var("stats.calc_stat_tbl")
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #7b  Converting NetCDF to CSV files (scenarios)"
    if cntx.export_nc_to_csv[0] and not cntx.opt_ra:
        fu.log(msg)
        fu.log("-")
        stats.conv_nc_csv(c.cat_scen)
    else:
        fu.log(msg + not_req)

    # Plots ------------------------------------------------------------------------------------------------------------

    fu.log("=")
    fu.log("Step #8a  Generating post-process, workflow, daily and monthly plots (scenarios)")
    if (cntx.opt_diagnostic and (len(cntx.opt_diagnostic_format) > 0)) or\
       (cntx.opt_cycle and (len(cntx.opt_cycle_format) > 0)):
        fu.log(msg)
        gen_per_var("calc_diag_cycle")
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #8b  Generating time series (scenarios, bias-adjusted values)"
    if cntx.opt_ts[0] and (len(cntx.opt_ts_format) > 0):
        fu.log(msg)
        gen_per_var("stats.calc_ts", c.view_ts)
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #8c  Generating time series (scenarios, raw values and bias)"
    if cntx.opt_ts_bias[0] and (len(cntx.opt_ts_format) > 0):
        fu.log(msg)
        gen_per_var("stats.calc_ts", c.view_ts_bias)
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #8d  Generating maps (scenarios)"
    if cntx.opt_ra and cntx.opt_map[0] and (len(cntx.opt_ts_format) > 0):
        fu.log(msg)
        gen_per_var("stats.calc_map")
    else:
        fu.log(msg + " (not required)")

    fu.log("-")
    msg = "Step #8e  Generating cluster plots (scenarios)"
    if cntx.opt_cluster and (len(cntx.opt_cluster_format) > 0):
        fu.log(msg)
        stats.calc_clusters()
    else:
        fu.log(msg + " (not required)")


def run_calib():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through combinations of stations and variables.
    for stn in cntx.stns:
        for var in cntx.vars.items:

            fu.log("-", True)
            fu.log("Station  : " + stn, True)
            fu.log("Variable : " + var.name, True)
            fu.log("-", True)

            # Perform bias correction.
            bias_adj(stn, var)


if __name__ == "__main__":
    run()
