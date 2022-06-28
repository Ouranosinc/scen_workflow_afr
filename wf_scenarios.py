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
import xesmf as xe
import xarray as xr
import xarray.core.variable as xcv
import warnings
from typing import List, Optional

# Workflow libraries.
import wf_file_utils as fu
import wf_plot
import wf_stats as stats
import wf_utils
from cl_constant import const as c
from cl_context import cntx
from wf_qm import train, predict

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard.cl_delta import Delta
from dashboard.cl_hor import Hor
from dashboard.cl_lib import Lib
from dashboard.cl_rcp import RCP
from dashboard.cl_sim import Sim
from dashboard.cl_stat import Stat
from dashboard.cl_varidx import VarIdx, VarIdxs
from dashboard.cl_view import View


def load_observations(
    var: VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Load obervations.

    Parameters
    ----------
    var: VarIdx
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Station list file and station files.
    d_stn = cntx.d_stn(var.name)
    p_stn_info = glob.glob(d_stn + ".." + cntx.sep + "*" + c.F_EXT_CSV)
    p_stn_l = glob.glob(d_stn + "*" + c.F_EXT_CSV)
    p_stn_l.sort()

    # Compile data.
    for i in range(0, len(p_stn_l)):

        stn = os.path.basename(p_stn_l[i]).replace(c.F_EXT_NC, "").split("_")[1]

        if not(stn in cntx.stns):
            continue

        obs  = pd.read_csv(p_stn_l[i], sep=cntx.f_sep)
        time = pd.to_datetime(
            obs["annees"].astype("str") + "-" + obs["mois"].astype("str") + "-" + obs["jours"].astype("str"))

        # Find longitude and latitude.
        lon_lat_data = pd.read_csv(p_stn_info[0], sep=cntx.f_sep)
        lon = float(lon_lat_data[lon_lat_data["station"] == stn][c.DIM_LON])
        lat = float(lon_lat_data[lon_lat_data["station"] == stn][c.DIM_LAT])

        # Temperature --------------------------------------------------------------------------------------------------

        if var.name in [c.V_TAS, c.V_TASMAX, c.V_TASMIN]:

            # Extract temperature.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values, axis=1), axis=2)

            # Create DataArray.
            da = xr.DataArray(arr, coords=[(c.DIM_TIME, time), (c.DIM_LON, [lon]), (c.DIM_LAT, [lat])])
            da.name = var.name
            da.attrs[c.ATTRS_SNAME] = "temperature"
            da.attrs[c.ATTRS_LNAME] = "temperature"
            da.attrs[c.ATTRS_UNITS] = c.UNIT_C
            da.attrs[c.ATTRS_GMAP] = "regular_lon_lat"
            da.attrs[c.ATTRS_COMMENTS] = "station data converted from degree C"

            # Create dataset.
            ds = da.to_dataset()

        # Precipitation, evaporation, evapotranspiration ---------------------------------------------------------------

        elif var.is_summable:

            # Extract variable and convert from mm to kg m-2 s-1.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values / c.SPD, axis=1), axis=2)

            # Create DataArray.
            da = xr.DataArray(arr, coords=[(c.DIM_TIME, time), (c.DIM_LON, [lon]), (c.DIM_LAT, [lat])])
            da.name = var.name
            da.attrs[c.ATTRS_SNAME] = "precipitation_flux"
            da.attrs[c.ATTRS_LNAME] = "Precipitation"
            da.attrs[c.ATTRS_UNITS] = c.UNIT_kg_m2s1
            da.attrs[c.ATTRS_GMAP] = "regular_lon_lat"
            da.attrs[c.ATTRS_COMMENTS] =\
                "station data converted from Total Precip (mm) using a density of 1000 kg/m³"

            # Create dataset.
            ds = da.to_dataset()

        # Wind ---------------------------------------------------------------------------------------------------------

        elif var.name in [c.V_UAS, c.V_VAS]:

            # Extract wind direction (dd).
            obs_dd = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("vv", axis=1)), index=time, columns=[stn])
            arr_dd = np.expand_dims(np.expand_dims(obs_dd[stn].values, axis=1), axis=2)
            da_dd = xr.DataArray(arr_dd, coords=[(c.DIM_TIME, time), (c.DIM_LON, [lon]), (c.DIM_LAT, [lat])])

            # Extract wind velocity (vv).
            obs_vv = pd.DataFrame(data=np.array(obs.iloc[:, 3:].drop("dd", axis=1)), index=time, columns=[stn])
            arr_vv = np.expand_dims(np.expand_dims(obs_vv[stn].values, axis=1), axis=2)
            da_vv = xr.DataArray(arr_vv, coords=[(c.DIM_TIME, time), (c.DIM_LON, [lon]), (c.DIM_LAT, [lat])])
            da_vv.attrs[c.ATTRS_UNITS] = c.UNIT_m_s

            # Calculate wind components.
            da_uas, da_vas = wf_utils.sfcwind_2_uas_vas(da_vv, da_dd)

            # Create DataArray.
            da = da_uas if var.name == c.V_UAS else da_vas
            da = xr.DataArray(da, coords=[(c.DIM_TIME, time), (c.DIM_LON, [lon]), (c.DIM_LAT, [lat])])
            da.name = var.name
            if var.name == c.V_UAS:
                da.attrs[c.ATTRS_SNAME] = "eastward_wind"
                da.attrs[c.ATTRS_LNAME] = "Eastward near-surface wind"
            else:
                da.attrs[c.ATTRS_SNAME] = "northward_wind"
                da.attrs[c.ATTRS_LNAME] = "Northward near-surface wind"
            da.attrs[c.ATTRS_UNITS] = c.UNIT_m_s
            da.attrs[c.ATTRS_GMAP]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        elif var.name == c.V_SFCWINDMAX:

            # Extract wind velocity.
            obs = pd.DataFrame(data=np.array(obs.iloc[:, 3:]), index=time, columns=[stn])
            arr = np.expand_dims(np.expand_dims(obs[stn].values, axis=1), axis=2)
            da = xr.DataArray(arr, coords=[(c.DIM_TIME, time), (c.DIM_LON, [lon]), (c.DIM_LAT, [lat])])

            # Create DataArray.
            da.name = var.name
            da.attrs[c.ATTRS_SNAME] = "wind"
            da.attrs[c.ATTRS_LNAME] = "near-surface wind"
            da.attrs[c.ATTRS_UNITS] = c.UNIT_m_s
            da.attrs[c.ATTRS_GMAP]  = "regular_lon_lat"

            # Create dataset.
            ds = da.to_dataset()

        else:
            ds = None

        # Wind, temperature or precipitation ---------------------------------------------------------------------------

        # Add attributes to lon, lat, time, elevation, and the grid.
        da      = xr.DataArray(np.full(len(time), np.nan), [(c.DIM_TIME, time)])
        da.name = "regular_lon_lat"
        da.attrs[c.ATTRS_GMAPNAME] = "lonlat"

        # Create dataset and add attributes.
        ds["regular_lon_lat"] = da
        ds.lon.attrs[c.ATTRS_SNAME]    = c.DIM_LONGITUDE
        ds.lon.attrs[c.ATTRS_LNAME]    = c.DIM_LONGITUDE
        ds.lon.attrs[c.ATTRS_UNITS]    = "degrees_east"
        ds.lon.attrs[c.ATTRS_AXIS]     = "X"
        ds.lat.attrs[c.ATTRS_SNAME]    = c.DIM_LATITUDE
        ds.lat.attrs[c.ATTRS_LNAME]    = c.DIM_LATITUDE
        ds.lat.attrs[c.ATTRS_UNITS]    = "degrees_north"
        ds.lat.attrs[c.ATTRS_AXIS]     = "Y"
        ds.attrs[c.ATTRS_STATION_NAME] = stn

        # Save data.
        p_stn = d_stn + var.name + "_" + ds.attrs[c.ATTRS_STATION_NAME] + c.F_EXT_NC
        desc = cntx.sep + c.CAT_OBS + cntx.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def preload_reanalysis_enacts(
    var_ra: VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine daily NetCDF files (one day per file) into annual single file (all days of one year per file).

    This is for the ENACTS ensemble.

    Parameters
    ----------
    var_ra: VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    # List NetCDF files.
    p_l = list(glob.glob(cntx.d_ra_day + var_ra.name + cntx.sep + "daily" + cntx.sep + "*" + c.F_EXT_NC))
    p_l.sort()

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
        if c.DIM_TIME not in ds.dims:
            date_str = re.split(r"[_|.]", os.path.basename(p))[id_token][0:8]
            time = pd.to_datetime(date_str[0:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
            da_time = xr.DataArray(time)
            ds[c.DIM_TIME] = da_time
            ds = ds.expand_dims(c.DIM_TIME)
            fu.save_netcdf(ds, p)

    # Combine files.
    for year in year_l:

        # Paths.
        p_pattern = cntx.d_ra_day + var_ra.name + cntx.sep + "daily" + cntx.sep + "*" + str(year) + "*" + c.F_EXT_NC
        p_l = list(glob.glob(p_pattern))
        p_l.sort()
        p = cntx.d_ra_day + var_ra.name + cntx.sep + var_ra.name + "_" + cntx.obs_src + "_day_" + str(year) + c.F_EXT_NC

        if (not os.path.exists(p)) or cntx.opt_force_overwrite:

            # Combine NetCDF files.
            ds = fu.open_netcdf(p_l, combine="nested", concat_dim=c.DIM_TIME).load()

            # Rename dimensions
            ds = ds.rename_dims({"Lon": c.DIM_LONGITUDE, "Lat": c.DIM_LATITUDE})
            ds[c.DIM_LONGITUDE] = ds["Lon"]
            ds[c.DIM_LATITUDE] = ds["Lat"]
            ds = ds.drop_vars(["Lon", "Lat"])
            ds[c.DIM_LONGITUDE].attrs["long_name"] = c.DIM_LONGITUDE
            ds[c.DIM_LATITUDE].attrs["long_name"] = c.DIM_LATITUDE
            ds[c.DIM_LONGITUDE].attrs["units"] = "degrees_east"
            ds[c.DIM_LATITUDE].attrs["units"] = "degrees_north"
            if var_ra.name not in ds.variables:
                if var_ra.name == c.V_ENACTS_RR:
                    da_name = "precip"
                else:
                    da_name = "temp"
                ds[var_ra.name] = ds[da_name]
                ds = ds.drop_vars([da_name])

            # Adjust units.
            if (var_ra.name in [c.V_ENACTS_TMIN, c.V_ENACTS_TMAX]) and\
               (c.UNIT_C in ds[var_ra.name].attrs[c.ATTRS_UNITS]):
                ds[var_ra.name] = ds[var_ra.name] + c.d_KC
                ds[var_ra.name].attrs[c.ATTRS_UNITS] = c.UNIT_K
            elif (var_ra.name in [c.V_ENACTS_RR, c.V_ENACTS_PET]) and\
                 (c.UNIT_mm in ds[var_ra.name].attrs[c.ATTRS_UNITS]):
                ds[var_ra.name] = ds[var_ra.name] / c.SPD
                ds[var_ra.name].attrs[c.ATTRS_UNITS] = c.UNIT_kg_m2s1

            # Sort/rename dimensions.
            ds = wf_utils.standardize_netcdf(ds, vi_name=var_ra.name)

            # Save combined datasets.
            fu.save_netcdf(ds, p, os.path.basename(p))


def load_reanalysis_ecmwf_enacts(
    var_ra: VarIdx
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Combine NetCDF files into a single file.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the call to fu.open_netcdf
    with a list of NetCDF files.

    This is compatible with ERA5, ERA5-Land and ENACTS ensembles.

    Parameters
    ----------
    var_ra: VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    var_name = VarIdx(var_ra.name).convert_name(c.ENS_CORDEX)

    # Paths.
    p_stn_l = list(glob.glob(cntx.d_ra_day + var_ra.name + cntx.sep + "*" + c.F_EXT_NC))
    p_stn_l.sort()
    p_stn = cntx.d_stn(var_name) + var_name + "_" + cntx.obs_src + c.F_EXT_NC
    d_stn = os.path.dirname(p_stn)
    if not (os.path.isdir(d_stn)):
        os.makedirs(d_stn)

    if (not os.path.exists(p_stn)) or cntx.opt_force_overwrite:

        # Combine datasets (the 'load' is necessary to apply the mask later).
        # ds = fu.open_netcdf(p_stn_l, combine="by_coords", concat_dim=const.dim_time).load()
        ds = fu.open_netcdf(p_stn_l, combine="nested", concat_dim=c.DIM_TIME)

        # Rename variables.
        if var_ra.name in [c.V_ECMWF_T2MMIN, c.V_ECMWF_T2MMAX]:
            var_ra_name = c.V_ECMWF_T2M
        elif var_ra.name in [c.V_ECMWF_U10MIN, c.V_ECMWF_U10MAX]:
            var_ra_name = c.V_ECMWF_U10
        elif var_ra.name in [c.V_ECMWF_V10MIN, c.V_ECMWF_V10MAX]:
            var_ra_name = c.V_ECMWF_V10
        elif var_ra.name == c.V_ECMWF_UV10MAX:
            var_ra_name = c.V_ECMWF_UV10
        else:
            var_ra_name = var_ra.name
        if (var_ra_name != var_ra.name) and (var_ra_name not in list(ds.variables)):
            var_ra_name = var_ra.name
        if var_name not in list(ds.variables):
            ds = ds.rename({var_ra_name: var_name})

        # Subset.
        ds = wf_utils.subset_lon_lat_time(ds, var_name, cntx.lon_bnds, cntx.lat_bnds)

        # Apply and create mask.
        if var_name != c.V_SFTLF:
            da_mask = xr.DataArray(fu.create_mask(ds if cntx.obs_src == c.ENS_ERA5_LAND else ""))
        else:
            da_mask = None
        if da_mask is not None:
            da = wf_utils.apply_mask(ds[var_name], da_mask)
            ds = da.to_dataset(name=var_name)

        # Units are set only for ERA5* reanalysis datasets.
        is_era5 = cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND]
        is_enacts = cntx.obs_src == c.ENS_ENACTS

        # Set attributes.
        ds[var_name].attrs[c.ATTRS_GMAP] = "regular_lon_lat"
        if var_name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
            ds[var_name].attrs[c.ATTRS_SNAME] = "temperature"
            ds[var_name].attrs[c.ATTRS_LNAME] = "Temperature"
            if is_enacts:
                ds = wf_utils.set_units(ds, var_name, c.UNIT_K)
            elif is_era5:
                ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_K
        elif var_ra.is_summable:
            if cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND]:
                ds[var_name] = ds[var_name] * 1000 / c.SPD
            if var_name == c.V_PR:
                ds[var_name].attrs[c.ATTRS_SNAME] = "precipitation_flux"
                ds[var_name].attrs[c.ATTRS_LNAME] = "Precipitation"
            elif var_name == c.V_EVSPSBL:
                ds[var_name].attrs[c.ATTRS_SNAME] = "evaporation_flux"
                ds[var_name].attrs[c.ATTRS_LNAME] = "Evaporation"
            elif var_name == c.V_EVSPSBLPOT:
                ds[var_name].attrs[c.ATTRS_SNAME] = "evapotranspiration_flux"
                ds[var_name].attrs[c.ATTRS_LNAME] = "Evapotranspiration"
            if is_era5:
                ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_kg_m2s1
        elif var_name in [c.V_UAS, c.V_VAS, c.V_SFCWINDMAX]:
            if var_name == c.V_UAS:
                ds[var_name].attrs[c.ATTRS_SNAME] = "eastward_wind"
                ds[var_name].attrs[c.ATTRS_LNAME] = "Eastward near-surface wind"
            elif var_name == c.V_VAS:
                ds[var_name].attrs[c.ATTRS_SNAME] = "northward_wind"
                ds[var_name].attrs[c.ATTRS_LNAME] = "Northward near-surface wind"
            else:
                ds[var_name].attrs[c.ATTRS_SNAME] = "wind"
                ds[var_name].attrs[c.ATTRS_LNAME] = "near-surface wind"
            if is_era5:
                ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_m_s
        elif var_name == c.V_RSDS:
            ds[var_name].attrs[c.ATTRS_SNAME] = "surface_solar_radiation_downwards"
            ds[var_name].attrs[c.ATTRS_LNAME] = "Surface solar radiation downwards"
            if is_era5:
                ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_J_m2
        elif var_name == c.V_HUSS:
            ds[var_name].attrs[c.ATTRS_SNAME] = "specific_humidity"
            ds[var_name].attrs[c.ATTRS_LNAME] = "Specific humidity"
            if is_era5:
                ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_1
        elif var_name == c.V_SFTLF:
            ds[var_name].attrs[c.ATTRS_SNAME] = "land_area_fraction"
            ds[var_name].attrs[c.ATTRS_LNAME] = "Land area fraction"
            ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_1

        # Change sign to have the same meaning between projections and reanalysis.
        # A positive sign for the following variable means that the transfer direction is from the surface toward the
        # atmosphere. A negative sign means that there is condensation.
        if (var_name in [c.V_EVSPSBL, c.V_EVSPSBLPOT]) and (cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND]):
            ds[var_name] = -ds[var_name]

        # Sort/rename dimensions.
        ds = wf_utils.standardize_netcdf(ds, vi_name=var_name)

        # Save NetCDF.
        desc = cntx.sep + c.CAT_OBS + cntx.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def load_reanalysis_chirps(
    var_ra: VarIdx
):
    """
    --------------------------------------------------------------------------------------------------------------------
    Combine NetCDF files into a single file.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the call to fu.open_netcdf
    with a list of NetCDF files.

    This is compatible with CHIRPS ensemble.

    Parameters
    ----------
    var_ra: VarIdx
        Variable (reanalysis).
    --------------------------------------------------------------------------------------------------------------------
    """

    var_name = VarIdx(var_ra.name).convert_name(c.ENS_CORDEX)

    # Paths.
    p_stn_l = list(glob.glob(cntx.d_ra_day + var_ra.name + cntx.sep + "*" + c.F_EXT_NC))
    p_stn_l.sort()
    p_stn = cntx.d_stn(var_name) + var_name + "_" + cntx.obs_src + c.F_EXT_NC
    d_stn = os.path.dirname(p_stn)
    if not (os.path.isdir(d_stn)):
        os.makedirs(d_stn)

    if (not os.path.exists(p_stn)) or cntx.opt_force_overwrite:

        # Combine datasets (the 'load' is necessary to apply the mask later).
        # ds = fu.open_netcdf(p_stn_l, combine="by_coords", concat_dim=const.dim_time).load()
        ds = fu.open_netcdf(p_stn_l, combine="nested", concat_dim=c.DIM_TIME)

        # Rename variables.
        var_ra_name = var_ra.name
        if var_name not in list(ds.variables):
            ds = ds.rename({var_ra_name: var_name})

        # Subset.
        ds = wf_utils.subset_lon_lat_time(ds, var_name, cntx.lon_bnds, cntx.lat_bnds)

        # Set attributes.
        ds[var_name].attrs[c.ATTRS_GMAP] = "regular_lon_lat"
        if var_ra.is_summable:
            ds[var_name] = ds[var_name] / c.SPD
            if var_name == c.V_PR:
                ds[var_name].attrs[c.ATTRS_SNAME] = "precipitation_flux"
                ds[var_name].attrs[c.ATTRS_LNAME] = "Precipitation"
            ds[var_name].attrs[c.ATTRS_UNITS] = c.UNIT_kg_m2s1

        # Sort/rename dimensions.
        ds = wf_utils.standardize_netcdf(ds, vi_name=var_name)

        # Save NetCDF.
        desc = cntx.sep + c.CAT_OBS + cntx.sep + os.path.basename(p_stn)
        fu.save_netcdf(ds, p_stn, desc=desc)


def extract(
    var: VarIdx,
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
    var: VarIdx
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
        lon_stn = round(float(ds_stn.lon.values), 1)
        lat_stn = round(float(ds_stn.lat.values), 1)
        lon_l = [lon_stn - cntx.radius, lon_stn + cntx.radius]
        lat_l = [lat_stn - cntx.radius, lat_stn + cntx.radius]

    # Reanalysis.
    # When using reanalysis data, need to include extra cells in case the resolution of the reanalysis dataset is lower
    # than the resolution of the projection dataset or if meshes are not overlapping perfectly.
    else:

        # Projections.
        # Must use xr.open_dataset here, otherwise there is a problem in parallel mode.
        p_proj = list(glob.glob(d_obs + var.name + cntx.sep + "*" + c.F_EXT_NC))[0]
        try:
            ds_proj = xr.open_dataset(p_proj).load()
        except xcv.MissingDimensionsError:
            ds_proj = xr.open_dataset(p_proj, drop_variables=["time_bnds"]).load()
        fu.close_netcdf(ds_proj)

        # Resolution of projections (based on the distance between coordinates, in each direction).
        res_proj_lon = abs(ds_proj.rlon.values[1] - ds_proj.rlon.values[0])
        res_proj_lat = abs(ds_proj.rlat.values[1] - ds_proj.rlat.values[0])

        # Resolution of reanalysis dataset.
        # Two different strategies are used for reanalysis data.
        # - If the grid has more than one cell in each direction, the approach used for the projections is used here.
        # - Otherwise, one of NetCDF files holding reanalysis data must be opened again, because it's impossible to
        #   determine resolution using a 1x1 grid.
        if (len(ds_stn.longitude.values) > 1) and (len(ds_stn.latitude.values) > 1):
            res_ra_lon = abs(ds_stn.longitude.values[1] - ds_stn.longitude.values[0])
            res_ra_lat = abs(ds_stn.latitude.values[1] - ds_stn.latitude.values[0])
        else:
            var_ra_name = var.convert_name(cntx.obs_src)
            p_stn_l = list(glob.glob(cntx.d_ra_day + var_ra_name + cntx.sep + "*" + c.F_EXT_NC))
            if len(p_stn_l) > 0:
                ds_stn = fu.open_netcdf(p_stn_l[0])
                res_ra_lon = abs(ds_stn.longitude.values[1] - ds_stn.longitude.values[0])
                res_ra_lat = abs(ds_stn.latitude.values[1] - ds_stn.latitude.values[0])
            else:
                res_ra_lon = res_proj_lon
                res_ra_lat = res_proj_lat

        # Calculate the number of points needed along each dimension.
        n_lon_ext = float(max(1.0, math.ceil(res_proj_lon / res_ra_lon)))
        n_lat_ext = float(max(1.0, math.ceil(res_proj_lat / res_ra_lat)))

        # Calculate extended boundaries.
        lon_l = [cntx.lon_bnds[0] - n_lon_ext * res_ra_lon, cntx.lon_bnds[1] + n_lon_ext * res_ra_lon]
        lat_l = [cntx.lat_bnds[0] - n_lat_ext * res_ra_lat, cntx.lat_bnds[1] + n_lat_ext * res_ra_lat]

    # Data extraction --------------------------------------------------------------------------------------------------

    # Patch: Create and discard an empty file to avoid stalling when writing the NetCDF file.
    #        It seems to wake up the hard disk.
    p_inc = p_raw.replace(c.F_EXT_NC, ".incomplete")
    if not os.path.exists(p_inc):
        open(p_inc, "a").close()
    if os.path.exists(p_inc):
        os.remove(p_inc)

    # List years.
    year_l = [min(min(cntx.per_ref), min(cntx.per_fut)), max(max(cntx.per_ref), max(cntx.per_fut))]

    # Find the data at the requested timestep.
    freq = "day" if var.name != c.V_SFTLF else "fx"
    p_obs = d_obs.replace(cntx.sep + "*" + cntx.sep, cntx.sep + freq + cntx.sep) +\
        var.name + cntx.sep + "*" + c.F_EXT_NC
    p_fut = d_fut.replace(cntx.sep + "*" + cntx.sep, cntx.sep + freq + cntx.sep) +\
        var.name + cntx.sep + "*" + c.F_EXT_NC
    p_l = sorted(glob.glob(p_obs)) + sorted(glob.glob(p_fut))

    # Open NetCDF.
    # Note: subset is not working with chunks.
    if var.name != c.V_SFTLF:
        ds_extract = fu.open_netcdf(p_l,
                                    chunks={c.DIM_TIME: 365},
                                    drop_variables=["time_vectors", "ts", "time_bnds"],
                                    combine="by_coords")
    else:
        ds_extract = fu.open_netcdf(p_l[0], drop_variables=["time_vectors", "ts", "time_bnds"])

    # Subset.
    ds_extract = wf_utils.subset_lon_lat_time(ds_extract, var.name, lon_l, lat_l, year_l)

    # Sort/rename dimensions.
    ds_extract = wf_utils.standardize_netcdf(ds_extract, vi_name=var.name)

    # Save NetCDF.
    desc = cntx.sep + c.CAT_RAW + cntx.sep + os.path.basename(p_raw)
    fu.save_netcdf(ds_extract, p_raw, desc=desc)


def interpolate(
    var: VarIdx,
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
    var: VarIdx
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

    # Load NetCDF.
    ds_raw = fu.open_netcdf(p_raw)

    # Sort/rename dimensions.
    ds_raw = wf_utils.standardize_netcdf(ds_raw, vi_name=var.name)

    msg = "Interpolating (time)"

    # This checks if the data is daily. If it is sub-daily, resample to daily. This can take a while, but should
    # save us computation time later.
    if ((var.name != c.V_SFTLF) and
        (ds_raw.time.isel(time=[0, 1]).diff(dim=c.DIM_TIME).values[0] <
         np.array([datetime.timedelta(1)], dtype="timedelta64[ms]")[0])):

        # Interpolate.
        fu.log(msg, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            ds_raw = ds_raw.resample(time="1D").mean(dim=c.DIM_TIME, keep_attrs=True)

        # Save NetCDF (raw) and load it.
        desc = cntx.sep + c.CAT_RAW + cntx.sep + os.path.basename(p_raw)
        fu.save_netcdf(ds_raw, p_raw, desc=desc)
        ds_raw = fu.open_netcdf(p_raw)

        # Sort/rename dimensions.
        ds_raw = wf_utils.standardize_netcdf(ds_raw, vi_name=var.name)

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

        # Sort/rename dimensions.
        ds_regrid = wf_utils.standardize_netcdf(ds_regrid, vi_name=var.name)

        # Save NetCDF (regrid).
        desc = cntx.sep + c.CAT_REGRID + cntx.sep + os.path.basename(p_regrid)
        fu.save_netcdf(ds_regrid, p_regrid, desc=desc)

    else:

        fu.log(msg + " (not required)", True)


def regrid(
    ds_data: xr.Dataset,
    ds_grid: xr.Dataset,
    var: VarIdx
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Perform grid change.

    Parameters
    ----------
    ds_data: xr.Dataset
        Dataset containing data.
    ds_grid: xr.Dataset
        Dataset containing grid
    var: VarIdx
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get longitude and latitude values (ds_data).
    data_lon_l = ds_data[var.name][c.DIM_LONGITUDE].values.ravel()
    data_lat_l = ds_data[var.name][c.DIM_LATITUDE].values.ravel()

    # Get longitude and latitude values (ds_grid).
    grid_lon_l = ds_grid[var.name][c.DIM_LONGITUDE].values.ravel()
    grid_lat_l = ds_grid[var.name][c.DIM_LATITUDE].values.ravel()

    # Remove grid points for which there is no data (not comprised within the data domain).
    # This is useful if data domain (derived from climatie simulations) is smaller than grid domain (derived from
    # reanalysis data), or not perfectly centered on the grid.
    for lon in grid_lon_l:
        if (lon < min(data_lon_l)) or (lon > max(data_lon_l)):
            grid_lon_l.remove(lon)
    for lat in grid_lat_l:
        if (lat < min(data_lat_l)) or (lat > max(data_lat_l)):
            grid_lat_l.remove(lat)

    # Create a Dataset for the new grid.
    ds_grid = xr.Dataset({c.DIM_LATITUDE: ([c.DIM_LATITUDE], grid_lat_l),
                          c.DIM_LONGITUDE: ([c.DIM_LONGITUDE], grid_lon_l)})

    # Interpolate.
    regridder = xe.Regridder(ds_data, ds_grid, "bilinear")
    da_regrid = regridder(ds_data[var.name])

    # Apply and create mask.
    da_mask = xr.DataArray(fu.create_mask())
    if da_mask is not None:
        da_regrid = wf_utils.apply_mask(da_regrid, da_mask)

    # Create dataset.
    ds_regrid = da_regrid.to_dataset(name=var.name)
    ds_regrid[var.name].attrs[c.ATTRS_UNITS] = ds_data[var.name].attrs[c.ATTRS_UNITS]

    return ds_regrid


def perturbate(
    ds: xr.Dataset,
    var: VarIdx
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Add small perturbation.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset.
    var: VarIdx
        Variable.

    Returns
    -------
    xr.Dataset
        Perturbed dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Get perturbation value.
    d_val = 1e-12
    if cntx.opt_bias_perturb is not None:
        for i in range(len(cntx.opt_bias_perturb)):
            if var.name == cntx.opt_bias_perturb[i][0]:
                d_val = float(cntx.opt_bias_perturb[i][1])
                break

    # Data array has a single dimension.
    if len(ds[var.name].dims) == 1:
        ds[var.name].values = ds[var.name].values + np.random.rand(ds[var.name].values.shape[0]) * d_val

    # Data array has 3 dimensions, with unknown ranking.
    else:
        vals = ds[var.name].values.shape[0]
        i_time = list(ds[var.name].dims).index(c.DIM_TIME)
        ds[var.name].values = ds[var.name].values +\
            np.random.rand(vals if i_time == 0 else 1, vals if i_time == 1 else 1, vals if i_time == 2 else 1) * d_val

    return ds


def preprocess(
    var: VarIdx,
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
    var: VarIdx
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

    # Load NetCDF.
    ds_fut = fu.open_netcdf(p_regrid)

    # Sort/rename dimensions.
    ds_fut = wf_utils.standardize_netcdf(ds_fut, vi_name=var.name)

    # Observations -----------------------------------------------------------------------------------------------------

    if (not os.path.exists(p_obs)) or cntx.opt_force_overwrite:

        # Drop February 29th and select reference period.
        ds_obs = wf_utils.remove_feb29(ds_stn)
        ds_obs = wf_utils.sel_period(ds_obs, cntx.per_ref)

        # Save NetCDF.
        desc = cntx.sep + c.CAT_OBS + cntx.sep + os.path.basename(p_obs)
        fu.save_netcdf(ds_obs, p_obs, desc=desc)

    # Simulated climate (future period) --------------------------------------------------------------------------------

    if os.path.exists(p_regrid_fut) and (not cntx.opt_force_overwrite):

        # Load NetCDF.
        ds_regrid_fut = fu.open_netcdf(p_regrid_fut)

        # Sort/rename dimensions.
        ds_regrid_fut = wf_utils.standardize_netcdf(ds_regrid_fut, vi_name=var.name)

    else:

        # Drop February 29th.
        if c.DIM_TIME in ds_fut.dims:
            ds_regrid_fut = wf_utils.remove_feb29(ds_fut)
        else:
            ds_regrid_fut = ds_fut

        # Adjust values that do not make sense.
        if var.is_summable or (var.name == c.V_CLT):
            ds_regrid_fut[var.name].values[ds_regrid_fut[var.name] < 0] = 0
            if var.name == c.V_CLT:
                ds_regrid_fut[var.name].values[ds_regrid_fut[var.name] > 100] = 100

        # Add small perturbation.
        if var.is_summable:
            perturbate(ds_regrid_fut, var)

        # Convert to a 365-day calendar.
        if c.DIM_TIME in ds_fut.dims:
            ds_regrid_fut = wf_utils.convert_to_365_calender(ds_regrid_fut)

        # Save NetCDF.
        desc = cntx.sep + c.CAT_REGRID + cntx.sep + os.path.basename(p_regrid_fut)
        fu.save_netcdf(ds_regrid_fut, p_regrid_fut, desc=desc)

    # Simulated climate (reference period) -----------------------------------------------------------------------------

    if (not os.path.exists(p_regrid_ref)) or cntx.opt_force_overwrite:

        # Select reference period.
        if c.DIM_TIME in ds_fut.dims:
            ds_regrid_ref = wf_utils.sel_period(ds_regrid_fut, cntx.per_ref)
        else:
            ds_regrid_ref = ds_regrid_fut

        if var.is_summable or (var.name == c.V_CLT):
            pos = np.where(np.squeeze(ds_regrid_ref[var.name].values) > 0.01)[0]
            ds_regrid_ref[var.name][pos] = 1e-12

        # Save NetCDF.
        desc = cntx.sep + c.CAT_REGRID + cntx.sep + os.path.basename(p_regrid_ref)
        fu.save_netcdf(ds_regrid_ref, p_regrid_ref, desc=desc)


def postprocess(
    var: VarIdx,
    nq: int,
    up_qmf: float,
    time_win: int,
    ds_stn: xr.Dataset,
    p_obs: str,
    p_sim: str,
    p_sim_adj: str,
    p_qmf: str,
    title: Optional[str] = "",
    p_fig: Optional[str] = "",
    p_ts_fig: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Performs post-processing.

    Parameters
    ----------
    var: VarIdx
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
    title: Optional[str]
        Title of figure.
    p_fig: Optional[str]
        Path of figure (large plot).
    p_ts_fig: Optional[str]
        Path of figure (time series).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Load datasets.
    # 'p_stn' and 'p_obs' cannot be opened using xr.open_mfdataset.
    if not cntx.opt_ra:
        if var.name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
            da_stn_attrs = ds_stn[var.name].attrs
            ds_stn[var.name] = ds_stn[var.name] + c.d_KC
            ds_stn[var.name].attrs = da_stn_attrs
            ds_stn[var.name].attrs[c.ATTRS_UNITS] = c.UNIT_K
        da_stn = ds_stn[var.name][:, 0, 0]
    else:
        da_stn = ds_stn[var.name]

    # Load NetCDFs (observation and simulation).
    ds_obs = fu.open_netcdf(p_obs)
    ds_sim = fu.open_netcdf(p_sim)

    # Sort/rename dimensions.
    ds_obs = wf_utils.standardize_netcdf(ds_obs, vi_name=var.name)
    ds_sim = wf_utils.standardize_netcdf(ds_sim, vi_name=var.name)

    da_obs = ds_obs[var.name]
    da_sim = ds_sim[var.name]
    ds_sim_adj = None

    # Simulation -------------------------------------------------------------------------------------------------------

    # Convert to a 365-day calendar.
    da_sim_365 = wf_utils.convert_to_365_calender(ds_sim)[var.name]

    # Observation ------------------------------------------------------------------------------------------------------

    if var.is_summable:
        kind = c.KIND_MULT
    elif var.name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
        kind = c.KIND_ADD
    else:
        kind = c.KIND_ADD

    # Interpolate.
    if not cntx.opt_ra:
        da_stn = da_stn.interpolate_na(dim=c.DIM_TIME)

    # Quantile Mapping Function ----------------------------------------------------------------------------------------

    # Load transfer function.
    if (p_qmf != "") and os.path.exists(p_qmf) and (not cntx.opt_force_overwrite):

        # Load NetCDF.
        chunks = {c.DIM_TIME: 1} if cntx.use_chunks else None
        ds_qmf = fu.open_netcdf(p_qmf, chunks=chunks)

    # Calculate transfer function.
    else:

        da_qmf = xr.DataArray(train(da_obs, da_stn, nq, c.GROUP, kind, time_win, detrend_order=c.DETREND_ORDER))
        if var.is_summable:
            da_qmf.values[da_qmf > up_qmf] = up_qmf
            da_qmf.values[da_qmf < -up_qmf] = -up_qmf
        ds_qmf = da_qmf.to_dataset(name=var.name)
        ds_qmf[var.name].attrs[c.ATTRS_GROUP] = da_qmf.attrs[c.ATTRS_GROUP]
        ds_qmf[var.name].attrs[c.ATTRS_KIND]  = da_qmf.attrs[c.ATTRS_KIND]
        ds_qmf[var.name].attrs[c.ATTRS_UNITS] = da_obs.attrs[c.ATTRS_UNITS]
        if p_qmf != "":
            desc = cntx.sep + c.CAT_QMF + cntx.sep + os.path.basename(p_qmf)
            fu.save_netcdf(ds_qmf, p_qmf, desc=desc)
            ds_qmf = fu.open_netcdf(p_qmf)

    # Quantile Mapping -------------------------------------------------------------------------------------------------

    # Load quantile mapping.
    if (p_sim_adj != "") and os.path.exists(p_sim_adj) and (not cntx.opt_force_overwrite):

        # Load NetCDF.
        chunks = {c.DIM_TIME: 1} if cntx.use_chunks else None
        ds_sim_adj = fu.open_netcdf(p_sim_adj, chunks=chunks)

    # Apply transfer function.
    else:

        try:

            # Adjust simulation.
            interp = (True if not cntx.opt_ra else False)
            da_sim_adj = xr.DataArray(predict(da_sim_365, ds_qmf[var.name], interp=interp,
                                              detrend_order=c.DETREND_ORDER))
            ds_sim_adj = da_sim_adj.to_dataset(name=var.name)
            del ds_sim_adj[var.name].attrs[c.ATTRS_BIAS]
            ds_sim_adj[var.name].attrs[c.ATTRS_SNAME] = da_stn.attrs[c.ATTRS_SNAME]
            ds_sim_adj[var.name].attrs[c.ATTRS_LNAME] = da_stn.attrs[c.ATTRS_LNAME]
            ds_sim_adj[var.name].attrs[c.ATTRS_UNITS] = da_stn.attrs[c.ATTRS_UNITS]
            if c.ATTRS_GMAP in da_stn.attrs:
                ds_sim_adj[var.name].attrs[c.ATTRS_GMAP]  = da_stn.attrs[c.ATTRS_GMAP]

            # Save NetCDF (adjusted simulation) and open it.
            if p_sim_adj != "":
                desc = cntx.sep + c.CAT_QQMAP + cntx.sep + os.path.basename(p_sim_adj)
                fu.save_netcdf(ds_sim_adj, p_sim_adj, desc=desc)
                ds_sim_adj = fu.open_netcdf(p_sim_adj)

        except ValueError as err:
            fu.log("Failed to create QQMAP NetCDF file.", True)
            fu.log(format(err), True)
            pass

    # Sort/rename dimensions.
    if ds_sim_adj is not None:
        ds_sim_adj = wf_utils.standardize_netcdf(ds_sim_adj, vi_name=var.name)

    # Create plots -----------------------------------------------------------------------------------------------------

    if p_fig != "":

        # Select reference period.
        ds_sim_adj_ref = wf_utils.sel_period(ds_sim_adj, cntx.per_ref)

        # Convert to data arrays.
        da_sim_adj_ref = ds_sim_adj_ref[var.name]
        da_sim_adj     = ds_sim_adj[var.name]
        da_qmf         = ds_qmf[var.name]

        # Convert units.
        if var.is_summable:
            da_stn         = wf_utils.set_units(da_stn, var.name, c.UNIT_mm)
            da_obs         = wf_utils.set_units(da_obs, var.name, c.UNIT_mm)
            da_sim         = wf_utils.set_units(da_sim, var.name, c.UNIT_mm)
            da_sim_adj     = wf_utils.set_units(da_sim_adj, var.name, c.UNIT_mm)
            da_sim_adj_ref = wf_utils.set_units(da_sim_adj_ref, var.name, c.UNIT_mm)
            da_qmf         = wf_utils.set_units(da_qmf, var.name, c.UNIT_mm)
        elif var.name in [c.V_TAS, c.V_TASMIN, c.V_TASMAX]:
            da_stn         = wf_utils.set_units(da_stn, var.name, c.UNIT_C)
            da_obs         = wf_utils.set_units(da_obs, var.name, c.UNIT_C)
            da_sim         = wf_utils.set_units(da_sim, var.name, c.UNIT_C)
            da_sim_adj     = wf_utils.set_units(da_sim_adj, var.name, c.UNIT_C)
            da_sim_adj_ref = wf_utils.set_units(da_sim_adj_ref, var.name, c.UNIT_C)

        # Select center coordinates.
        da_stn_xy         = da_stn
        da_obs_xy         = da_obs
        da_sim_xy         = da_sim
        da_sim_adj_ref_xy = da_sim_adj_ref
        da_sim_adj_xy     = da_sim_adj
        da_qmf_xy         = da_qmf
        if cntx.opt_ra:
            if cntx.p_bounds == "":
                da_stn_xy         = wf_utils.subset_ctrl_pt(da_stn_xy)
                da_obs_xy         = wf_utils.subset_ctrl_pt(da_obs_xy)
                da_sim_xy         = wf_utils.subset_ctrl_pt(da_sim_xy)
                da_sim_adj_xy     = wf_utils.subset_ctrl_pt(da_sim_adj_xy)
                da_sim_adj_ref_xy = wf_utils.subset_ctrl_pt(da_sim_adj_ref_xy)
                da_qmf_xy         = wf_utils.subset_ctrl_pt(da_qmf_xy)
            else:
                da_stn_xy         = wf_utils.squeeze_lon_lat(da_stn_xy)
                da_obs_xy         = wf_utils.squeeze_lon_lat(da_obs_xy)
                da_sim_xy         = wf_utils.squeeze_lon_lat(da_sim_xy)
                da_sim_adj_xy     = wf_utils.squeeze_lon_lat(da_sim_adj_xy)
                da_sim_adj_ref_xy = wf_utils.squeeze_lon_lat(da_sim_adj_ref_xy)
                da_qmf_xy         = wf_utils.squeeze_lon_lat(da_qmf_xy)

        if cntx.opt_diagnostic:

            # Update context.
            # The colormap of temperature anomalies is used here, independently of the real variable.
            cntx.delta = Delta("True")
            cntx.varidx = VarIdx(c.V_TAS)

            # Generate summary plot.
            wf_plot.plot_bias(da_stn_xy, da_obs_xy, da_sim_xy, da_sim_adj_xy, da_sim_adj_ref_xy, da_qmf_xy,
                              var, title, p_fig)

            # Generate time series only.
            wf_plot.plot_bias_ts(da_stn_xy, da_sim_xy, da_sim_adj_xy, var, title, p_ts_fig)

    return ds_sim_adj if (p_fig == "") else None


def bias_adj(
    stn: str,
    var: VarIdx,
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
    var: VarIdx
        Variable.
    sim_name: str
        Simulation name.
    calc_err: bool
        If True, only calculate the error (will not work properly in parallel mode).
    -------------------------------------------------------------------------------------------------------------------
    """

    # List regrid files.
    d_regrid = cntx.d_scen(c.CAT_REGRID, var.name)
    p_regrid_l = fu.list_files(d_regrid)
    if p_regrid_l is None:
        fu.log("The required files are not available.", True)
        return
    p_regrid_l = [i for i in p_regrid_l if "_4qqmap" not in i]
    p_regrid_l.sort()

    # Loop through simulation sets.
    for i in range(len(p_regrid_l)):
        p_regrid_tokens = p_regrid_l[i].split(cntx.sep)
        sim_name_i = p_regrid_tokens[len(p_regrid_tokens) - 1].replace(var.name + "_", "").replace(c.F_EXT_NC, "")

        # Skip iteration if it does not correspond to the specified simulation name.
        if (sim_name != "") and (sim_name != sim_name_i):
            continue

        # NetCDF files.
        p_stn        = cntx.p_stn(var.name, stn)
        p_regrid     = p_regrid_l[i]
        p_regrid_ref = p_regrid.replace(c.F_EXT_NC, "_ref_4qqmap" + c.F_EXT_NC)
        p_regrid_fut = p_regrid.replace(c.F_EXT_NC, "_4qqmap" + c.F_EXT_NC)
        p_qqmap      = p_regrid.replace(c.CAT_REGRID, c.CAT_QQMAP)
        p_qmf        = p_regrid.replace(c.CAT_REGRID, c.CAT_QMF)

        # Verify if required files exist.
        msg = "File missing: "
        if ((not os.path.exists(p_stn)) or (not os.path.exists(p_regrid_ref)) or
            (not os.path.exists(p_regrid_fut))) and\
           (not cntx.opt_force_overwrite):
            if not(os.path.exists(p_stn)):
                fu.log(msg + os.path.basename(p_stn), True)
            if not(os.path.exists(p_regrid_ref)):
                fu.log(msg + os.path.basename(p_regrid_ref), True)
            if not(os.path.exists(p_regrid_fut)):
                fu.log(msg + os.path.basename(p_regrid_fut), True)
            continue

        # Load NetCDF (station), drop February 29th, sort/rename dimensions, and select reference period.
        ds_stn = fu.open_netcdf(p_stn)
        ds_stn = wf_utils.standardize_netcdf(ds_stn, vi_name=var.name)
        ds_stn = wf_utils.remove_feb29(ds_stn)
        ds_stn = wf_utils.sel_period(ds_stn, cntx.per_ref)

        # Path of CSV and PNG files (large plot).
        fn_fig = var.name + "_" + sim_name_i + "_" + c.CAT_FIG_BIAS + c.F_EXT_PNG
        p_fig = cntx.d_fig(c.CAT_FIG_BIAS, var.name) + fn_fig
        p_csv = p_fig.replace(cntx.sep + var.name + cntx.sep,
                              cntx.sep + var.name + "_" + c.F_CSV + cntx.sep).\
            replace(c.F_EXT_PNG, c.F_EXT_CSV)

        # Path of CSV and PNG files (time series).
        p_ts_fig = p_fig.replace(c.F_EXT_PNG, "_ts" + c.F_EXT_PNG)
        p_ts_csv = p_csv.replace(c.F_EXT_CSV, "_ts" + c.F_EXT_CSV)

        # Verify if all CSV files related to the large plot exist.
        p_csv_exists = True
        stats_l = [c.STAT_MEAN] + [c.STAT_CENTILE] * len(cntx.opt_bias_centiles)
        centiles_l = [-1] + cntx.opt_bias_centiles
        for j in range(len(stats_l)):

            # Create a statistic instance.
            stat = Stat(stats_l[j], centiles_l[j])
            if stat.centile == 0:
                stat = Stat(c.STAT_MIN)
            elif stat.centile == 1:
                stat = Stat(c.STAT_MAX)

            # Path of CSV file related to the statistic.
            if stat.code in [c.STAT_MEAN, c.STAT_MIN, c.STAT_MAX, c.STAT_SUM]:
                suffix = stat.code
            else:
                suffix = stat.centile_as_str

            # Verify if this file exists
            p_csv_i = p_csv.replace(c.F_EXT_CSV, "_" + suffix + c.F_EXT_CSV)
            if not os.path.exists(p_csv_i):
                p_csv_exists = False
                break

        # Bias adjustment ----------------------------------------------------------------------------------

        if not calc_err:

            msg = "Step #5bc Statistical downscaling and adjusting bias"
            if (not os.path.exists(p_fig)) or \
               ((not p_csv_exists) and (c.F_CSV in cntx.opt_diagnostic_format)) or \
               (not os.path.exists(p_ts_fig)) or \
               ((not os.path.exists(p_ts_csv)) and (c.F_CSV in cntx.opt_diagnostic_format)) or \
               (not os.path.exists(p_qqmap)) or \
               (not os.path.exists(p_qmf)) or \
               cntx.opt_force_overwrite:

                # Calculate QQ and generate bias adjustment plots.
                fu.log(msg, True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    postprocess(var, cntx.opt_bias_nq, cntx.opt_bias_up_qmf, cntx.opt_bias_time_win,
                                ds_stn, p_regrid_ref, p_regrid_fut, p_qqmap, p_qmf, sim_name_i, p_fig,
                                p_ts_fig)
            else:
                fu.log(msg + " (not required)", True)

        # Bias error ---------------------------------------------------------------------------------------

        elif os.path.exists(p_qqmap):

            # Extract the reference period from the adjusted simulation.
            ds_qqmap_ref = fu.open_netcdf(p_qqmap)
            ds_qqmap_ref = wf_utils.standardize_netcdf(ds_qqmap_ref, vi_name=var.name)
            ds_qqmap_ref = wf_utils.remove_feb29(ds_qqmap_ref)
            ds_qqmap_ref = wf_utils.sel_period(ds_qqmap_ref, cntx.per_ref)

            # Calculate the error between observations and simulation for the reference period.
            bias_err_current = float(round(wf_utils.calc_error(ds_stn[var.name], ds_qqmap_ref[var.name]), 4))

            # Set bias adjustment parameters (nq, up_qmf and time_win) and calculate error according to the
            # selected method.
            col_names = ["nq", "up_qmf", "time_win", "bias_err"]
            col_values = [float(cntx.opt_bias_nq), cntx.opt_bias_up_qmf, float(cntx.opt_bias_time_win),
                          bias_err_current]
            row = (cntx.opt_bias_df["sim_name"] == sim_name) &\
                  (cntx.opt_bias_df["stn"] == stn) &\
                  (cntx.opt_bias_df["var"] == var.name)
            cntx.opt_bias_df.loc[row, col_names] = col_values
            fu.save_csv(cntx.opt_bias_df, cntx.opt_bias_fn)


def init_bias_params():

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize bias adjustment parameters.
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

    # Attempt loading the bias adjustment file.
    if os.path.exists(cntx.opt_bias_fn):
        cntx.opt_bias_df = pd.read_csv(cntx.opt_bias_fn)
        if len(cntx.opt_bias_df) > 0:
            sim_name_l = list(cntx.opt_bias_df["sim_name"])
            stn_l      = list(cntx.opt_bias_df["stn"])
            vi_code_l  = list(cntx.opt_bias_df["var"])
            nq_l       = list(cntx.opt_bias_df["nq"])
            up_qmf_l   = list(cntx.opt_bias_df["up_qmf"])
            time_win_l = list(cntx.opt_bias_df["time_win"])
            bias_err_l = list(cntx.opt_bias_df["bias_err"])
        fu.log("Bias adjustment file loaded.", True)

    # List CORDEX files.
    list_cordex = fu.list_cordex(cntx.d_proj, cntx.rcps.code_l, "day")

    # Stations.
    stns = cntx.stns
    if cntx.opt_ra:
        stns = [cntx.obs_src]

    # List simulation names, stations and variables.
    for rcp in cntx.rcps.items:
        sim_l = list_cordex[rcp.code]
        sim_l.sort()
        for i_sim in range(0, len(sim_l)):
            list_i = list_cordex[rcp.code][i_sim].split(cntx.sep)
            sim_name = list_i[cntx.rank_inst()] + "_" + list_i[cntx.rank_inst() + 1]
            for stn in stns:
                for var in cntx.vars.items:

                    # Skip if the variable is the mask.
                    if var.name == c.V_SFTLF:
                        continue

                    # Add the combination if it does not already exist.
                    if (cntx.opt_bias_df is None) or\
                       (len(cntx.opt_bias_df.loc[(cntx.opt_bias_df["sim_name"] == sim_name) &
                                                 (cntx.opt_bias_df["stn"] == stn) &
                                                 (cntx.opt_bias_df["var"] == var.name)]) == 0):
                        sim_name_l.append(sim_name)
                        stn_l.append(stn)
                        vi_code_l.append(var.name)
                        nq_l.append(cntx.opt_bias_nq)
                        up_qmf_l.append(cntx.opt_bias_up_qmf)
                        time_win_l.append(cntx.opt_bias_time_win)
                        bias_err_l.append(-1)

                        # Update dataframe.
                        cntx.opt_bias_df =\
                            build_df(sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Build dataframe.
    cntx.opt_bias_df = build_df(sim_name_l, stn_l, vi_code_l, nq_l, up_qmf_l, time_win_l, bias_err_l)

    # Save bias adjustment statistics to a CSV file.
    if cntx.opt_bias_fn != "":

        # Create directory if it does not already exist.
        d = os.path.dirname(cntx.opt_bias_fn)
        if not (os.path.isdir(d)):
            os.makedirs(d)

        # Save file.
        fu.save_csv(cntx.opt_bias_df, cntx.opt_bias_fn)
        if os.path.exists(cntx.opt_bias_fn):
            fu.log("Bias adjustment file created or updated.", True)


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
    d_exec = cntx.d_scen("", "")
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
        for var_ra in VarIdxs(cntx.vars_ra.code_l).items:
            if cntx.obs_src == c.ENS_ENACTS:
                preload_reanalysis_enacts(var_ra)
            if cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND, c.ENS_ENACTS]:
                load_reanalysis_ecmwf_enacts(var_ra)
            elif cntx.obs_src == c.ENS_CHIRPS:
                load_reanalysis_chirps(var_ra)

    fu.log("=")
    fu.log("Step #3-5 Producing climate scenarios")

    # Loop through variables.
    for var in cntx.vars.items:

        # List directories potentially containing CORDEX files (but not necessarily for all selected variables).
        list_cordex = fu.list_cordex(cntx.d_proj, cntx.rcps.code_l, "fx" if var.name == c.V_SFTLF else "day")

        # Select file names for observation (or reanalysis).
        if not cntx.opt_ra:
            d_stn = cntx.d_stn(var.name)
            p_stn_l = glob.glob(d_stn + "*" + c.F_EXT_NC)
            p_stn_l.sort()
        else:
            p_stn_l = [cntx.d_stn(var.name) + var.name + "_" + cntx.obs_src + c.F_EXT_NC]

        # Loop through stations.
        for i_stn in range(0, len(p_stn_l)):

            # Station name.
            p_stn = p_stn_l[i_stn]
            if not cntx.opt_ra:
                stn = os.path.basename(p_stn).replace(c.F_EXT_NC, "").replace(var.name + "_", "")
                if not (stn in cntx.stns):
                    continue
            else:
                stn = cntx.obs_src

            # Directories.
            d_stn             = cntx.d_stn(var.name)
            d_obs             = cntx.d_scen(c.CAT_OBS, var.name)
            d_raw             = cntx.d_scen(c.CAT_RAW, var.name)
            d_regrid          = cntx.d_scen(c.CAT_REGRID, var.name)
            d_qqmap           = cntx.d_scen(c.CAT_QQMAP, var.name)
            d_qmf             = cntx.d_scen(c.CAT_QMF, var.name)
            d_fig_bias        = cntx.d_fig(c.CAT_FIG_BIAS, var.name)
            d_fig_postprocess = cntx.d_fig(c.CAT_FIG_POSTPROCESS, var.name)
            d_fig_workflow    = cntx.d_fig(c.CAT_FIG_WORKFLOW, var.name)

            # Load station data, drop February 29th and select reference period.
            ds_stn = fu.open_netcdf(p_stn)
            ds_stn = wf_utils.standardize_netcdf(ds_stn, vi_name=var.name)
            ds_stn = wf_utils.remove_feb29(ds_stn)
            if var.name == c.V_SFTLF:
                ds_stn = ds_stn * 0 + 1
            else:
                ds_stn = wf_utils.sel_period(ds_stn, cntx.per_ref)

            # Create directories (required because of parallel processing).
            if not (os.path.isdir(d_stn)):
                os.makedirs(d_stn)
            if (not os.path.isdir(d_obs)) and (var.name != c.V_SFTLF):
                os.makedirs(d_obs)
            if not (os.path.isdir(d_raw)):
                os.makedirs(d_raw)
            if not (os.path.isdir(d_regrid)):
                os.makedirs(d_regrid)
            if (not os.path.isdir(d_qmf)) and (var.name != c.V_SFTLF):
                os.makedirs(d_qmf)
            if (not os.path.isdir(d_qqmap)) and (var.name != c.V_SFTLF):
                os.makedirs(d_qqmap)
            if var.name != c.V_SFTLF:
                if not (os.path.isdir(d_fig_bias)):
                    os.makedirs(d_fig_bias)
                if not (os.path.isdir(d_fig_postprocess)):
                    os.makedirs(d_fig_postprocess)
                if not (os.path.isdir(d_fig_workflow)):
                    os.makedirs(d_fig_workflow)

            # Loop through RCPs.
            for rcp in cntx.rcps.items:

                # Extract and sort simulations lists.
                list_cordex_ref = list_cordex[rcp.code + "_historical"]
                list_cordex_fut = list_cordex[rcp.code]
                list_cordex_ref.sort()
                list_cordex_fut.sort()
                n_sim = len(list_cordex_ref)

                fu.log("Processing: " + var.name + ", " + stn + ", " + rcp.code, True)

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
                        n_sim_proc_before = len(list(glob.glob(d_qqmap + "*" + c.F_EXT_NC)))

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
                        n_sim_proc_after = len(list(glob.glob(d_qqmap + "*" + c.F_EXT_NC)))

                        # If no simulation has been processed during a loop iteration, this means that the work is done.
                        if (cntx.n_proc == 1) or (n_sim_proc_before == n_sim_proc_after):
                            break

                # Calculate bias adjustment errors.
                if var.name != c.V_SFTLF:
                    for i_sim in range(n_sim):
                        tokens = list_cordex_fut[i_sim].split(cntx.sep)
                        sim_name = tokens[cntx.rank_inst()] + "_" + tokens[cntx.rank_gcm()]
                        bias_adj(stn, var, sim_name, True)


def gen_single(
    list_cordex_ref: [str],
    list_cordex_fut: [str],
    ds_stn: xr.Dataset,
    d_raw: str,
    var: VarIdx,
    stn: str,
    rcp: RCP,
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
    var: VarIdx
        Variable.
    stn: str
        Station.
    rcp: RCP
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
    fu.log("RCP        : " + rcp.code)
    fu.log("Simulation : " + sim_name)
    fu.log("=")

    # Skip iteration if the variable 'var' is not available in the current directory.
    p_sim_ref_l = list(glob.glob(d_sim_ref + cntx.sep + var.name + cntx.sep + "*" + c.F_EXT_NC))
    p_sim_fut_l = list(glob.glob(d_sim_fut + cntx.sep + var.name + cntx.sep + "*" + c.F_EXT_NC))

    if (len(p_sim_ref_l) == 0) or (len(p_sim_fut_l) == 0):
        fu.log("Skipping iteration: data not available for simulation-variable.", True)
        if cntx.n_proc > 1:
            fu.log("Work done!", True)
        return

    # Files within CORDEX or CORDEX-NA.
    if "cordex" in d_sim_fut.lower():
        p_raw = d_raw + var.name + "_" + tokens[cntx.rank_inst()] + "_" +\
                tokens[cntx.rank_gcm()].replace("*", "_") + c.F_EXT_NC
    elif len(d_sim_fut) == 3:
        p_raw = d_raw + var.name + "_Ouranos_" + d_sim_fut + c.F_EXT_NC
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
    p_regrid     = p_raw.replace(c.CAT_RAW, c.CAT_REGRID)
    p_qqmap      = p_raw.replace(c.CAT_RAW, c.CAT_QQMAP)
    p_qmf        = p_raw.replace(c.CAT_RAW, c.CAT_QMF)
    p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + c.CAT_QQMAP + c.F_EXT_NC
    p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + c.CAT_QQMAP + c.F_EXT_NC
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
    if (((not(os.path.isfile(p_obs)) or not(os.path.isfile(p_regrid_ref)) or not(os.path.isfile(p_regrid_fut))) or
         cntx.opt_force_overwrite) and (var.name != c.V_SFTLF)):
        fu.log(msg)
        preprocess(var, ds_stn, p_obs, p_regrid, p_regrid_ref, p_regrid_fut)
    else:
        fu.log(msg + " (not required)")

    # Step #5: Post-processing.

    # Step #5a: Calculate adjustment factors.
    fu.log("-")
    msg = "Step #5a  Calculating adjustment factors"
    if cntx.opt_bias and (var.name != c.V_SFTLF):
        fu.log(msg)
        bias_adj(stn, var, sim_name)
    else:
        fu.log(msg + " (not required)")
    df_sel = cntx.opt_bias_df.loc[(cntx.opt_bias_df["sim_name"] == sim_name) &
                                  (cntx.opt_bias_df["stn"] == stn) &
                                  (cntx.opt_bias_df["var"] == var.name)]
    if (df_sel is not None) and (len(df_sel) > 0):
        nq       = float(df_sel["nq"])
        up_qmf   = float(df_sel["up_qmf"])
        time_win = float(df_sel["time_win"])
    else:
        nq       = float(cntx.opt_bias_nq)
        up_qmf   = float(cntx.opt_bias_up_qmf)
        time_win = float(cntx.opt_bias_time_win)

    # Step #5b: Statistical downscaling.
    # Step #5c: Bias correction.
    # This creates one .nc file in ~/sim_climat/<country>/<project>/<stn>/qqmap/<var>/.
    fu.log("-")
    msg = "Step #5bc Statistical downscaling and adjusting bias"
    if ((not(os.path.isfile(p_qqmap)) or not(os.path.isfile(p_qmf))) or cntx.opt_force_overwrite) and\
       (var.name != c.V_SFTLF):
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

    Parameters
    ----------
    func_name: str
        Name of function to be called.
    view_code: Optional[str]
        View code.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List of variables to analyze.
    var_code_l = cntx.vars.code_l
    if c.V_SFTLF in var_code_l:
        var_code_l.remove(c.V_SFTLF)

    # Number of variables to process.
    n_var = len(var_code_l)

    # Scalar mode.
    if (cntx.n_proc == 1) or (n_var < 2):
        for i_var in range(n_var):
            if func_name == "calc_diag_cycle":
                calc_diag_cycle(var_code_l, i_var)
            elif func_name == "stats.calc_map":
                stats.calc_map(var_code_l, i_var)
            elif func_name == "stats.calc_ts":
                stats.calc_ts(view_code, var_code_l, i_var)
            else:
                stats.calc_tbl(var_code_l, i_var)

    # Parallel processing mode.
    else:

        for i in range(math.ceil(n_var / cntx.n_proc)):

            # Select variables to process in the current loop.
            i_first = i * cntx.n_proc
            n_proc = min(cntx.n_proc, n_var)
            i_last = i_first + n_proc - 1
            var_name_l = var_code_l[i_first:(i_last + 1)]
            var_name_l.remove(c.V_SFTLF)

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
                    func = functools.partial(stats.calc_tbl, var_name_l)
                pool.map(func, list(range(len(var_name_l))))
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
    var = VarIdx(var_name)

    # Update context.
    cntx.code = c.PLATFORM_SCRIPT
    cntx.view = View(c.VIEW_CYCLE)
    cntx.lib = Lib(c.LIB_MAT)
    cntx.delta = Delta("False")
    cntx.varidx = VarIdx(var.name)

    # Loop through stations.
    stns = (cntx.stns if not cntx.opt_ra else [cntx.obs_src])
    for stn in stns:

        fu.log("Processing: " + var_name + ", " + stn, True)

        # Path ofo NetCDF file containing station data.
        p_obs = cntx.p_obs(stn, var_name)

        # Loop through raw NetCDF files.
        p_raw_l = list(glob.glob(cntx.d_scen(c.CAT_RAW, var_name) + "*" + c.F_EXT_NC))
        for i in range(len(p_raw_l)):
            p_raw = p_raw_l[i]

            # Path of NetCDF files.
            p_regrid = p_raw.replace(c.CAT_RAW, c.CAT_REGRID)
            p_qqmap = p_raw.replace(c.CAT_RAW, c.CAT_QQMAP)
            p_regrid_ref = p_regrid[0:len(p_regrid) - 3] + "_ref_4" + c.CAT_QQMAP + c.F_EXT_NC
            p_regrid_fut = p_regrid[0:len(p_regrid) - 3] + "_4" + c.CAT_QQMAP + c.F_EXT_NC

            # File name.
            fn_fig = p_regrid_fut.split(cntx.sep)[-1].replace("_4qqmap" + c.F_EXT_NC, "_<per>_<cat_fig>" + c.F_EXT_PNG)

            # Generate diagnostic plots.
            if cntx.opt_diagnostic and (len(cntx.opt_diagnostic_format) > 0):

                # Plot title.
                title = fn_fig.replace(c.F_EXT_PNG, "").replace("_<per>", "").replace("_<cat_fig>", "")

                # This creates one file:
                #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/postprocess/<var>/<hor>/*.png
                p_fig = cntx.d_fig(c.CAT_FIG_POSTPROCESS, var_name) + fn_fig.replace("<per>_", "")
                p_fig = p_fig.replace("_<per>", "").replace("<cat_fig>", c.CAT_FIG_POSTPROCESS)
                stats.calc_postprocess(p_obs, p_regrid_fut, p_qqmap, var, p_fig, title)

                # This creates one file:
                #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/workflow/<var>/*.png
                p_fig = cntx.d_fig(c.CAT_FIG_WORKFLOW, var_name) + fn_fig.replace("<per>_", "")
                p_fig = p_fig.replace("_<per>", "").replace("<cat_fig>", c.CAT_FIG_WORKFLOW)
                stats.calc_workflow(var, p_regrid_ref, p_regrid_fut, p_fig, title)

            # Generate monthly and daily plots.
            if cntx.opt_cycle and (len(cntx.opt_cycle_format) > 0):

                # Load NetCDF and sort dimensions.
                ds_qqmap = fu.open_netcdf(p_qqmap)
                ds_qqmap = wf_utils.standardize_netcdf(ds_qqmap, vi_name=var.name)

                # Loop through horizons.
                for per in cntx.per_hors:
                    per_str = str(per[0]) + "_" + str(per[1])

                    # Update context.
                    sim_code = os.path.basename(p_qqmap).replace(c.F_EXT_NC, "").replace(var_name + "_", "")
                    cntx.rcp = RCP("")
                    cntx.sim = Sim(sim_code)
                    cntx.hor = Hor(per)

                    # This creates 2 files:
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>/<hor>/*.png
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ma/<var>_csv/<hor>/*.csv
                    title = fn_fig.replace(c.F_EXT_PNG, "").replace("<per>", per_str).\
                        replace("<cat_fig>", c.VIEW_CYCLE_MS)
                    stats.calc_cycle(ds_qqmap, stn, var, per, c.FREQ_MS, title)

                    # This creates 2 files:
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>/<hor>/*.png
                    #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>_csv/<hor>/*.csv
                    title = fn_fig.replace(c.F_EXT_PNG, "").replace("<per>", per_str).\
                        replace("<cat_fig>", c.VIEW_CYCLE_D)
                    stats.calc_cycle(ds_qqmap, stn, var, per, c.FREQ_D, title)

        if os.path.exists(p_obs) and cntx.opt_cycle and (len(cntx.opt_cycle_format) > 0):

            # Load data.
            ds_obs = fu.open_netcdf(p_obs)
            ds_obs = wf_utils.standardize_netcdf(ds_obs, vi_name=var.name)
            per_str = str(cntx.per_ref[0]) + "_" + str(cntx.per_ref[1])

            # Update context.
            cntx.rcp = RCP(c.REF)
            cntx.sim = Sim("")
            cntx.hor = Hor(cntx.per_ref)

            # This creates 2 files:
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>/*.png
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_ms/<var>_csv/*.csv
            title = var_name + "_" + c.REF + "_" + per_str + "_" + c.VIEW_CYCLE_MS
            stats.calc_cycle(ds_obs, stn, var, cntx.per_ref, c.FREQ_MS, title)

            # This creates 2 files:
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen/cycle_d/<var>/*.png
            #     ~/sim_climat/<country>/<project>/<stn>/fig/scen_cycle_d/<var>_csv/*.csv
            title = var_name + "_" + c.REF + "_" + per_str + "_" + c.VIEW_CYCLE_D
            stats.calc_cycle(ds_obs, stn, var, cntx.per_ref, c.FREQ_D, title)


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

    fu.log("-")
    msg = "Step #7   Converting NetCDF to CSV files (scenarios)"
    if cntx.export_nc_to_csv[0] and not cntx.opt_ra:
        fu.log(msg)
        fu.log("-")
        stats.conv_nc_csv(c.CAT_STN)
        stats.conv_nc_csv(c.CAT_SCEN)
    else:
        fu.log(msg + not_req)

    # Results ----------------------------------------------------------------------------------------------------------

    fu.log("=")
    msg = "Step #8a  Generating post-process, workflow and cycle plots (scenarios)"
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
        gen_per_var("stats.calc_ts", c.VIEW_TS)
    else:
        fu.log(msg + not_req)

    fu.log("=")
    msg = "Step #8c  Calculating statistics tables (scenarios)"
    if cntx.opt_tbl[0]:
        fu.log(msg)
        gen_per_var("stats.calc_tbl")
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #8d  Generating time series (scenarios, raw values and bias)"
    if cntx.opt_ts_bias and (len(cntx.opt_ts_format) > 0):
        fu.log(msg)
        gen_per_var("stats.calc_ts", c.VIEW_TS_BIAS)
    else:
        fu.log(msg + not_req)

    fu.log("-")
    msg = "Step #8e  Generating maps (scenarios)"
    if cntx.opt_ra and cntx.opt_map[0] and (len(cntx.opt_ts_format) > 0):
        fu.log(msg)
        gen_per_var("stats.calc_map")
    else:
        fu.log(msg + " (not required)")

    fu.log("-")
    msg = "Step #8f  Generating cluster plots (scenarios)"
    if cntx.opt_cluster and (len(cntx.opt_cluster_format) > 0):
        fu.log(msg)
        stats.calc_clusters()
    else:
        fu.log(msg + " (not required)")


def run_bias():

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
