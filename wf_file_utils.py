# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions for file manipulation.
#
# Contact information:
# 1. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020-2022 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import altair as alt
import glob
import holoviews as hv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import sys
import xarray as xr
import xarray.core.variable as xcv
import xesmf as xe
import warnings
from distutils import dir_util
from typing import Union, List, Optional

# Workflow libraries.
import wf_utils
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
sys.path.append("scen_workflow_afr_dashboard")
from scen_workflow_afr_dashboard import dash_plot, cl_delta, cl_hor, cl_lib, cl_rcp, cl_varidx, cl_view


def info_cordex():

    """
    --------------------------------------------------------------------------------------------------------------------
     Creates an array that contains information about CORDEX simulations.

     A verification is made to ensure that there is at least one dataset file available for each variable.

    Return
    ------
    sets : [str, str, str, str, [str]]
        List of simulation sets.
        sets[0] is the institute that created this simulation.
        sets[1] is the regional circulation model (RCM).
        sets[2] is the global circulation model (GCM).
        sets[3] is the simulation name.
        sets[4] is a list of variables available.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Results.
    sets = []

    # List simulation files.
    p_sim_l = []
    for domain in cntx.domains:
        p_sim_l = p_sim_l + list(glob.glob(cntx.p_proj(domain=domain)))

    # Loop through simulations.
    for p_sim_i in p_sim_l:
        n_token  = len(p_sim_i.split(cntx.sep)) - 2
        tokens_i = p_sim_i.split(cntx.sep)

        # Extract institute, RGM, CGM and emission scenario.
        inst = tokens_i[n_token + 1]
        rgm  = tokens_i[n_token + 2]
        cgm  = tokens_i[n_token + 3].split("_")[1]
        scen = tokens_i[n_token + 3].split("_")[2]

        # Extract variables and ensure that there is at least one datset file available for each one.
        vars_i = []

        for p_sim_j in list(glob.glob(p_sim_i + "*" + cntx.sep)):
            n_files = len(glob.glob(p_sim_j + "*" + c.F_EXT_NC))
            if n_files > 0:
                tokens_j = p_sim_j.split(cntx.sep)
                var      = tokens_j[len(tokens_j) - 2]
                vars_i.append(var)

        sets.append([inst, rgm, cgm, scen, vars_i])

    return sets


def list_files(
    p: str
) -> [str]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Lists data files (NetCDF or Zarr) contained in a directory.

    Parameters
    ----------
    p: str
        Path of directory.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List.
    p_l = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for p in f:
            p_new = ""
            if c.F_EXT_NC in p:
                p_new = os.path.join(r, p)
            elif c.F_EXT_ZARR in p:
                p_new = os.path.dirname(r)
            if (p_new != "") and (p_new not in p_l):
                p_l.append(p_new)

    # Sort.
    p_l.sort()

    return p_l


def create_mask(
    template: Optional[Union[str, xr.Dataset]] = ""
) -> xr.DataArray:

    """
    --------------------------------------------------------------------------------------------------------------------
    Create a mask based on an existing layer for the current simulation, reanalysis data, or simulation data.

    The behaviour of the current function depends on the ensemble and climate variable/index.
    - A mask is not required if 'nan' values are present over the sea (ex: c.ENS_ENACTS, c.ENS_CHIRPS).
    - Sometimes, 'nan' values are given for a variable, but unavailable for another variable of the same ensemble.
    - The mask can be used if available (ex: ERA5 or CORDEX), or it can be derived from another variable
      (ex: ERA5-Land).
    - The order of attemps is related to resolution that can be achieved (ascending for resolution), hoping to get the
      finest details near the coasts.

    Parameters
    ----------
    template: Optional[Union[str, xr.Dataset]]
        Option #1: Path of datset file used to build the mask.
        Option #2: Dataset used to build the mask.
        For both options, if the minimum and maximum values are identical (considering all time stepts), we can assume
        that values in these cells should be 'nan'.

    Notes
    -----
    The layer corresponding to c.V_SFTLF (CORDEX) contains values between 0 and 100, whereas
    the layer correspondong to c.V_ECMWF_LSM (ERA5) contains values between 0 and 1.

    Table. Values over sea in recognized ensembles:
    Ensemble         Variable       Value            Action
    -------------------------------------------------------------------------------------------------------------
    CORDEX           Temperature    value            Nothing
                     Precipitation  value            Nothing
    ECMWF.ERA5       Temperature    value            Apply mask using c.V_ECMWF_LSM or c.V_SFTLF.
                     Precipitation  value            Apply mask using c.V_ECMWF_LSM or c.V_SFTLF.
    ECMWF.ERA5-Land  Temperature    np.nan           Nothing
                     Precipitation  value            Apply mask based on the variable itself (min=max).
    ENACTS           Temperature    np.nan or value  Apply mask using c.V_SFTLF.
                     Precipitation  np.nan or value  Apply mask using c.V_SFTLF.
    CHIRPS           Precipitation  np.nan           Nothing
    -------------------------------------------------------------------------------------------------------------
    Any              Index          value            Apply mask based on the first variable used to compute this
                                                     index.

    Returns
    -------
    xr.DataArray
        Mask containing either 1 (over land) or np.nan (elsewhere).
    --------------------------------------------------------------------------------------------------------------------
    """

    da_mask = None

    # Apply a mask based on a climate scenario or index.
    if template != "":

        # Open dataset file.
        if isinstance(template, str):
            ds = open_dataset(template)
        else:
            ds = template

        # Create mask.
        var_name = list(ds.data_vars)[0]
        da_min = ds[var_name].min(dim={c.DIM_TIME})
        da_max = ds[var_name].max(dim={c.DIM_TIME})
        da_nan = xr.DataArray(da_min == da_max).astype(int)
        da_mask = xr.where((da_nan == 1) | (da_min.isnull()), np.nan, 1)

    # Apply a mask based on ECMWF data (c.V_ECMWF_LSM or c.V_ECMWF_T2M).
    if (cntx.ens_ref_grid in [c.ENS_ERA5, c.ENS_ERA5_LAND]) and (da_mask is None):

        # Sometimes, the temperature variable sometimes include 'nan' values over the sea, whereas values for another
        # variable are provided and incorrect. This is why there is an attempt to consider any temperature variable.
        for var_name in [c.V_SFTLF, c.V_TAS]:

            # List data files.
            f_l = glob.glob(cntx.d_ref() + "*" + cntx.sep + var_name + "*" + c.F_EXT_NC)
            f_l.sort()
            if len(f_l) > 0:

                # Open dataset file.
                ds = open_dataset(f_l[0])
                var_name = list(ds.data_vars)[0]

                # Create mask.
                if var_name == c.V_SFTLF:
                    da_mask = ds[var_name][0]
                else:
                    da_mask = ds[var_name][0] * 0 + 1
                da_mask = xr.where(da_mask == 0, np.nan, da_mask)

            if (da_mask is not None) or (cntx.ens_ref_grid == c.ENS_ERA5):
                break

    # Apply a mask based on CORDEX data (c.V_SFTLF).
    if ((cntx.ens_ref_grid in [c.ENS_ERA5, c.ENS_ERA5_LAND]) and (da_mask is None)) or\
       (cntx.ens_ref_grid == c.ENS_ENACTS):

        # List data files related to land area fraction.
        var_name = c.V_SFTLF
        f_l = glob.glob(cntx.p_scen(c.CAT_REGRID, var_name, sim_code="*"))
        f_l.sort()
        if len(f_l) > 0:

            # Open dataset file.
            ds = open_dataset(f_l[0])

            # Create mask.
            da_mask = xr.DataArray(ds[var_name] > 0).astype(float)
            da_mask = xr.where(da_mask == 0, np.nan, da_mask)

    return da_mask


def open_dataset(
    p: Union[str, List[str]],
    drop_variables: [str] = None,
    chunks: Union[int, dict] = None,
    combine: str = None,
    concat_dim: str = None,
    desc: str = ""
) -> xr.Dataset:

    """
    --------------------------------------------------------------------------------------------------------------------
    Open dataset file(s) (NetCDF or ZARR).

    Parameters
    ----------
    p: Union[str, [str]]
        Path of file to be created.
    drop_variables: [str]
        Drop-variables parameter.
    chunks: Union[int,dict]
        Chunks parameter
    combine: str
        Combine parameter.
    concat_dim: str
        Concatenate dimension.
    desc: str
        Description.

    Returns
    -------
    xr.Dataset
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine the engine.
    engine = None
    if (isinstance(p, str) and (c.F_EXT_ZARR in p)) or \
       ((type(p) == list) and (len(p) > 0) and (c.F_EXT_ZARR in p[0])):
        engine = c.F_ZARR

    if desc == "":
        desc = (os.path.basename(p) if isinstance(p, str) else os.path.basename(p[0]))

    if cntx.opt_trace:
        log("Opening dataset: " + desc, True)

    # There is a single file to open.
    if isinstance(p, str):

        # Open file normally.
        ds = xr.open_dataset(p, drop_variables=drop_variables, engine=engine).load()
        close_netcdf(ds)

        # Determine the number of chunks.
        if cntx.use_chunks and (cntx.n_proc == 1) and (chunks is None) and (c.CAT_SCEN in p) and\
           (c.DIM_TIME in ds.dims):
            chunks = {c.DIM_TIME: len(ds[c.DIM_TIME])}

        # Reopen file using chunks.
        if chunks is not None:
            ds = xr.open_dataset(p, drop_variables=drop_variables, chunks=chunks, engine=engine).copy(deep=True).load()
            close_netcdf(ds)

    # There are multiple files to open.
    else:
        ds = xr.open_mfdataset(p, drop_variables=drop_variables, chunks=chunks, combine=combine,
                               concat_dim=concat_dim, lock=False, engine=engine).load()
        close_netcdf(ds)

    if cntx.opt_trace:
        log("Opened dataset", True)

    return ds


def save_dataset(
    ds: Union[xr.Dataset, xr.DataArray],
    p: str,
    desc: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a dataset or a data array to a dataset file (NetCDF or Zarr).

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset.
    p: str
        Path of file to be created.
    desc: str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine the engine.
    engine = "netcdf4"
    if c.F_EXT_ZARR in p:
        engine = c.F_ZARR

    if desc == "":
        desc = os.path.basename(p)

    if cntx.opt_trace:
        log("Saving dataset: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.exists(d)):
        os.makedirs(d)

    # Create a temporary file to indicate that writing is in progress.
    p_inc = p.replace(c.F_EXT_NC if engine == "netcdf4" else c.F_EXT_ZARR, ".incomplete")
    if not os.path.exists(p_inc):
        open(p_inc, "a").close()

    # Discard file if it already exists.
    if os.path.exists(p):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)

    # Save dataset.
    mode = "w"
    if engine == "netcdf4":
        ds.to_netcdf(p, mode=mode)
    else:
        if c.DIM_TIME in ds.dims:
            ds = ds.chunk({c.DIM_TIME: 365})
        try:
            ds.to_zarr(p, mode=mode)
        except NotImplementedError:
            ds.to_zarr(p, mode=mode, safe_chunks=False)

    # Discard the temporary file.
    if os.path.exists(p_inc):
        os.remove(p_inc)

    if cntx.opt_trace:
        log("Saved dataset", True)


def close_netcdf(
    ds: Union[xr.Dataset, xr.DataArray]
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Close a dataset file (NetCDF or Zarr).

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    if ds is not None:
        try:
            ds.close()
        finally:
            pass


def grid_transform(
    ds_da: Union[xr.Dataset, xr.DataArray],
    vi_name: str,
    lon_l: List[float] = None,
    lat_l: List[float] = None
) -> Union[xr.Dataset, xr.DataArray, None]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert from rotated to regular coordinates.

    The assumption is that the following coordintes are available : rlon, rlat, lon, lat.

    This is done in 2 steps:
    - Transformation from rotated to regular coordinates.
    - Regridding.

    Parameters
    ----------
    ds_da: Union[xr.Dataset, xr.DataArray]
        Dataset.
    vi_name: str
        Variable or index name.
    lon_l: List[float]
        Longitude values.
    lat_l: List[float]
        Latitude values.

    Returns
    -------
    Union[xr.Dataset, xr.DataArray, None]
        Dataset.
    --------------------------------------------------------------------------------------------------------------------
    """

    maps_required = False

    # Function that calculates cell size.
    def cell_size(
        ds: xr.Dataset
    ) -> List[float]:

        # Extract coordinates.
        if c.DIM_RLON in ds.dims:
            _lon_l = ds[c.DIM_RLON].values
            _lat_l = ds[c.DIM_RLAT].values
        else:
            _lon_l = ds[c.DIM_LON].values.mean(axis=0)
            _lat_l = ds[c.DIM_LAT].transpose().values.mean(axis=0)

        # Calculate distance between adjacent coordinates.
        _d_lon = [_lon_l[i + 1] - _lon_l[i] for i in range(len(_lon_l) - 1)]
        _d_lat = [_lat_l[i + 1] - _lat_l[i] for i in range(len(_lat_l) - 1)]

        return [float(np.mean(_d_lon)), float(np.mean(_d_lat))]

    # Calculate cell size.
    cs = cell_size(ds_da)

    # Create a DataArray for the current grid.
    if maps_required:
        rot_ctr_l = [ds_da[c.ATTRS_ROT_LAT_LON].attrs[c.ATTRS_G_NP_LON],
                     ds_da[c.ATTRS_ROT_LAT_LON].attrs[c.ATTRS_G_NP_LAT]]
        bottom_left = list(grid_transform_pt([min(lon_l), min(lat_l)], [rot_ctr_l[0], rot_ctr_l[1]], 1))
        top_right = list(grid_transform_pt([max(lon_l), max(lat_l)], [rot_ctr_l[0], rot_ctr_l[1]], 1))
        bottom_right = list(grid_transform_pt([max(lon_l), min(lat_l)], [rot_ctr_l[0], rot_ctr_l[1]], 1))
        top_left = list(grid_transform_pt([min(lon_l), max(lat_l)], [rot_ctr_l[0], rot_ctr_l[1]], 1))
        lat_rot_l = [min(bottom_left[1], bottom_right[1]), max(top_left[1], top_right[1])]
        lat_rot_l.sort()
        lon_rot_l = [min(bottom_left[0], bottom_right[0]), max(top_left[0], top_right[0])]
        lon_rot_l.sort()
        da_rot = ds_da.sel(rlat=slice(lat_rot_l[0], lat_rot_l[1]), rlon=slice(lon_rot_l[0], lon_rot_l[1]))[vi_name]
        da_rot[c.DIM_LONGITUDE] = da_rot[c.DIM_LON]
        da_rot[c.DIM_LATITUDE] = da_rot[c.DIM_LAT]
    else:
        da_rot = None

    # Generate coordinates between .
    def gen_coord_l(
        cs_coord: float,
        coord_bnds_l: List[float]
    ) -> List[float]:

        # Calculate the distance between the first and last coordinates.
        dist = coord_bnds_l[1] - coord_bnds_l[0]

        # Calculate the first coordinate, and ensure that the nodes are centered relative to boundaries.
        coord_0 = coord_bnds_l[0] - ((dist / float(cs_coord)) - (dist // float(cs_coord))) / 2

        # Determine the number of coordinates.
        n_coords = math.ceil(dist / cs_coord) + 1

        # List new coordinates.
        coord_new_l = [coord_0 + i * cs_coord for i in range(n_coords)]

        return coord_new_l

    # Create a DataArray for the new grid.
    grid_lat_l = gen_coord_l(cs[1], lat_l)
    grid_lon_l = gen_coord_l(cs[0], lon_l)
    ds_grid = xr.Dataset({c.DIM_LAT: ([c.DIM_LAT], grid_lat_l),
                          c.DIM_LON: ([c.DIM_LON], grid_lon_l)})
    regridder = xe.Regridder(ds_da, ds_grid, "bilinear")
    da_reg = regridder(ds_da[vi_name])
    da_reg = da_reg.rename({c.DIM_LAT: c.DIM_LATITUDE, c.DIM_LON: c.DIM_LONGITUDE})

    def gen_map(
        da: xr.DataArray,
        p: str
    ):

        # Calculate mean values over the reference period.
        ds_map = wf_utils.sel_period(da.to_dataset(name=vi_name), cntx.per_ref)
        da_map = ds_map[vi_name].mean(dim=c.DIM_TIME)
        da_map.attrs = da.attrs

        # Determine if the dataset as a rotated coordinate system.
        has_rotated_coords = wf_utils.has_rotated_coords(da_map, vi_name)

        # Determine the number of coordinates required.
        if has_rotated_coords:
            _n_lon = len(da_map[c.DIM_RLON].values)
            _n_lat = len(da_map[c.DIM_RLAT].values)
        else:
            _n_lon = len(da_map[c.DIM_LONGITUDE].values)
            _n_lat = len(da_map[c.DIM_LATITUDE].values)

        # Collect coordinates and values.
        arr_lon, arr_lat, arr_val = [], [], []
        for m in range(_n_lon):
            for n in range(_n_lat):
                if has_rotated_coords:
                    arr_lon.append(float(da_map.longitude.values[n, m]))
                    arr_lat.append(float(da_map.latitude.values[n, m]))
                else:
                    arr_lon.append(float(da_map.longitude.values[m]))
                    arr_lat.append(float(da_map.latitude.values[n]))
                arr_val.append(float(da_map.values[n, m]))

        # Generated DataFrame.
        dict_pd = {c.DIM_LONGITUDE: arr_lon, c.DIM_LATITUDE: arr_lat, "val": arr_val}
        df = pd.DataFrame(dict_pd)

        # Calculate range.
        z_range = [min(df["val"]), max(df["val"])]

        # Generate end export map.
        plot = dash_plot.gen_map(df, z_range)
        save_plot(plot, p)

    if maps_required:

        # Update context.
        cntx.lib = cl_lib.Lib(c.LIB_HV)
        cntx.delta = cl_delta.Delta("False")
        cntx.hor = cl_hor.Hor(cntx.per_ref)
        cntx.rcp = cl_rcp.RCP(c.REF)
        cntx.varidx = cl_varidx.VarIdx(vi_name)
        cntx.view = cl_view.View(c.VIEW_MAP)

        # Generate a pre-rotation map.
        gen_map(da_rot, "/home/yrousseau/Downloads/grid_transform_1.png")

        # Generate a post-rotation map.
        gen_map(da_reg, "/home/yrousseau/Downloads/grid_transform_2.png")

        exit()

    # Copy attributes.
    da_reg.name = vi_name
    ds_reg = da_reg.to_dataset()
    ds_reg.attrs = ds_da.attrs
    ds_reg[vi_name].attrs = ds_da[vi_name].attrs
    if c.ATTRS_UNITS in ds_da[vi_name].attrs:
        ds_reg[vi_name].attrs[c.ATTRS_UNITS] = ds_da[vi_name].attrs[c.ATTRS_UNITS]

    return ds_reg


def grid_transform_pt(
    coords: List[float],
    coords_rot_ctr: List[float],
    option: int,
) -> List[float]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert from regular to rotated coordinates, or the opposite.

    Parameters
    ----------
    coords: List[float]
        Longitude and latitude (original).
    coords_rot_ctr: List[float]
        Longitude and latitude of the rotation center.
    option: int
        Option: 1= regular to rotated; 2= rotated to regular.

    Returns
    -------
    List[float]
        Longitude and latitude (modified).
    --------------------------------------------------------------------------------------------------------------------
    """

    # Longitude and latitude.
    lon = coords[0]
    lat = coords[1]

    # Convert degrees to radians.
    lon = lon * math.pi / 180
    lat = lat * math.pi / 180

    # Rotation around the y-axis.
    theta = 90 + coords_rot_ctr[1]

    # Rotation around the z-axis.
    phi = coords_rot_ctr[0]

    # Convert degrees to radians.
    phi = phi * math.pi / 180
    theta = theta * math.pi / 180

    # Convert from spherical to cartesian coordinates.
    x = math.cos(lon) * math.cos(lat)
    y = math.sin(lon) * math.cos(lat)
    z = math.sin(lat)

    # Regular to rotated.
    if option == 1:
        x_new = math.cos(theta) * math.cos(phi) * x + math.cos(theta) * math.sin(phi) * y + math.sin(theta) * z
        y_new = -math.sin(phi) * x + math.cos(phi) * y
        z_new = -math.sin(theta) * math.cos(phi) * x - math.sin(theta) * math.sin(phi) * y + math.cos(theta) * z

    # Rotated to regular.
    else:
        phi = -phi
        theta = -theta
        x_new = math.cos(theta) * math.cos(phi) * x + math.sin(phi) * y + math.sin(theta) * math.cos(phi) * z
        y_new = -math.cos(theta) * math.sin(phi) * x + math.cos(phi) * y - math.sin(theta) * math.sin(phi) * z
        z_new = -math.sin(theta) * x + math.cos(theta) * z

    # Convert cartesian back to spherical coordinates.
    lon_new = math.atan2(y_new, x_new)
    lat_new = math.asin(z_new)

    # Convert radians back to degrees.
    lon_new = (lon_new * 180) / math.pi
    lat_new = (lat_new * 180) / math.pi

    return [lon_new, lat_new]


def clean_dataset(
    d: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Remove incomplete datasets.

    A .nc file or .zarr directory will be removed if there's an .incomplete file with the same name. The .incomplete
    file is removed as well. This is done to avoid having potentially incomplete data files, which can result in a
    crash.

    Parameters
    ----------
    d: str
        Base directory to search from.
    --------------------------------------------------------------------------------------------------------------------
    """

    msg = "Cleaning datasets: "
    if cntx.d_ref() in d:
        msg += "~" + cntx.sep + "stn" + cntx.sep + "..." + cntx.sep + d.replace(cntx.d_ref(), "")
    else:
        msg += "~" + cntx.sep + "res" + cntx.sep + "..." + cntx.sep + d.replace(cntx.d_res, "")
    msg += "*"
    log(msg, True)

    # List temporary files.
    if d[len(d) - 1] != cntx.sep:
        d = d + cntx.sep
    p_inc_l = glob.glob(d + "**" + cntx.sep + "*.incomplete", recursive=True)

    # Loop through temporary files.
    for p_inc in p_inc_l:

        # Attempt removing an associated dataset file.
        p_nc = p_inc.replace(".incomplete", c.F_EXT_NC)
        p_zarr = p_inc.replace(".incomplete", c.F_EXT_ZARR)
        if os.path.exists(p_nc):
            log("Removing: " + p_nc)
            os.remove(p_nc)
        if os.path.exists(p_zarr):
            log("Removing: " + p_zarr)
            shutil.rmtree(p_zarr)

        # Remove the temporary file.
        if os.path.exists(p_inc):
            log("Removing: " + p_inc)
            os.remove(p_inc)


def crop_dataset(
    p_in: str,
    keyword_l: List[str],
    lon_l: List[float],
    lat_l: List[float],
    p_out: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Subset datasets based on a list of keywords and a box describing the range of longitudes and latitudes of interest.

    Parameters
    ----------
    p_in: str
        Path of input file.
    keyword_l: List[str]
        List of keywords.
    lon_l: List[float]
        List of longitudes.
    lat_l: List[float]
        List of latitudes.
    p_out: str
        Path of output file.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Try to find one of the keywords in the path.
    keyword_found = False
    for keyword in keyword_l:
        if keyword in p_in:
            keyword_found = True
            break

    # Skip iteration if the file to create already exists or if no keyword is comprised in the path.
    if ((os.path.exists(p_out)) and not cntx.opt_force_overwrite) or not keyword_found:
        return

    # Load dataset.
    # The except clause is necessary in the case there are two time dimensions (in CORDEX).
    try:
        ds = open_dataset(p_in).load()
    except xcv.MissingDimensionsError:
        ds = open_dataset(p_in, drop_variables=["time_bnds"]).load()

    # Get variable name.
    vi_name = ""
    for item in list(ds.variables):
        if (item in c.V_CORDEX) or (item in c.V_ECMWF) or (item in c.V_ENACTS) or (item in c.V_CHIRPS):
            vi_name = item
            break

    # Crop.
    ds_i_out = wf_utils.subset_lon_lat_time(ds, vi_name, lon_l, lat_l)

    # Save dataset.
    save_dataset(ds_i_out, p_out)


def save_plot(
    plot: Union[alt.Chart, any, plt.Figure],
    p: str,
    desc: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a plot to a file.

    Parameters
    ----------
    plot: Union[alt.Chart, any, plt.Figure]
        Plot.
    p: str
        Path of file to be created.
    desc: str
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cntx.opt_trace:
        log("Saving plot: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save file.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Library: matplot.
        if isinstance(plot, plt.Figure):
            plot.savefig(p)

        # Library: altair (not tested).
        elif isinstance(plot, alt.Chart):
            plot.save(p)

        # Library: hvplot.
        else:
            hv.extension("bokeh")
            renderer = hv.renderer("bokeh")
            if c.F_HTML in cntx.opt_map_format:
                renderer.save(plot, p.replace(c.F_EXT_PNG, ""))
            renderer.save(plot, p.replace(c.F_EXT_PNG, ""), fmt=c.F_PNG)

    if cntx.opt_trace:
        log("Saving plot", True)


def save_csv(
    df: pd.DataFrame,
    p: str,
    desc: Optional[str] = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save a dataset or a data array to a CSV file.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    p: str
        Path of file to be created.
    desc: Optional[str]
        Description.
    --------------------------------------------------------------------------------------------------------------------
    """

    if desc == "":
        desc = os.path.basename(p)

    if cntx.opt_trace:
        log("Saving CSV file: " + desc, True)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Save CSV file.
    df.to_csv(p, index=False)

    if cntx.opt_trace:
        log("Saved CSV file", True)


def log(
    msg: str,
    indent=False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Log message to console and into a file.

    Parameters
    ----------
    msg: str
        Message.
    indent: bool
        If True, indent text.
    --------------------------------------------------------------------------------------------------------------------
    """

    ln = ""

    # Start line with a timestamp, unless this is a divide.
    if (msg != "-") and (msg != "="):
        ln = wf_utils.datetime_str()

    if indent:
        ln += " " * c.LOG_N_BLANK
    if (msg == "-") or (msg == "="):
        if indent:
            ln += msg * c.LOG_SEP_LEN
        else:
            ln += msg * (c.LOG_SEP_LEN + c.LOG_N_BLANK)
    else:
        ln += " " + msg

    # Print to console.
    pid_current = os.getpid()
    if pid_current == cntx.pid:
        print(ln)

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(cntx.p_log)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Print to file.
    p_log = cntx.p_log
    if pid_current != cntx.pid:
        p_log = p_log.replace(c.F_EXT_LOG, "_" + str(pid_current) + c.F_EXT_LOG)
    if p_log != "":
        f = open(p_log, "a")
        f.writelines(ln + "\n")
        f.close()


def gen_data(
    p: str
) -> bool:

    """
    --------------------------------------------------------------------------------------------------------------------
    Function that determines whether data should be generated/regenerated.

    Parameters
    ----------
    p: str
        Path

    Returns
    -------
    bool
        True if data (a file or a directory) needs to be generated/regenerated.
    --------------------------------------------------------------------------------------------------------------------
    """

    return (((cntx.f_data_out == c.F_NC) and not os.path.isfile(p)) or
            ((cntx.f_data_out == c.F_ZARR) and not os.path.isdir(p)) or
            cntx.opt_force_overwrite)


def rename(
    p: str,
    text_to_modify: str,
    text_to_replace_with: str,
    rename_files: Optional[bool] = True,
    rename_directories: Optional[bool] = True,
    update_content: Optional[bool] = False,
    recursive: Optional[bool] = False
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Rename files, directories and content under a path.

    This function is not used by the workflow. It is useful to batch rename files from a project if a new version of the
    code has a different structure.

    Parameters
    ----------
    p: str
        Path.
    text_to_modify: str
        Text to modify.
    text_to_replace_with: str
        Text to replace with.
    rename_files: Optional[bool]
        If True, rename files.
    rename_directories: Optional[bool]
        If True, rename directories.
    update_content: Optional[bool]
        If True, update file content.
    recursive: Optional[bool]
        If True, rename recursively (if 'p' is a directory).
    --------------------------------------------------------------------------------------------------------------------
    """

    # List files and directories.
    if os.path.isdir(p):
        p += cntx.sep if p[len(p) - 1] != cntx.sep else ""
        item_l = os.listdir(p)
    else:
        item_l = [p]
    item_l.sort()

    # Loop through items.
    for item_i in item_l:

        # Rename directory or file name.
        if (text_to_modify in item_i) and\
           (((not os.path.isdir(p + item_i)) and rename_files) or ((os.path.isdir(p + item_i)) and rename_directories)):
            shutil.move(p + item_i, p + item_i.replace(text_to_modify, text_to_replace_with))
            item_i = item_i.replace(text_to_modify, text_to_replace_with)

        # Update the content of a CSV file (and remove the index).
        if ((text_to_modify in item_i) or (text_to_replace_with in item_i)) and\
           (c.F_EXT_CSV in item_i) and update_content:

            df = pd.read_csv(p + item_i)
            columns: List[str] = list(df.columns)

            # View 'map': rename the column holding the value to 'val'.
            if cntx.sep + c.VIEW_MAP + cntx.sep in p:
                df.columns = [columns[i].replace(text_to_modify, "val").replace(text_to_replace_with, "val")
                              for i in range(len(columns))]

            # View 'ts': rename column names for lower, middle and upper values.
            elif ((cntx.sep + c.VIEW_TS + cntx.sep in p) or (cntx.sep + c.VIEW_TS_BIAS + cntx.sep in p)) and\
                 ("_rcp" in p):
                df.columns = [columns[i].replace("min", "lower").replace("mean", "middle").replace("moy", "middle")
                                        .replace("max", "upper") for i in range(len(columns))]

            # View 'tbl': rename column 'q' to 'centile', ajust values in columns 'stat' and 'centile'.
            elif cntx.sep + c.VIEW_TBL + cntx.sep in p:
                df.columns = [columns[i].replace("q", c.STAT_CENTILE) for i in range(len(columns))]
                def update_stat(i): return c.STAT_CENTILE if i == c.STAT_QUANTILE else i
                df["stat"] = df["stat"].map(update_stat)
                def update_centile(i): return i * 100 if (i > 0) and (i < 1) else i
                df["centile"] = df["centile"].map(update_centile)

            # Remove the index.
            if ("index" in df.columns[0]) or ("unnamed" in str(df.columns[0]).lower()):
                df = df.iloc[:, 1:]

            save_csv(df, p + item_i)

        # Rename items in a children directory.
        if os.path.isdir(p + item_i) and recursive:
            rename(p + item_i, text_to_modify, text_to_replace_with, rename_files, rename_directories,
                   update_content, recursive)


def migrate(
    platform: str,
    country_region: str,
    version_pre_migration: float,
    version_post_migration: float
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Migrate a project.

    Parameters
    ----------
    platform: str
        Platform = {c.PLATFORM_SCRIPT, c.PLATFORM_STREAMLIT, c.PLATFORM_JUPYTER}
    country_region: str
        Country and region ("<country>-<region>").
    version_pre_migration: float
        Version of workflow, to migrate from.
    version_post_migration: float
        Version of workflow, to migrate to.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Base path of the project.
    if platform == c.PLATFORM_SCRIPT:
        p = cntx.d_res
    else:
        p = os.getcwd() + cntx.sep + "dashboard" + cntx.sep + "data" + cntx.sep + country_region + cntx.sep

    if (version_pre_migration == 1.2) and (version_post_migration == 1.4):

        # Determine the path of the directory holding maps and time series.
        p_map, p_ts = p, p
        if platform == c.PLATFORM_SCRIPT:
            p_map += c.CAT_FIG + cntx.sep
            p_ts += c.CAT_FIG + cntx.sep
        p_map += c.VIEW_MAP + cntx.sep
        p_ts += c.VIEW_TS + cntx.sep

        # Rename cycle plots.
        rename(p, "daily", c.VIEW_CYCLE_D, recursive=True)
        rename(p, "monthly", c.VIEW_CYCLE_MS, recursive=True)

        # Using 'centile' (instead of 'quantile'), and the range of values is from 0 to 100 (instead of 0 to 1).
        rename(p_map, "q10", "c010", update_content=False, recursive=True)
        rename(p_map, "q90", "c090", update_content=False, recursive=True)

        # Indices.
        idx_name_l =\
            [["cdd"] * 2,
             ["cwd"] * 2,
             ["dc", "drought_code"],
             ["drydays", "dry_days"],
             ["drydurtot", "dry_spell_total_length"],
             ["etr"] * 2,
             ["heatwavemaxlen", "heat_wave_max_length"],
             ["heatwavetotlen", "heat_wave_total_length"],
             ["hotspellfreq", "hot_spell_frequency"],
             ["hotspellmaxlen", "hot_spell_max_length"],
             ["prcptot"] * 2,
             ["pr"] * 2,
             ["rainstart", "rain_season_start"],
             ["rainend", "rain_season_end"],
             ["raindur", "rain_season_length"],
             ["rainqty", "rain_season_prcptot"],
             ["r10mm"] * 2,
             ["r20mm"] * 2,
             ["rnnmm", "wet_days"],
             ["rx1day"] * 2,
             ["rx5day"] * 2,
             ["sdii"] * 2,
             ["sfcWindmax"] * 2,
             ["tasmax"] * 2,
             ["tasmin"] * 2,
             ["tas"] * 2,
             ["tgg"] * 2,
             ["tndaysbelow", "tn_days_below"],
             ["tng"] * 2,
             ["tngmonthsbelow", "tng_months_below"],
             ["tnx"] * 2,
             ["tropicalnights", "tropical_nights"],
             ["tx90p"] * 2,
             ["txdaysabove", "tx_days_above"],
             ["txg"] * 2,
             ["txx"] * 2,
             ["uas"] * 2,
             ["vas"] * 2,
             ["wetdays", "wet_days"],
             ["wgdaysabove", "wg_days_above"],
             ["wsdi"] * 2,
             ["wxdaysabove", "wx_days_above"]]

        # Loop through the items to rename.
        for i in range(len(idx_name_l)):
            rename(p, idx_name_l[i][0], idx_name_l[i][1], rename_files=True, rename_directories=True,
                   update_content=True, recursive=True)
            rename(p_ts, "_era5_land", "_rcp", rename_files=True, rename_directories=False,
                   update_content=False, recursive=True)


def deploy():

    """
    --------------------------------------------------------------------------------------------------------------------
    Deploy a project.

    This copies CSV files from the workflow to the dashboard.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Base path of the dashboard project.
    p_dash = os.getcwd() + cntx.sep + "dashboard" + cntx.sep + "data" + cntx.sep + cntx.country + "-" +\
        cntx.region + "-" + (cntx.ens_ref_grid if cntx.ens_ref_grid != "" else cntx.ens_ref_stn) + cntx.sep

    # Copy figures.
    for view_code in [c.VIEW_CYCLE_D, c.VIEW_CYCLE_MS, c.VIEW_MAP, c.VIEW_TAYLOR, c.VIEW_TBL, c.VIEW_TS, c.VIEW_TS_BIAS]:
        for p_work_i in glob.glob(cntx.d_fig(view_code) + "*_" + c.F_CSV):
            p_dash_i = p_dash + view_code + cntx.sep + os.path.basename(p_work_i).replace("_" + c.F_CSV, "")
            dir_util.copy_tree(p_work_i, p_dash_i)
        if view_code == c.VIEW_MAP:
            shutil.copy(cntx.p_bounds, p_dash + view_code + cntx.sep + c.F_BOUNDS)
