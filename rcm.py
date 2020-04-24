# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions linked to Ouranos RCM.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import config as cfg

# Ouranos packages.
from xclim import subset

# Other packages.
import glob
import numpy as np
import os
import xarray as xr


def build_list(variables, rcps, priority_timestep, priority_nam22=False, ouranos_multiple_canesm2=False):

    """
    --------------------------------------------------------------------------------------------------------------------
    Searches CORDEX, NA-CORDEX, and the CRCM5-Ouranos to find the models that contain all the required variables,
    then assembles everything in a dict.
    # TODO: This needs to be completed. Not doing anything.

    Parameters
    ----------
    variables: [str]
        List of required variables.
    rcps : [str]
        List of required rcps, ex: ["rcp45","rcp85"].
        There is no need to list "historical".
    priority_timestep: [int]
        List (length = number of variables)
        "ann","mon","day","6h","3h","1h"
        If a variable exists at multiple timesteps, it will only return the one specified.
        If a variable doesn't exist at the requested time step, it will return the closest one.
    priority_nam22 : boolean, optional
        If true, will remove NAM-44 simulations if a NAM-22 exists.
    ouranos_multiple_canesm2: boolean, optional
        If false, will only select the first CanESM2 member with CRCM5-Ouranos.
    --------------------------------------------------------------------------------------------------------------------
    """

    list_cordex = None

    # TODO: Function 'find_cordex' does not exist.
    # ERROR: list_cordex = find_cordex(variables, rcps, priority_timestep, priority_nam22)

    # TODO: Function 'find_cordexna' does not exist.
    # ERROR: list_cordexna = find_cordexna(variables, rcps, priority_timestep, priority_nam22)

    # TODO: Combine list_cordex and list_cordexna and remove the duplicates that exist in the CORDEX and CORDEX-NA
    #       directories.

    # TODO: Build the list of simulations for CRCM5.

    return list_cordex


def extract_variable(path_ref, path_fut, var, lat_bnds, lon_bnds, priority_timestep=None, tmpdir=None):

    """
    --------------------------------------------------------------------------------------------------------------------
    Uses xarray the extract the data, then uses xclim to subset on the region and the years.
    TODO: Some of the parameters should be optional.

    Parameters
    ----------
    path_ref : str
        For CORDEX, directory where the reference dataset is located.
        For CRCM5, the 3 letters are used internally.
    path_fut : str
        For CORDEX, directory where the future dataset is located.
        For CRCM5, the 3 letters are used internally.
    lat_bnds : float[]
        Latitude boundaries.
    lon_bnds : float[]
        Longitude boundaries
    var : str
        Weather variable.
        All other inputs are lists formatted as [min, max].
    priority_timestep : {"ann","mon","day","6h","3h","1h"}
        Used for CORDEX and NA-CORDEX
        If a variable doesn't exist at the requested time step, it will return the closest one.
    tmpdir : str
        Temporary directory using during processing.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Create an array for all the required years of data.
    all_yr = np.unique(np.concatenate((np.arange(min(cfg.per_ref), max(cfg.per_ref) + 1),
                                       np.arange(min(cfg.per_fut), max(cfg.per_fut) + 1))))

    ds_subset = None

    # CORDEX or CORDEX-NA ----------------------------------------------------------------------------------------------

    if "CORDEX" in path_ref:

        # Find the data at the requested timestep.
        ref_new = path_ref.replace("/*/", "/" + priority_timestep + "/") + var + "/*.nc"
        fut_new = path_fut.replace("/*/", "/" + priority_timestep + "/") + var + "/*.nc"
        files   = sorted(glob.glob(ref_new)) + sorted(glob.glob(fut_new))

        # Since CORDEX-NA (once again) screwed up their directories, we need
        # to check in /raw/ as well.
        if not files:
            ref_new = path_ref.replace("/*/", "/" + priority_timestep + "/") + cfg.cat_raw + "/" + var + "/*.nc"
            fut_new = path_fut.replace("/*/", "/" + priority_timestep + "/") + cfg.cat_raw + "/" + var + "/*.nc"
            files   = sorted(glob.glob(ref_new)) + sorted(glob.glob(fut_new))

        # If no data was found, search other time resolutions.
        if not files:
            timestep_order = ["ann", "mon", "day", "6h", "6hr", "3h", "3hr", "1h", "1hr"]
            index_ts       = timestep_order.index(priority_timestep) + 1

            while not files and index_ts <= len(timestep_order)-1:
                ref_new = path_ref.replace("/*/", "/" + timestep_order[index_ts] + "/") + var + "/*.nc"
                fut_new = path_fut.replace("/*/", "/" + timestep_order[index_ts] + "/") + var + "/*.nc"
                files   = sorted(glob.glob(ref_new)) + sorted(glob.glob(fut_new))
                if not files:
                    ref_new = path_ref.replace("/*/", "/" + timestep_order[index_ts] + "/") + cfg.cat_raw + "/" + var +\
                        "/*.nc"
                    fut_new = path_fut.replace("/*/", "/" + timestep_order[index_ts] + "/") + cfg.cat_raw + "/" + var +\
                        "/*.nc"
                    files   = sorted(glob.glob(ref_new)) + sorted(glob.glob(fut_new))
                index_ts = index_ts+1

        # Extract the files with xarray.
        ds = xr.open_mfdataset(files, chunks={"time": 365}, drop_variables=["time_vectors", "ts", "time_bnds"],
                               combine="by_coords")
        ds_subset = ds.sel(rlat=slice(min(lat_bnds), max(lat_bnds)), rlon=slice(min(lon_bnds), max(lon_bnds))).\
            sel(time=slice(str(min(all_yr)), str(max(all_yr))))
        try:
            grid = ds[var].attrs["grid_mapping"]
            # Rotated_pole gets borked during subset.
            ds_subset[grid] = ds[grid]
        except:
            print("Warning: Could not overwrite grid mapping")

    # CRCM5-Ouranos ----------------------------------------------------------------------------------------------------

    elif len(path_ref) == 3:

        suffix = "/series/" + ("[0-9]"*6) + "/" + var + "_*.nc"

        # Search all 3 drives to find where the data is located.
        files_ref = glob.glob(cfg.path_ds1 + path_ref + suffix)
        if len(files_ref) == 0:
            files_ref.extend(glob.glob(cfg.path_ds2 + path_ref + suffix))
            if len(files_ref) == 0:
                files_ref.extend(glob.glob(cfg.path_ds3 + path_ref + suffix))

        # There are some duplicates, so we need these 'if' statements.
        files_fut = glob.glob(cfg.path_ds1 + path_fut + suffix)
        if len(files_fut) == 0:
            files_fut.extend(glob.glob(cfg.path_ds2 + path_fut + suffix))
            if len(files_fut) == 0:
                files_fut.extend(glob.glob(cfg.path_ds3 + path_fut + suffix))

        files_all = files_ref + files_fut

        # Because there is so much data, we only keep the files for the years we actually want.
        files = []
        for x in range(len(all_yr)):
            files.extend([s for s in files_all if "/" + str(all_yr[x]) in s])
        files = sorted(files)

        # This takes a while, but allows us to use the code without overloading the server. Basically, for each year
        # (12 files), we extract the data and save it to a NetCDF.
        for y in np.arange(all_yr[0], all_yr[len(all_yr)-1], 10):

            sub_files = [f for f in files if "/" + str(y) in f]
            for yy in range(y+1, y+10):
                sub_files.extend([f for f in files if "/" + str(yy) in f])

            ds_tmp = xr.open_mfdataset(sub_files, chunks={"time": 31},
                                       drop_variables=["time_vectors", "ts", "time_bnds"])

            # Spatio-temporal averaging.
            # ds_tmp_dly = ds_tmp.resample(time="1D").mean(dim="time", keep_attrs=True)
            ds_tmp_dly = ds_tmp
            ds_subset_tmp = subset.subset_bbox(ds_tmp_dly, lat_bnds=lat_bnds, lon_bnds=lon_bnds)

            # Rotated_pole gets dropped when we do the resample.
            ds_subset_tmp["rotated_pole"] = ds_tmp.rotated_pole

            # Save each year as a temporary NetCDF.
            ds_subset_tmp.to_netcdf(tmpdir + str(y) + ".nc")

        new_files = sorted(glob.glob(tmpdir + ("[0-9]"*4) + ".nc"))
        ds_subset = xr.open_mfdataset(new_files, chunks={"time": 365})

        # Remove temporary files.
        for f in new_files:
            os.remove(f)

    else:
        print("Folder format is not recognized!")

    # TODO: Extract only REF or only FUT.

    # TODO: Allow for discontinued periods where REF overlaps in FUT (ex: 1981-2010 + 2041-2070).

    # TODO: Check that the boundaries exist, otherwise extract everything.

    return ds_subset
