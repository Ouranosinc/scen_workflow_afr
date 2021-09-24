# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions linked to Ouranos RCM.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca (current)
# 2. bourgault.marcandre@ouranos.ca (second)
# 3. rondeau-genesse.gabriel@ouranos.ca (original)
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import glob
import numpy as np
import os
import utils
import clisops.core.subset as subset


def extract_variable(
    d_ref: str,
    d_fut: str,
    var: str,
    lat_bnds: [float],
    lon_bnds: [float],
    priority_timestep: str = None,
    tmpdir: str = None
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Uses xarray the extract the data, then uses xclim to subset on the region and the years.
    Caution: This function is only compatible with scalar processing (n_proc=1) owing to the call to the function
    utils.open_netcdf with a list of NetCDF files.
    TODO.MAB: Some of the parameters should be optional.
    TODO.MAB: Extract only REF or only FUT.
    TODO.MAB: Allow for discontinued periods where REF overlaps in FUT (ex: 1981-2010 + 2041-2070).
    TODO.MAB: Check that the boundaries exist, otherwise extract everything.

    Parameters
    ----------
    d_ref : str
        For CORDEX, directory where the reference dataset is located.
        For CRCM5, the 3 letters are used internally.
    d_fut : str
        For CORDEX, directory where the future dataset is located.
        For CRCM5, the 3 letters are used internally.
    lat_bnds : [float]
        Latitude boundaries.
    lon_bnds : [float]
        Longitude boundaries
    var : str
        Weather variable.
        All other inputs are lists formatted as [min, max].
    priority_timestep : str
        Used for CORDEX and NA-CORDEX
        If a variable doesn't exist at the requested time step, it will return the closest one.
        {"ann", "mon", "day", "6h", "3h", "1h"}
    tmpdir : str
        Temporary directory using during processing.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Create an array for all the required years of data.
    all_yr = np.unique(np.concatenate((np.arange(min(cfg.per_ref), max(cfg.per_ref) + 1),
                                       np.arange(min(cfg.per_fut), max(cfg.per_fut) + 1))))

    ds_subset = None

    # CORDEX or CORDEX-NA ----------------------------------------------------------------------------------------------

    if "cordex" in d_ref.lower():

        # Find the data at the requested timestep.
        p_ref = d_ref.replace(cfg.sep + "*" + cfg.sep, cfg.sep + priority_timestep + cfg.sep) +\
            var + cfg.sep + "*" + cfg.f_ext_nc
        p_fut = d_fut.replace(cfg.sep + "*" + cfg.sep, cfg.sep + priority_timestep + cfg.sep) +\
            var + cfg.sep + "*" + cfg.f_ext_nc
        p_l   = sorted(glob.glob(p_ref)) + sorted(glob.glob(p_fut))

        # Since CORDEX-NA (once again) screwed up their directories, we need
        # to check in /raw/ as well.
        if not p_l:
            p_ref = d_ref.replace(cfg.sep + "*" + cfg.sep, cfg.sep + priority_timestep + cfg.sep) +\
                cfg.cat_raw + cfg.sep + var + cfg.sep + "*" + cfg.f_ext_nc
            p_fut = d_fut.replace(cfg.sep + "*" + cfg.sep, cfg.sep + priority_timestep + cfg.sep) +\
                cfg.cat_raw + cfg.sep + var + cfg.sep + "*" + cfg.f_ext_nc
            p_l   = sorted(glob.glob(p_ref)) + sorted(glob.glob(p_fut))

        # If no data was found, search other time resolutions.
        if not p_l:
            timestep_order = ["ann", "mon", "day", "6h", "6hr", "3h", "3hr", "1h", "1hr"]
            index_ts       = timestep_order.index(priority_timestep) + 1

            while not p_l and index_ts <= len(timestep_order)-1:
                p_ref = d_ref.replace(cfg.sep + "*" + cfg.sep, cfg.sep + timestep_order[index_ts] + cfg.sep) +\
                    var + cfg.sep + "*" + cfg.f_ext_nc
                p_fut = d_fut.replace(cfg.sep + "*" + cfg.sep, cfg.sep + timestep_order[index_ts] + cfg.sep) +\
                    var + cfg.sep + "*" + cfg.f_ext_nc
                p_l   = sorted(glob.glob(p_ref)) + sorted(glob.glob(p_fut))
                if not p_l:
                    p_ref  = d_ref.replace(cfg.sep + "*" + cfg.sep, cfg.sep + timestep_order[index_ts] + cfg.sep) +\
                        cfg.cat_raw + cfg.sep + var + cfg.sep + "*" + cfg.f_ext_nc
                    p_fut  = d_fut.replace(cfg.sep + "*" + cfg.sep, cfg.sep + timestep_order[index_ts] + cfg.sep) +\
                        cfg.cat_raw + cfg.sep + var + cfg.sep + "*" + cfg.f_ext_nc
                    p_l = sorted(glob.glob(p_ref)) + sorted(glob.glob(p_fut))
                index_ts = index_ts + 1

        # Extract the files with xarray.
        ds = utils.open_netcdf(p_l, chunks={cfg.dim_time: 365}, drop_variables=["time_vectors", "ts", "time_bnds"],
                               combine="by_coords")
        ds_subset = ds.sel(rlat=slice(min(lat_bnds), max(lat_bnds)), rlon=slice(min(lon_bnds), max(lon_bnds))).\
            sel(time=slice(str(min(all_yr)), str(max(all_yr))))
        # Rotated_pole gets borked during subset.
        try:
            grid = ds[var].attrs[cfg.attrs_gmap]
            ds_subset[grid] = ds[grid]
        except:
            utils.log("Warning: Could not overwrite grid mapping:", True)

    # CRCM5-Ouranos ----------------------------------------------------------------------------------------------------

    elif len(d_ref) == 3:

        suffix = cfg.sep + "series" + cfg.sep + ("[0-9]"*6) + cfg.sep + var + "_*" + cfg.f_ext_nc

        # List files for the reference and future periods.
        p_ref_l = glob.glob(cfg.d_proj + d_ref + suffix)
        p_fut_l = glob.glob(cfg.d_proj + d_fut + suffix)
        p_all_l = p_ref_l + p_fut_l

        # Because there is so much data, we only keep the files for the years we actually want.
        p_l = []
        for x in range(len(all_yr)):
            p_l.extend([s for s in p_all_l if cfg.sep + str(all_yr[x]) in s])
        p_l = sorted(p_l)

        # This takes a while, but allows us to use the code without overloading the server. Basically, for each year
        # (12 files), we extract the data and save it to a NetCDF.
        for y in np.arange(all_yr[0], all_yr[len(all_yr)-1], 10):

            p_l_sub = [f for f in p_l if cfg.sep + str(y) in f]
            for yy in range(y + 1, y + 10):
                p_l_sub.extend([f for f in p_l if cfg.sep + str(yy) in f])

            ds_tmp = utils.open_netcdf(p_l_sub, chunks={cfg.dim_time: 31},
                                       drop_variables=["time_vectors", "ts", "time_bnds"])

            # Spatio-temporal averaging.
            ds_tmp_dly = ds_tmp
            ds_subset_tmp = subset.subset_bbox(ds_tmp_dly, lat_bnds=lat_bnds, lon_bnds=lon_bnds)

            # Rotated_pole gets dropped when we do the resample.
            ds_subset_tmp["rotated_pole"] = ds_tmp.rotated_pole

            # Save each year as a temporary NetCDF.
            utils.save_netcdf(ds_subset_tmp, tmpdir + str(y) + cfg.f_ext_nc)

        p_l_new = sorted(glob.glob(tmpdir + ("[0-9]"*4) + cfg.f_ext_nc))
        ds_subset = utils.open_netcdf(p_l_new, chunks={cfg.dim_time: 365})

        # Remove temporary files.
        for f in p_l_new:
            os.remove(f)

    else:
        utils.log("Folder format is not recognized!", True)

    return ds_subset
