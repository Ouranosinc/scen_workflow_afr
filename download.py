# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Download functions.
#
# Create a configuration file, as explained here:
# https://pypi.org/project/cdsapi/
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import cdsapi
import functools
import glob
import multiprocessing
import os
import sys

# Workflow libraries.
import file_utils as fu
from def_constant import const as c
from def_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import def_varidx as vi


def download_from_copernicus(
    p_base: str,
    obs_src: str,
    area: [float],
    var: vi.VarIdx,
    year: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Downloads a data set from Copernicus.

    Parameters
    ----------
    p_base: str
        Path of directory where data is saved.
    obs_src: str
        Set code: {def_varidx.ens_era5, def_varidx.ens_era5_land}
    area: [float]
        Bounding box defining the 4 limits of the area of interest (in decimal degrees):
        [North, West, South, East].
    var: vi.VarIdx
        Variable.
        Supported variables are the following: cntx.v_era5_d2m, cntx.v_era5_e, cntx.v_era5_pev, cntx.v_era5_sp,
        cntx.v_era5_ssrd, cntx.v_era5_t2m, cntx.v_era5_tp, cntx.v_era5_u10, cntx.v_era5_v10}
    year: int
        Year.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Basic configuration.
    set_name = ""
    if obs_src == c.ens_era5:
        set_name = "era5-single-levels"
    elif obs_src == c.ens_era5_land:
        set_name = "era5-land"

    # Lists of months, days and times.
    months = [str(d).zfill(2) for d in range(13)]
    days   = [str(d).zfill(2) for d in range(32)]
    times  = ["{}:00".format(str(t).zfill(2)) for t in range(24)]

    # Variable names.
    var_ra_name = ""
    # Equivalent to c.v_huss.
    if var.name == c.v_era5_d2m:
        var_ra_name = "2m_dewpoint_temperature"
    # Equivalent to c.v_evspsbl.
    elif var.name == c.v_era5_e:
        var_ra_name = "evaporation"
    # Equivalent to c.v_evspsblpot.
    elif var.name == c.v_era5_pev:
        var_ra_name = "potential_evaporation"
    # Equivalent to c.v_ps.
    elif var.name == c.v_era5_sp:
        var_ra_name = "surface_pressure"
    # Equivalent to c.v_rsds.
    elif var.name == c.v_era5_ssrd:
        var_ra_name = "surface_solar_radiation_downwards"
    # Equivalent to c.v_tas.
    elif var.name == c.v_era5_t2m:
        var_ra_name = "2m_temperature"
    # Equivalent to c.v_pr.
    elif var.name == c.v_era5_tp:
        var_ra_name = "total_precipitation"
    # Equivalent to c.v_uas.
    elif var.name == c.v_era5_u10:
        var_ra_name = "10m_u_component_of_wind"
    # Equivalent to c.v_vas.
    elif var.name == c.v_era5_v10:
        var_ra_name = "10m_v_component_of_wind"
    # Equivalent to vi.v_sfcwindmax.
    elif var.name == c.v_era5_uv10:
        var_ra_name = "10m_wind"

    # Form file name.
    fn = p_base + var.name + cntx.sep + var.name + "_" + obs_src + "_hour_" + str(year) + c.f_ext_nc
    if not os.path.exists(fn):

        p = os.path.dirname(fn)
        if not(os.path.isdir(p)):
            os.makedirs(p)

        client = cdsapi.Client()
        api_request = {
            "product_type": "reanalysis",
            "variable": var_ra_name,
            "year": str(year),
            "month": months,
            "day": days,
            "time": times,
            "area": area,
            "format": "netcdf",
        }
        client.retrieve(
            "reanalysis-" + set_name,
            api_request,
            fn)

    if cntx.n_proc > 1:
        fu.log("Work done!", True)


def download_merra2(
    p_base: str,
    set_version: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Downloads MERRA2.

    This requires an Earthdata account, such as explained here.
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Register+For+an+EarthData+Login+Profile
    and authorize the application "NASA GESDISC DATA ARCHIVE", such as explained here:
    https://disc.gsfc.nasa.gov/earthdata-login

    Parameters
    ----------
    p_base: str
        Path of directory where data is saved.
    set_version: str
        Data set version: {"M2SDNXSLV.5.12.4"}
    --------------------------------------------------------------------------------------------------------------------
    """

    # Basic configuration.
    usr = cntx.obs_src_username
    pwd = cntx.obs_src_password

    # Loop through years.
    for year in range(1980, 2020):
        year_str = str(year)

        # URL template.
        if year <= 1991:
            set_name = "MERRA2_100.statD_2d_slv_Nx"
        elif year <= 2000:
            set_name = "MERRA2_200.statD_2d_slv_Nx"
        elif year <= 2010:
            set_name = "MERRA2_300.statD_2d_slv_Nx"
        else:
            set_name = "MERRA2_400.statD_2d_slv_Nx"
        url_template = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/" + set_version + \
                       "/<year>/<month>/" + set_name + ".<year><month><day>" + c.f_ext_nc4

        # Loop through months.
        for month in range(1, 13):
            month_str = str(month).rjust(2, "0")

            # Loop through days.
            for day in range(1, 32):
                day_str   = str(day).rjust(2, "0")

                # Skip if necessary.
                if (((month == 4) or (month == 6) or (month == 9) or (month == 11)) and (day > 30)) or \
                   ((month == 2) and (day > 29) and (year % 4 == 0)) or \
                   ((month == 2) and (day > 28) and (year % 4 > 0)):
                    continue

                # Form url.
                url  = url_template.replace("<year>", year_str).replace("<month>", month_str).replace("<day>", day_str)

                # Form local path and file name.
                d = p_base + year_str + cntx.sep
                if not (os.path.isdir(d)):
                    os.makedirs(d)
                p = d + "merra2_day_" + year_str + month_str + day_str + c.f_ext_nc4

                # Download.
                cmd = "wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies " +\
                      "--content-disposition --user " + usr + " --password " + pwd + " " + url + " -O " + p
                os.system(cmd)


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Area: [North, West, South, East, ]
    # Ex1: Africa      = [47, -29, -50, 65, ]
    # Ex2: West Africa = [40, -30, -15, 30, ]
    area = [cntx.lat_bnds_download[1], cntx.lon_bnds_download[1], cntx.lat_bnds_download[0], cntx.lon_bnds_download[0]]

    # Path of input data.
    d_prefix = os.path.dirname(cntx.d_ra_raw) + cntx.sep

    # ERA5 or ERA5_LAND.
    if (cntx.obs_src == c.ens_era5_land) or (cntx.obs_src == c.ens_era5):

        # Path, set code and years.
        years = []
        if cntx.obs_src == c.ens_era5_land:
            years = range(1981, 2019 + 1)
        elif cntx.obs_src == c.ens_era5:
            years = range(1979, 2019 + 1)

        # Loop through variable codes.
        for var in cntx.vars_download.code_l:

            # Need to loop until all files were generated.
            done = False
            while not done:

                # Scalar processing mode.
                if cntx.n_proc == 1:
                    for year in years:
                        download_from_copernicus(d_prefix, cntx.obs_src, area, var, year)

                # Parallel processing mode.
                else:
                    try:
                        fu.log("Processing: " + var.name, True)
                        fu.log("Splitting work between " + str(cntx.n_proc) + " threads.", True)
                        pool = multiprocessing.Pool(processes=min(cntx.n_proc, len(years)))
                        func = functools.partial(download_from_copernicus, d_prefix, cntx.obs_src, area, var)
                        pool.map(func, years)
                        pool.close()
                        pool.join()
                        fu.log("Fork ended.", True)
                    except Exception as e:
                        fu.log(str(e))
                        pass

                # Verify if treatment is done. Files are sometimes forgotten.
                years_processed = glob.glob(d_prefix + var.name + cntx.sep + "*" + c.f_ext_nc)
                years_processed.sort()
                for i in range(len(years_processed)):
                    years_processed[i] = int(years_processed[i].replace(c.f_ext_nc, "")[-4:])
                years_processed.sort()
                done = (list(years) == years_processed)

    # MERRA2.
    elif cntx.obs_src == c.ens_merra2:

        # Download.
        download_merra2(d_prefix, "M2SDNXSLV.5.12.4")


if __name__ == "__main__":
    run()
