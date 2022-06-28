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
import wf_file_utils as fu
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard.cl_varidx import VarIdx, VarIdxs


def download_from_ecmwf(
    p_base: str,
    obs_src: str,
    area: [float],
    var: VarIdx,
    year: int
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Downloads a data set from ECMWF.

    Parameters
    ----------
    p_base: str
        Path of directory where data is saved.
    obs_src: str
        Set code: {cl_varidx.ens_era5, cl_varidx.ens_era5_land}
    area: [float]
        Bounding box defining the 4 limits of the area of interest (in decimal degrees):
        [North, West, South, East].
    var: VarIdx
        Variable.
        Supported variables are the following: cntx.v_era5_d2m, cntx.v_era5_e, cntx.v_era5_pev, cntx.v_era5_sp,
        cntx.v_era5_ssrd, cntx.v_era5_t2m, cntx.v_era5_tp, cntx.v_era5_u10, cntx.v_era5_v10}
    year: int
        Year.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Basic configuration.
    set_name = ""
    if obs_src == c.ENS_ERA5:
        set_name = "era5-single-levels"
    elif obs_src == c.ENS_ERA5_LAND:
        set_name = "era5-land"

    # Lists of months, days and times.
    if var.name != c.V_ECMWF_LSM:
        months = [str(d).zfill(2) for d in range(13)]
        days   = [str(d).zfill(2) for d in range(32)]
        times  = ["{}:00".format(str(t).zfill(2)) for t in range(24)]
    else:
        months = ["01"]
        days   = ["01"]
        times  = ["01:00"]

    # Variable names.
    var_ra_name = ""
    # Equivalent to c.V_HUSS.
    if var.name == c.V_ECMWF_D2M:
        var_ra_name = "2m_dewpoint_temperature"
    # Equivalent to c.V_EVSPSBL.
    elif var.name == c.V_ECMWF_E:
        var_ra_name = "evaporation"
    # Equivalent to c.V_EVSPSBLPOT.
    elif var.name == c.V_ECMWF_PEV:
        var_ra_name = "potential_evaporation"
    # Equivalent to c.V_PS.
    elif var.name == c.V_ECMWF_SP:
        var_ra_name = "surface_pressure"
    # Equivalent to c.V_RSDS.
    elif var.name == c.V_ECMWF_SSRD:
        var_ra_name = "surface_solar_radiation_downwards"
    # Equivalent to c.V_TAS.
    elif var.name == c.V_ECMWF_T2M:
        var_ra_name = "2m_temperature"
    # Equivalent to c.V_PR.
    elif var.name == c.V_ECMWF_TP:
        var_ra_name = "total_precipitation"
    # Equivalent to c.V_UAS.
    elif var.name == c.V_ECMWF_U10:
        var_ra_name = "10m_u_component_of_wind"
    # Equivalent to c.V_VAS.
    elif var.name == c.V_ECMWF_V10:
        var_ra_name = "10m_v_component_of_wind"
    # Equivalent to c.V_SFCWINDMAX.
    elif var.name == c.V_ECMWF_UV10:
        var_ra_name = "10m_wind"
    # Equivalent to c.V_SFTLF.
    elif var.name == c.V_ECMWF_LSM:
        var_ra_name = "land_sea_mask"

    # Form file name.
    fn = p_base + var.name + cntx.sep + var.name + "_" + obs_src + "_hour_" + str(year) + c.F_EXT_NC
    if not os.path.exists(fn):

        p = os.path.dirname(fn)
        if not(os.path.isdir(p)):
            os.makedirs(p)

        fu.log("Getting:" + fn, True)

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
    for year in range(cntx.opt_download_per[0], cntx.opt_download_per[1] + 1):
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
                       "/<year>/<month>/" + set_name + ".<year><month><day>" + c.F_EXT_NC4

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
                p = d + "merra2_day_" + year_str + month_str + day_str + c.F_EXT_NC4

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
    # Ex1: Africa      = [47, -29, -50, 65]
    # Ex2: West Africa = [40, -30, -15, 30]
    area = [cntx.opt_download_lat_bnds[1], cntx.opt_download_lon_bnds[0],
            cntx.opt_download_lat_bnds[0], cntx.opt_download_lon_bnds[1]]

    # Path of input data.
    d_prefix = os.path.dirname(cntx.d_ra_raw) + cntx.sep

    # ERA5 or ERA5_LAND.
    if cntx.obs_src in [c.ENS_ERA5_LAND, c.ENS_ERA5]:

        # List variables to download, which involves translating variable names from CORDEX to ECMWF.
        var_ra_name_l = []
        for var in cntx.vars.items:

            # Skip if variable does not exist in the current dataset.
            if (cntx.obs_src == c.ENS_ERA5_LAND) and (var.name == c.V_SFTLF):
                continue

            # Convert name.
            var_ra_name = var.convert_name(cntx.obs_src)
            if var_ra_name in [c.V_ECMWF_T2MMIN, c.V_ECMWF_T2MMAX]:
                var_ra_name_l.append(c.V_ECMWF_T2M)
            elif var_ra_name == c.V_ECMWF_UV10MAX:
                var_ra_name_l.append(c.V_ECMWF_U10)
                var_ra_name_l.append(c.V_ECMWF_V10)
            else:
                var_ra_name_l.append(var_ra_name)

        # Create list of variables into an instance.
        var_ra_name_l = list(set(var_ra_name_l))
        var_ra_name_l.sort()
        vars_ra = VarIdxs(var_ra_name_l)

        # Loop through variable codes.
        for var_ra in vars_ra.items:

            # Determine years.
            years = []
            year_n = cntx.opt_download_per[1]
            if cntx.obs_src == c.ENS_ERA5_LAND:
                year_0 = cntx.opt_download_per[0]
                years = [year_0] if var_ra.code == c.V_ECMWF_LSM else range(year_0, year_n + 1)
            elif cntx.obs_src == c.ENS_ERA5:
                year_0 = cntx.opt_download_per[0] if var_ra.code != c.V_ECMWF_LSM else 1981
                years = [year_0] if var_ra.code == c.V_ECMWF_LSM else range(year_0, year_n + 1)

            # Need to loop until all files were generated.
            done = False
            while not done:

                # Scalar processing mode.
                if cntx.n_proc == 1:
                    for year in years:
                        download_from_ecmwf(d_prefix, cntx.obs_src, area, var_ra, year)

                # Parallel processing mode.
                else:
                    try:
                        fu.log("Processing: " + var_ra.name, True)
                        fu.log("Splitting work between " + str(cntx.n_proc) + " threads.", True)
                        pool = multiprocessing.Pool(processes=min(cntx.n_proc, len(years)))
                        func = functools.partial(download_from_ecmwf, d_prefix, cntx.obs_src, area, var_ra)
                        pool.map(func, list(range(len(years))))
                        pool.close()
                        pool.join()
                        fu.log("Fork ended.", True)
                    except Exception as e:
                        fu.log(str(e))
                        pass

                # Verify if treatment is done. Files are sometimes forgotten.
                years_processed = glob.glob(d_prefix + var_ra.name + cntx.sep + "*" + c.F_EXT_NC)
                years_processed.sort()
                for i in range(len(years_processed)):
                    years_processed[i] = int(years_processed[i].replace(c.F_EXT_NC, "")[-4:])
                done = True
                for year in list(years):
                    if year not in years_processed:
                        done = False
                        break

    # MERRA2.
    elif cntx.obs_src == c.ENS_MERRA2:

        # Download.
        download_merra2(d_prefix, "M2SDNXSLV.5.12.4")


if __name__ == "__main__":
    run()
