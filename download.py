# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Download functions.
#
# Create a configuration file, as explained here:
# https://pypi.org/project/cdsapi/
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import cdsapi
import config as cfg
import functools
import glob
import multiprocessing
import os
import utils


def download_from_copernicus(p_base: str, obs_src: str, area: [float], var: str, year: int):

    """
    --------------------------------------------------------------------------------------------------------------------
    Downloads a data set from Copernicus.

    Parameters
    ----------
    p_base : str
        Path of directory where data is saved.
    obs_src : str
        Set code: {cfg.obs_src_era5, cfg.obs_src_era5_land}
    area : [float]
        Bounding box defining the 4 limits of the area of interest (in decimal degrees):
        [North, West, South, East].
    var : str
        Variable code.
        {cfg.var_era5_d2m, cfg.var_era5_e, cfg.var_era5_pev, cfg.var_era5_sp, cfg.var_era5_ssrd, cfg.var_era5_t2m,
        cfg.var_era5_tp, cfg.var_era5_u10, cfg.var_era5_v10}
    year : int
        Year.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Basic configuration.
    set_name = ""
    if obs_src == cfg.obs_src_era5:
        set_name = "era5-single-levels"
    elif obs_src == cfg.obs_src_era5_land:
        set_name = "era5-land"

    # Lists of months, days and times.
    months = [str(d).zfill(2) for d in range(13)]
    days   = [str(d).zfill(2) for d in range(32)]
    times  = ["{}:00".format(str(t).zfill(2)) for t in range(24)]

    # Variable names.
    var_code = ""
    # Can be transformed to become equivalent to cfg.var_co_huss.
    if var == cfg.var_era5_d2m:
        var_code = "2m_dewpoint_temperature"
    # Equivalent to cfg.var_cordex_evspsbl.
    elif var == cfg.var_era5_e:
        var_code = "evaporation"
    # Equivalent to cfg.var_cordex_evspsblpot.
    elif var == cfg.var_era5_pev:
        var_code = "potential_evaporation"
    # Equivalent to cfg.var_cordex_ps.
    elif var == cfg.var_era5_sp:
        var_code = "surface_pressure"
    # Equivalent to cfg.var_cordex_rsds.
    elif var == cfg.var_era5_ssrd:
        var_code = "surface_solar_radiation_downwards"
    # Equivalent to cfg.var_cordex_tas.
    elif var == cfg.var_era5_t2m:
        var_code = "2m_temperature"
    # Equivalent to cfg.var_cordex_pr.
    elif var == cfg.var_era5_tp:
        var_code = "total_precipitation"
    # Equivalent to cfg.var_cordex_uas.
    elif var == cfg.var_era5_u10:
        var_code = "10m_u_component_of_wind"
    # Equivalent to cfg.var_cordex_vas.
    elif var == cfg.var_era5_v10:
        var_code = "10m_v_component_of_wind"

    # Form file name.
    fn = p_base + var + "/" + var + "_" + obs_src + "_hour_" + str(year) + cfg.f_ext_nc
    if os.path.exists(fn):
        return
    p = os.path.dirname(fn)
    if not(os.path.isdir(p)):
        os.makedirs(p)

    c = cdsapi.Client()
    api_request = {
        "product_type": "reanalysis",
        "variable": var_code,
        "year": str(year),
        "month": months,
        "day": days,
        "time": times,
        "area": area,
        "format": "netcdf",
    }
    c.retrieve(
        "reanalysis-" + set_name,
        api_request,
        fn)


def download_merra2(p_base: str, set_version: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Downloads MERRA2.

    This requires an Earthdata account, such as explained here.
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Register+For+an+EarthData+Login+Profile
    and authorize the application "NASA GESDISC DATA ARCHIVE", such as explained here:
    https://disc.gsfc.nasa.gov/earthdata-login

    Parameters
    ----------
    p_base : str
        Path of directory where data is saved.
    set_version : str
        Data set version: {"M2SDNXSLV.5.12.4"}
    --------------------------------------------------------------------------------------------------------------------
    """

    # Basic configuration.
    usr = cfg.obs_src_username
    pwd = cfg.obs_src_password

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
                       "/<year>/<month>/" + set_name + ".<year><month><day>" + cfg.f_ext_nc4

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
                d = p_base + year_str + "/"
                if not (os.path.isdir(d)):
                    os.makedirs(d)
                p = d + "merra2_day_" + year_str + month_str + day_str + cfg.f_ext_nc4

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
    area = [cfg.lat_bnds_download[1], cfg.lon_bnds_download[1], cfg.lat_bnds_download[0], cfg.lon_bnds_download[0], ]

    # Path of input data.
    d_prefix = os.path.dirname(cfg.d_ra_raw) + "/"

    # ERA5 or ERA5_LAND.
    if (cfg.obs_src == cfg.obs_src_era5_land) or (cfg.obs_src == cfg.obs_src_era5):

        # Path, set code and years.
        years = []
        if cfg.obs_src == cfg.obs_src_era5_land:
            years = range(1981, 2019 + 1)
        elif cfg.obs_src == cfg.obs_src_era5:
            years = range(1979, 2019 + 1)

        # Loop through variable codes.
        for var in cfg.variables_download:

            # Need to loop until all files were generated.
            done = False
            while not done:

                # Scalar processing mode.
                if cfg.n_proc == 1:
                    for year in years:
                        download_from_copernicus(d_prefix, cfg.obs_src, area, var, year)

                # Parallel processing mode.
                else:
                    try:
                        utils.log("Processing: '" + var + "'", True)
                        utils.log("Splitting work between " + str(cfg.n_proc) + " threads.", True)
                        pool = multiprocessing.Pool(processes=min(cfg.n_proc, len(years)))
                        func = functools.partial(download_from_copernicus, d_prefix, cfg.obs_src, area, var)
                        pool.map(func, years)
                        pool.close()
                        pool.join()
                        utils.log("Fork ended.", True)
                    except Exception as e:
                        utils.log(str(e))
                        pass

                # Verify if treatment is done. Files are sometimes forgotten.
                years_processed = glob.glob(d_prefix + var + "/*" + cfg.f_ext_nc)
                years_processed.sort()
                for i in range(len(years_processed)):
                    years_processed[i] = int(years_processed[i].replace(cfg.f_ext_nc, "")[-4:])
                years_processed.sort()
                done = (list(years) == years_processed)

    # MERRA2.
    elif cfg.obs_src == cfg.obs_src_merra2:

        # Download.
        download_merra2(d_prefix, "M2SDNXSLV.5.12.4")


if __name__ == "__main__":
    run()
