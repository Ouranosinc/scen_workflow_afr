# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import utils

# Project information.
country = ""  # Country name.
project = ""  # Project name.

# Reanalysis data.
obs_src           = ""  # Provider of observations or reanalysis set.
obs_src_username  = ""  # Username of account to download 'obs_src_era5' or 'obs_src_era5_land' data.
obs_src_password  = ""  # Password of account to download 'obs_src_era5' or 'obs_src_era5_land' data.
obs_src_era5      = "era5"
obs_src_era5_land = "era5_land"

# Base directories.
path_base_in1       = ""  # Base directory #1 (input).
path_base_in2       = ""  # Base directory #2 (input).
path_base_out       = ""  # Base directory #3 (output).
path_username       = ""  # Username on the machine running the script.

# Input-only files and directories.
path_bounds         = ""  # geog.json file comprising political boundaries.
path_era5_hour      = ""  # ERA5 reanalysis set (hourly frequency).
path_era5_land_hour = ""  # ERA5-Land reanalysis set (hourly frequency).
path_cordex         = ""  # Climate projections (CORDEX).
path_ds1            = ""  # TODO: Determine what this variable does.
path_ds2            = ""  # TODO: Determine what this variable does.
path_ds3            = ""  # TODO: Determine what this variable does.

# Output-only files and directories.
path_sim            = ""  # Climate projections.
path_stn            = ""  # Grid version of observations.

# Input and output files and directories.
path_era5_day       = ""  # ERA5 reanalysis set (daily frequency).
path_era5_land_day  = ""  # ERA5-Land reanalysis set (daily frequency).

# Emission scenarios, periods and horizons.
rcp_26  = "rcp26"                   # Emission scenario RCP 2.6.
rcp_45  = "rcp45"                   # Emission scenario RCP 4.5.
rcp_85  = "rcp85"                   # Emission scenario RCP 8.5.
rcps    = [rcp_26, rcp_45, rcp_85]  # All emission scenarios.
per_ref = [1981, 2010]                                # Reference period.
per_fut = [1981, 2100]                                # Future period.
per_hor = [[2021, 2040], [2041, 2060], [2061, 2080]]  # Horizons.

# Indices.
idx_names         = []  # Indices.
idx_threshs       = []  # Thresholds.
idx_tx_days_above = "tx_days_above"

# Geographic boundaries and search radius.
# Latitude (southern and northern boundaries).
lat_bnds = [0, 0]
# Longitude (western dn eastern boundaries).
lon_bnds = [0, 0]
# Search radius (around any given location).
radius = 0.5

# Numerical parameters.
group         = "time.dayofyear"
detrend_order = None
n_proc        = 4      # Number of processes (for multiprocessing).
spd           = 86400  # Number of seconds per day.

# Variables (cordex).
var_cordex_uas    = "uas"     # Wind speed, eastward.
var_cordex_vas    = "vas"     # Wind speed, northward.
var_cordex_ps     = "ps"      # Barometric pressure.
var_cordex_pr     = "pr"      # Precipitation.
var_cordex_rsds   = "rsds"    # Solar radiation.
var_cordex_tas    = "tas"     # Temperature (daily mean).
var_cordex_tasmin = "tasmin"  # Temperature (daily minimum).
var_cordex_tasmax = "tasmax"  # Temperature (daily maximum).
var_cordex_clt    = "clt"     # Cloud cover.
var_cordex_huss   = "huss"    # Specific humidity.

# Variables (era5 and era5_land).
var_era5_d2m      = "d2m"     # Dew temperature.
var_era5_e        = "e"       # Evaporation.
var_era5_pev      = "pev"     # Potential evapotranspiration.
var_era5_sp       = "sp"      # Barometric pressure.
var_era5_ssrd     = "ssrd"    # Solar radiation.
var_era5_t2m      = "t2m"     # Temperature.
var_era5_tp       = "tp"      # Precipitation.
var_era5_u10      = "u10"     # Wind speed, eastward.
var_era5_v10      = "v10"     # Wind speed, northward.
var_era5_sh       = "sh"      # Specific humidity.

# Data categories: observation, raw, regrid, qqmap or figure.
cat_obs           = "obs"
cat_raw           = "raw"
cat_regrid        = "regrid"
cat_qqmap         = "qqmap"
cat_fig           = "fig"

# Calendar types: no-leap, 360 days or 365 days.
cal_noleap = "noleap"
cal_360day = "360_day"
cal_365day = "365_day"

# Date types.
dtype_obj = "object"
dtype_64  = "datetime64[ns]"

# Stations.
# Observations are located in directories /exec/<user_name>/<country>/<project>/obs/<obs_provider>/<var>/*.csv
stns = []

# Variables.
variables_cordex = []  # CORDEX data.
variables_ra     = []  # Reanalysis data.
priority_timestep = ["day"] * len(variables_cordex)

# List of simulation and var-simulation combinations that must be avoided to avoid a crash.
# Example: "RCA4_AFR-44_ICHEC-EC-EARTH_rcp85" (for sim_excepts).
#          var_cordex_pr + "_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85.nc" (for var_sim_excepts).
# TODO.YR: Determine why exception lists are required (probably due to calendar format).
sim_excepts     = []
var_sim_excepts = []

# Index of simulation set (if the analysis is related to a single set; there are 3 files per simulation).
idx_sim = []

# Bias correction.
# The parameter 'time_int' is the number of days before and after any given days (15 days before and after = 30 days).
# This needs to be adjusted as there is period of adjustment between cold period and monsoon). It's possible that a
# very small precipitation amount be considered extreme. We need to limit correction factors.
# Default values.
nq_default       = 50   # Default 'nq' value.
up_qmf_default   = 3.0  # Default 'up_qmf' value.
time_int_default = 30   # Default 'time_int' value.
# For calibration.
# Array of values to test for each calibration parameter.
nq_calib         = None   # List of 'nq' values to test during calibration.
up_qmf_calib     = None   # List of 'up_wmf' values to test during calibration.
time_int_calib   = None   # List of 'time_int' values to test during calibration.
# For workflow.
# Dictionary with 3 dimensi_nameons [sim][stn][var] where 'sim' is simulation name, 'stn' is station name,
# and 'var' is the variable.
nq               = None   # Number of quantiles (calibrated value).
up_qmf           = None   # Upper limit for quantile mapping function.
time_int         = None   # Windows size (i.e. number of days before + number of days after).

# Step 2 - Download options.
opt_download = False

# Step 3 - Aggregation options.
opt_aggregate = False

# Step 4 - Scenario options.
opt_scen                 = True
opt_scen_read_obs_netcdf = True   # If True, converts observations to NetCDF files.
opt_scen_extract         = True   # If True, forces extraction.
opt_scen_itp_time        = True   # If True, performs temporal interpolation during extraction.
opt_scen_itp_space       = True   # If True, perform spatial interpolation during extraction.
opt_scen_regrid          = False  # If True, relies on the regrid for interpolation. Otherwise, takes nearest point.
opt_scen_preprocess      = True   # If True, forces pre-processing.
opt_scen_postprocess     = True   # If True, forces post-processing.
opt_calib           = True  # If True, explores the sensitivity to nq, up_qmf and time_int parameters.
opt_calib_auto      = True  # If True, calibrates for nq, up_qmf and time_int parameters.
opt_calib_bias      = True  # If True, examines bias correction.
opt_calib_coherence = True  # If True, examines physical coherence.
opt_calib_qqmap     = True  # If true, calculate qqmap.
opt_calib_extra     = True  # If True, overlaps additional curves on time-series.

# Indices options.
opt_idx = True  # If True, calculate indices.

# Plot options.
opt_plt_pp_fut_obs = True  # If True, generates plots of future and observation (in workflow).
opt_plt_ref_fut    = True  # If True, generates plots of reference vs future (in workflow).
opt_plt_365vs360   = True  # If True, generates plots of temporal interpolation (in worflow; for debug purpose only).
opt_plt_save       = True  # If True, save plots.
opt_plt_close      = True  # If True, close plots.


def get_idx_inst():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the token corresponding to the institute.
    --------------------------------------------------------------------------------------------------------------------
    """

    return len(path_cordex.split("/"))


def get_idx_gcm():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the rank of token corresponding to the GCM.
    --------------------------------------------------------------------------------------------------------------------
    """

    return get_idx_inst() + 1


def get_var_desc(var, set_name="cordex"):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets the description of a variable.

    Parameters
    ----------
    var : str
        Variable.
    set_name : str
        Station name.
    --------------------------------------------------------------------------------------------------------------------
    """

    var_desc = ""
    if set_name == "cordex":
        if var == var_cordex_tas:
            var_desc = "Temp. moyenne"
        elif var == var_cordex_tasmin:
            var_desc = "Temp. minimale"
        elif var == var_cordex_tasmax:
            var_desc = "Temp. maximale"
        elif var == var_cordex_ps:
            var_desc = "Pression barométrique"
        elif var == var_cordex_pr:
            var_desc = "Précipitations"
        elif var == var_cordex_rsds:
            var_desc = "Radiation solaire"
        elif var == var_cordex_uas:
            var_desc = "Vent (dir. est)"
        elif var == var_cordex_vas:
            var_desc = "Vent (dir. nord)"
        elif var == var_cordex_clt:
            var_desc = "Couvert nuageux"
        elif var == var_cordex_huss:
            var_desc = "Humidité spécifique"
    elif (set_name == obs_src_era5) or (set_name == obs_src_era5_land):
        if var == var_era5_d2m:
            var_desc = "Point de rosée"
        elif var == var_era5_t2m:
            var_desc = "Température"
        elif var == var_era5_sp:
            var_desc = "Pression barométrique"
        elif var == var_era5_tp:
            var_desc = "Précipitations"
        elif var == var_era5_u10:
            var_desc = "Vent (dir. est)"
        elif var == var_era5_v10:
            var_desc = "Vent (dir. nord)"
        elif var == var_era5_ssrd:
            var_desc = "Radiation solaire"
        elif var == var_era5_e:
            var_desc = "Évaporation"
        elif var == var_era5_pev:
            var_desc = "Évapotranspiration potentielle"
        elif var == var_era5_sh:
            var_desc = "Humidité spécifique"

    return var_desc


def get_var_unit(var, set_name="cordex"):
    """
    --------------------------------------------------------------------------------------------------------------------
    Gets the unit of a variable.

    Parameters
    ----------
    var : str
        Variable.
    set_name : str
        Station name.
    --------------------------------------------------------------------------------------------------------------------
    """

    var_unit = ""
    if set_name == "cordex":
        if (var == var_cordex_tas) or (var == var_cordex_tasmin) or (var == var_cordex_tasmax):
            var_unit = "°C"
        elif var == var_cordex_rsds:
            var_unit = "Pa"
        elif var == var_cordex_pr:
            var_unit = "mm"
        elif (var == var_cordex_uas) or (var == var_cordex_vas):
            var_unit = "m s-1"
        elif var == var_cordex_clt:
            var_unit = "%"
        elif var == var_cordex_huss:
            var_unit = "1"
    elif (set_name == obs_src_era5) or (set_name == obs_src_era5_land):
        if (var == var_era5_d2m) or (var == var_era5_t2m):
            var_unit = "°"
        elif var == var_era5_sp:
            var_unit = "Pa"
        elif (var == var_era5_u10) or (var == var_era5_v10):
            var_unit = "m s-1"
        elif var == var_era5_ssrd:
            var_unit = "J m-2"
        elif (var == var_era5_tp) or (var == var_era5_e) or (var == var_era5_pev):
            var_unit = "m"
        elif var == var_era5_sh:
            var_unit = "1"

    return var_unit


def get_path_sim(stn, category, var=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets a path.

    Parameters
    ----------
    stn : str
        Station.
    category : str
        Category.
    var : str, optional
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = path_sim
    if stn != "":
        path = path + stn + "/"
    if category != "":
        path = path + category + "/"
    if var != "":
        path = path + var + "/"

    return path


def get_path_stn(var, stn):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets path of directory containing stations.

    Parameters
    ----------
    var : str
        Variable.
    stn : str
        Station.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = ""
    if var != "":
        path = path_stn + var + "/"
        if stn != "":
            path = path + stn + ".nc"

    return path


def get_path_obs(stn_name, var, category=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets the path of observations.

    Parameters
    ----------
    stn_name : str
        Localisation.
    var : str
        Variable.
    category : str, optional
        Category.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = get_path_sim(stn_name, "obs") + var + "/" + var + "_" + stn_name
    if category != "":
        path = path + "_4qqmap"
    path = path + ".nc"

    return path


def init_calib_params():

    """
    -----------------------------------------------------------------------------------------------------------------
    Initialize calibration parameters.
    --------------------------------------------------------------------------------------------------------------------
    """

    global nq, up_qmf, time_int
    nq = utils.create_multi_dict(3, float)
    up_qmf = utils.create_multi_dict(3, float)
    time_int = utils.create_multi_dict(3, float)
    list_cordex = utils.list_cordex(path_cordex, rcps)
    for idx_rcp in range(len(rcps)):
        rcp = rcps[idx_rcp]
        for idx_sim_i in range(0, len(list_cordex[rcp])):
            list_i = list_cordex[rcp][idx_sim_i].split("/")
            sim_name = list_i[get_idx_inst()] + "_" + list_i[get_idx_inst() + 1]
            for stn in stns:
                for var in variables_cordex:
                    nq[sim_name][stn][var] = nq_default
                    up_qmf[sim_name][stn][var] = up_qmf_default
                    time_int[sim_name][stn][var] = time_int_default
