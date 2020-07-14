# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
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
d_base_in1        = ""  # Base directory #1 (input).
d_base_in2        = ""  # Base directory #2 (input).
d_base_exec       = ""  # Base directory #3 (stations and output).
d_username        = ""  # Username on the machine running the script.

# Input-only files and directories.
d_bounds          = ""  # geog.json file comprising political boundaries.
d_era5_hour       = ""  # ERA5 reanalysis set (hourly frequency).
d_era5_land_hour  = ""  # ERA5-Land reanalysis set (hourly frequency).
d_cordex          = ""  # Climate projections (CORDEX).
d_crcm5           = ""  # Climate projections (CRCM5).

# Output-only files and directories.
d_sim             = ""  # Climate projections.
d_stn             = ""  # Grid version of observations.

# Input and output files and directories.
d_era5_day        = ""  # ERA5 reanalysis set (daily frequency).
d_era5_land_day   = ""  # ERA5-Land reanalysis set (daily frequency).

# Files.
p_log             = ""           # Log file (date and time).
p_calib           = "calib.csv"  # Calibration file (bias adjustment parameters).

# Emission scenarios, periods and horizons.
rcp_26  = "rcp26"                   # Emission scenario RCP 2.6.
rcp_45  = "rcp45"                   # Emission scenario RCP 4.5.
rcp_85  = "rcp85"                   # Emission scenario RCP 8.5.
rcps    = [rcp_26, rcp_45, rcp_85]  # All emission scenarios.
per_ref = [1981, 2010]                                 # Reference period.
per_fut = [1981, 2100]                                 # Future period.
per_hors = [[2021, 2040], [2041, 2060], [2061, 2080]]  # Horizons.

# Indices.
idx_names         = []  # Indices.
idx_threshs       = []  # Thresholds.
idx_tx_days_above = "tx_days_above"

# Geographic boundaries and search radius.
# Longitude and latitude boundaries.
lon_bnds = [0, 0]
lat_bnds = [0, 0]
# Spatial resolution for climate indices mapping.
resol_idx = 0.05
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
# TODO.YR: Determine why exception lists are required.
#          This seems to be due to calendar format. This should be fixed.
sim_excepts     = []
var_sim_excepts = []

# Index of simulation set (if the analysis is related to a single set; there are 3 files per simulation).
idx_sim = []

# Bias correction.
# The parameter 'time_int' is the number of days before and after any given day (15 days before and after = 30 days).
# This needs to be adjusted as there is period of adjustment between cold period and monsoon). It's possible that a
# very small precipitation amount be considered extreme. We need to limit correction factors.
# Default values.
nq_default       = 50   # Default 'nq' value.
up_qmf_default   = 3.0  # Default 'up_qmf' value.
time_int_default = 30   # Default 'time_int' value.
bias_err_default = -1   # Default 'bias_err' value.
# For calibration.
# Array of values to test for each calibration parameter.
nq_calib         = None   # List of 'nq' values to test during calibration.
up_qmf_calib     = None   # List of 'up_wmf' values to test during calibration.
time_int_calib   = None   # List of 'time_int' values to test during calibration.
bias_err_calib   = None   # List of 'bias_err' values to test during calibration.
# For workflow.
# Dictionaries with 3 dimensions [sim][stn][var] where 'sim' is simulation name, 'stn' is station name,
# and 'var' is the variable.
# nq               = None   # Number of quantiles (calibrated value).
# up_qmf           = None   # Upper limit for quantile mapping function.
# time_int         = None   # Windows size (i.e. number of days before + number of days after).
# bias_err         = None   # Bias adjustment error.
df_calib = None  # Calibration parameters (pandas dataframe).

# Step 2 - Download options.
opt_download = False

# Step 3 - Aggregation options.
opt_aggregate = False

# Step 4 - Scenario options.
opt_scen                  = True     # If True, produce climate scenarios.
opt_scen_read_obs_netcdf  = True     # If True, converts observations to NetCDF files.
opt_scen_extract          = True     # If True, forces extraction.
opt_scen_itp_time         = True     # If True, performs temporal interpolation during extraction.
opt_scen_itp_space        = True     # If True, perform spatial interpolation during extraction.
opt_scen_regrid           = False    # If True, relies on the regrid for interpolation. Otherwise, takes nearest point.
opt_scen_preprocess       = True     # If True, forces pre-processing.
opt_scen_postprocess      = True     # If True, forces post-processing.
opt_calib                 = True     # If True, explores the sensitivity to nq, up_qmf and time_int parameters.
opt_calib_auto            = True     # If True, calibrates for nq, up_qmf and time_int parameters.
opt_calib_bias            = True     # If True, examines bias correction.
opt_calib_bias_meth       = "rrmse"  # Error quantification method (select one of the following methods).
opt_calib_bias_meth_r2    = "r2"     # Coefficient of determination.
opt_calib_bias_meth_mae   = "mae"    # Mean absolute error.
opt_calib_bias_meth_rmse  = "rmse"   # Root mean square error.
opt_calib_bias_meth_rrmse = "rrmse"  # Relative root mean square error.
opt_calib_coherence       = False    # If True, examines physical coherence.
opt_calib_qqmap           = True     # If true, calculate qqmap.

# Indices options.
opt_idx = True  # If True, calculate indices.

# Plot options.
opt_plt_pp_fut_obs = True  # If True, generates plots of future and observation (in workflow).
opt_plt_ref_fut    = True  # If True, generates plots of reference vs future (in workflow).
opt_plt_365vs360   = True  # If True, generates plots of temporal interpolation (in worflow; for debug purpose only).
opt_plt_save       = True  # If True, save plots.
opt_plt_close      = True  # If True, close plots.

# Log options.
log_n_blank = 10  # Number of blanks at the beginning of a message.
log_sep_len = 70  # Number of instances of the symbol "-" in a separator line.


def get_idx_inst():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the token corresponding to the institute.
    --------------------------------------------------------------------------------------------------------------------
    """

    return len(d_cordex.split("/"))


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


def get_rcp_desc(rcp):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the description of an emission scenario.

    Parameters
    ----------
    rcp : str
        Emission scenario, e.g., {"ref", "rcp26", "rcp45", "rcp85"}
    --------------------------------------------------------------------------------------------------------------------
    """

    if rcp == "ref":
        rcp_desc = "reference"
    elif ("rcp" in rcp) and (len(rcp) == 5):
        rcp_desc = rcp[0:3].upper() + " " + rcp[3] + "." + rcp[4]

    return rcp_desc


def get_d_sim(stn, category, var=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get directory of simulations.

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

    d = d_sim
    if stn != "":
        d = d + stn + "/"
    if category != "":
        d = d + category + "/"
    if var != "":
        d = d + var + "/"

    return d


def get_d_stn(var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get directory of stations.

    Parameters
    ----------
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    d = ""
    if var != "":
        d = d_stn + var + "/"

    return d


def get_p_stn(var, stn):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get path of stations.

    Parameters
    ----------
    var : str
        Variable.
    stn : str
        Station.
    --------------------------------------------------------------------------------------------------------------------
    """

    p = d_stn + var + "/" + var + "_" + stn + ".nc"

    return p


def get_p_obs(stn_name, var, category=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get direction of observations.

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

    p = get_d_sim(stn_name, "obs") + var + "/" + var + "_" + stn_name
    if category != "":
        p = p + "_4qqmap"
    p = p + ".nc"

    return p
