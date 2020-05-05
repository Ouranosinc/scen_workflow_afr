# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import utils

# Project information.
# Country name.
country = ""
# Project.
project = ""
# User name.
user_name = ""
# Provider of observations.
obs_provider = ""

# File system.
# Simulation data.
path_src = "/dmf2/scenario/external_data/CORDEX-AFR/"
path_ds1 = "/expl6/climato/arch/"
path_ds2 = "/dmf2/climato/arch/"
path_ds3 = "/expl7/climato/arch/"
# Rank of token corresponding to the institute.
idx_institute = len(path_src.split("/"))
# Rank of token corresponding to the GCM.
idx_gcm = idx_institute + 1

# Emission scenarios and periods.
rcps    = ["rcp26", "rcp45", "rcp85"]  # Emission scenarios.
per_ref = [1981, 2010]                 # Reference period.
per_fut = [1981, 2100]                 # Future period.

# Geographic boundaries and search radius.
# Latitude (southern and northern boundaries).
lat_bnds = [8, 16]
# Longitude (western dn eastern boundaries).
lon_bnds = [-6, 3]
# Search radius (around any given location).
radius   = 0.5

# Numerical parameters.
group         = "time.dayofyear"
detrend_order = None

# Variables.
var_uas    = "uas"     # Wind in the eastward direction.
var_vas    = "vas"     # Wind in the northward direction.
var_pr     = "pr"      # Precipitation.
var_tas    = "tas"     # Temperature (daily mean).
var_tasmin = "tasmin"  # Temperature (daily minimum).
var_tasmax = "tasmax"  # Temperature (daily maximum).
var_clt    = "clt"     # Cloud cover.

# Data categories: observation, raw, regrid, qqmap or figure.
cat_obs    = "obs"
cat_raw    = "raw"
cat_regrid = "regrid"
cat_qqmap  = "qqmap"
cat_fig    = "fig"

# Calendar types: no-leap, 360 days or 365 days.
cal_noleap = "noleap"
cal_360day = "360_day"
cal_365day = "365_day"

# Date types.
dtype_obj = "object"
dtype_64 = "datetime64[ns]"

# Number of seconds per day.
spd = 86400

# Stations.
# Observations are located in directories /exec/<user_name>/<country>/<project>/obs/<obs_provider>/<var>/*.csv
stn_names = [""]

# Variables.
variables = []
priority_timestep = ["day"] * len(variables)

# List of simulation and var-simulation combinations that must be avoided to avoid a crash.
# Example: "RCA4_AFR-44_ICHEC-EC-EARTH_rcp85" (for sim_excepts).
#          var_pr + "_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85.nc" (for var_sim_excepts).
# TODO.YR: Determine why exception lists are required.
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
# Dictionary with 3 dimensi_nameons [sim][stn_name][var] where 'sim' is simulation name, 'stn_name' is station name,
# and 'var' is the variable.
nq               = None   # Number of quantiles (calibrated value).
up_qmf           = None   # Upper limit for quantile mapping function.
time_int         = None   # Windows size (i.e. number of days before + number of days after).

# Calibration options.
opt_calib           = True  # If True, explores the sensitivity to nq, up_qmf and time_int parameters.
opt_calib_auto      = True  # If True, calibrates for nq, up_qmf and time_int parameters.
opt_calib_bias      = True  # If True, examines bias correction.
opt_calib_coherence = True  # If True, examines physical coherence.
opt_calib_qqmap     = True  # If true, calculate qqmap.
opt_calib_extra     = True  # If True, overlaps additional curves on time-series.

# Workflow options.
opt_wflow_read_obs_netcdf = True   # If True, converts observations to NetCDF files.
opt_wflow_extract         = True   # If True, forces extraction.
opt_wflow_itp_time        = True   # If True, performs temporal interpolation during extraction.
opt_wflow_itp_space       = True   # If True, perform spatial interpolation during extraction.
opt_wflow_regrid          = False  # If True, relies on the regrid for interpolation. Otherwise, takes nearest point.
opt_wflow_preprocess      = True   # If True, forces pre-processing.
opt_wflow_postprocess     = True   # If True, forces post-processing.

# Plot options.
opt_plt_pp_fut_obs = True  # If True, generates plots of future and observation (in workflow).
opt_plt_ref_fut    = True  # If True, generates plots of reference vs future (in workflow).
opt_plt_365vs360   = True  # If True, generates plots of temporal interpolation (in worflow; for debug purpose only).
opt_plt_save       = True  # If True, save plots.
opt_plt_close      = True  # If True, close plots.


def get_var_desc(var):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets the description of a variable.

    Parameters
    ----------
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine description.
    var_desc = ""
    if var == var_tas:
        var_desc = "Temp. moyenne"
    elif var == var_tasmin:
        var_desc = "Temp. minimale"
    elif var == var_tasmax:
        var_desc = "Temp. maximale"
    elif var == var_pr:
        var_desc = "Précipitations"
    elif var == var_uas:
        var_desc = "Vent (dir. est)"
    elif var == var_vas:
        var_desc = "Vent (dir. nord)"
    elif var == var_clt:
        var_desc = "Couvert nuageux"

    return var_desc


def get_var_unit(var):
    """
    --------------------------------------------------------------------------------------------------------------------
    Gets the unit of a variable.

    Parameters
    ----------
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine unit.
    var_unit = ""
    if (var == var_tas) or (var == var_tasmin) or (var == var_tasmax):
        var_unit = "°C"
    elif var == var_pr:
        var_unit = "mm"
    elif (var == var_uas) or (var == var_vas):
        var_unit = "m s-1"
    elif var == var_clt:
        var_unit = "%"

    return var_unit


def get_path_out(stn_name, category, var=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets a path.

    Parameters
    ----------
    stn_name : str
        Station name.
    category : str
        Category.
    var : str, optional
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = "/exec/" + user_name + "/sim_climat/" + country + "/" + project + "/"
    if stn_name != "":
        path = path + stn_name + "/"
    if category != "":
        path = path + category + "/"
    if var != "":
        path = path + var + "/"

    return path


def get_path_stn(var, stn_name):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets path of directory containing stations.

    Parameters
    ----------
    var : str
        Variable.
    stn_name : str
        Station name.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = "/exec/" + user_name + "/" + country + "/" + project + "/" + cat_obs + "/" + obs_provider + "/"
    if var != "":
        path = path + var + "/"
        if stn_name != "":
            path = path + stn_name + ".nc"

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

    path = get_path_out(stn_name, "obs") + var + "/" + var + "_" + stn_name
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
    nq       = utils.create_multi_dict(3, float)
    up_qmf   = utils.create_multi_dict(3, float)
    time_int = utils.create_multi_dict(3, float)
    list_cordex = utils.list_cordex(path_src, rcps)
    for idx_rcp in range(len(rcps)):
        rcp = rcps[idx_rcp]
        for idx_sim in range(0, len(list_cordex[rcp])):
            list     = list_cordex[rcp][idx_sim].split("/")
            sim_name = list[idx_institute] + "_" + list[idx_institute + 1]
            for stn_name in stn_names:
                for var in variables:
                    nq[sim_name][stn_name][var]       = nq_default
                    up_qmf[sim_name][stn_name][var]   = up_qmf_default
                    time_int[sim_name][stn_name][var] = time_int_default
