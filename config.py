# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Project information.
# Country name.
country = "burkina"
# Project.
project = "pcci"
# User name.
user_name = "yrousseau"
# File prefix (Python files).
f_prefix = country + "_" + project
# Provider of observations.
obs_provider = "anam"

# Path of directory containing simulation data.
path_src = "/dmf2/scenario/external_data/CORDEX-AFR/"
path_ds1 = "/expl6/climato/arch/"
path_ds2 = "/dmf2/climato/arch/"
path_ds3 = "/expl7/climato/arch/"

# Emission scenarios and horizons.
# Emission scenarios.
rcps = ["rcp26", "rcp45", "rcp85"]
# Reference period.
per_ref = [1988, 2017]
# Future period
per_fut = [1988, 2095]

# Geographic boundaries and search radius.
# Latitude (southern and northern boundaries).
lat_bnds = [8, 16]
# Longitude (western dn eastern boundaries).
lon_bnds = [-6, 3]
# Search radius (around any given location).
radius = 0.5

# Numerical parameters.
# Number of days before and after any given days (15 days before and after = 30 days).
# This needs to be adjusted as there is period of adjustment between cold period and monsoon). It's possible that a very
# small precipitation amount be considered extreme. We need to limit correction factors.
time_int = 30
group = "time.dayofyear"
detrend_order = None

# Weather variables.
# Wind in the eastward direction.
var_uas = "uas"
# Wind in the northward direction.
var_vas = "vas"
# Precipitation.
var_pr = "pr"
# Temperature (daily mean).
var_tas = "tas"
# Temperature (daily minimum).
var_tasmin = "tasmin"
# Temperature (daily maximum).
var_tasmax = "tasmax"
# Cloud cover.
var_clt = "clt"

# Data categories.
# Observation data.
cat_obs = "obs"
# Raw data.
cat_raw = "raw"
# Regrid data.
cat_regrid = "regrid"
# QQMap data.
cat_qqmap = "qqmap"

# Calendar types.
# No-leap year.
cal_noleap = "noleap"
# Year described with 360 days.
cal_360day = "360_day"
# Year described with 365 days.
cal_365day = "365_day"

# Date types.
dtype_obj = "object"
dtype_64 = "datetime64[ns]"

# Number of seconds per day.
spd = 86400


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
    if category != "":
        path = path + category + "/"
    if stn_name != "":
        path = path + stn_name + "/"
    if var != "":
        path = path + var + "/"

    return path


def get_path_stn(var=""):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets path of directory containing stations.

    Parameters
    ----------
    var : str, optional
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    path = "/exec/" + user_name + "/" + country + "/" + project + "/" + cat_obs + "/" + obs_provider + "/"
    if var != "":
        path = path + var + "/"

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
