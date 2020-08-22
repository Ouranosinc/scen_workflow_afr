# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# TODO.YR: Determine why exception lists are required (sim_excepts, var_sim_excepts). This seems to be due to calendar
#          format.

# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------


# Constants ------------------------------------------------------------------------------------------------------------

# Reanalysis data.
obs_src_era5        = "era5"        # ERA5.
obs_src_era5_land   = "era5_land"   # ERA5-Land.
obs_src_merra2      = "merra2"      # Merra2.

# Emission scenarios.
rcp_ref             = "ref"         # Reference period.
rcp_26              = "rcp26"       # Future period RCP 2.6.
rcp_45              = "rcp45"       # Future period RCP 4.5.
rcp_85              = "rcp85"       # Future period RCP 8.5.

# ==========================================================
# TODO.CUSTOMIZATION.BEGIN
# Add new CORDEX AND ERA5* variables below.
# ==========================================================

# Variables (cordex).
var_cordex_uas      = "uas"         # Wind speed, eastward.
var_cordex_vas      = "vas"         # Wind speed, northward.
var_cordex_ps       = "ps"          # Barometric pressure.
var_cordex_pr       = "pr"          # Precipitation.
var_cordex_rsds     = "rsds"        # Solar radiation.
var_cordex_tas      = "tas"         # Temperature (daily mean).
var_cordex_tasmin   = "tasmin"      # Temperature (daily minimum).
var_cordex_tasmax   = "tasmax"      # Temperature (daily maximum).
var_cordex_clt      = "clt"         # Cloud cover.
var_cordex_huss     = "huss"        # Specific humidity.

# Variables (era5 and era5_land).
var_era5_d2m        = "d2m"         # Dew temperature.
var_era5_e          = "e"           # Evaporation.
var_era5_pev        = "pev"         # Potential evapotranspiration.
var_era5_sp         = "sp"          # Barometric pressure.
var_era5_ssrd       = "ssrd"        # Solar radiation.
var_era5_t2m        = "t2m"         # Temperature.
var_era5_tp         = "tp"          # Precipitation.
var_era5_u10        = "u10"         # Wind speed, eastward.
var_era5_v10        = "v10"         # Wind speed, northward.
var_era5_sh         = "sh"          # Specific humidity.

# ==========================================================
# TODO.CUSTOMIZATION.END
# ==========================================================

# Directory names.
# Observations vs. simulations.
cat_obs             = "obs"         # Observation.
cat_sim             = "sim"         # Simulation.
# Scenario files (in order of generation).
cat_raw             = "raw"         # Raw.
cat_regrid          = "regrid"      # Reggrided.
cat_qqmap           = "qqmap"       # Adjusted simulation.
# Scenarios vs. indices.
cat_scen            = "scen"        # Scenarios.
cat_idx             = "idx"         # Indices.
# Other.
cat_stat            = "stat"        # Statistics.
cat_fig             = "fig"         # Figures.

# Calendar types.
cal_noleap          = "noleap"      # No-leap.
cal_360day          = "360_day"     # 360 days.
cal_365day          = "365_day"     # 365 days.

# Date type.
dtype_obj           = "object"
dtype_64            = "datetime64[ns]"

# Data frequency.
freq_D              = "D"           # Daily.
freq_YS             = "YS"          # Annual.

# Scenarios.
group               = "time.dayofyear"  # Grouping period.

# Calibration.
opt_calib_bias_meth_r2     = "r2"       # Coefficient of determination.
opt_calib_bias_meth_mae    = "mae"      # Mean absolute error.
opt_calib_bias_meth_rmse   = "rmse"     # Root mean square error.
opt_calib_bias_meth_rrmse  = "rrmse"    # Relative root mean square error.

# Indices.
idx_tx_days_above   = "tx_days_above"   # Number of days per year with a maximum temperature above a threshold value.

# Statistics.
stat_mean           = "mean"        # Mean value.
stat_min            = "min"         # Minimum value.
stat_max            = "max"         # Maximum value.
stat_sum            = "sum"         # Sum of values.
stat_quantile       = "quantile"    # Value associated with a given quantile.

# Numerical parameters.
spd                 = 86400         # Number of seconds per day.

# Files.
file_sep            = ","           # File separator (in CSV files).

# Context --------------------------------------------------------------------------------------------------------------

# Country and project.
country             = ""            # Country name.
project             = ""            # Project acronym.

# Emission scenarios, periods and horizons.
rcps                = [rcp_26, rcp_45, rcp_85]                      # All emission scenarios.
per_ref             = [1981, 2010]                                  # Reference period.
per_fut             = [1981, 2100]                                  # Future period.
per_hors            = [[2021, 2040], [2041, 2060], [2061, 2080]]    # Horizons.

# Stations.
# Observations are located in directories /exec/<user_name>/<country>/<project>/obs/<obs_provider>/<var>/*.csv
stns                = []            # Station names.

# Reanalysis data.
obs_src             = ""            # Provider of observations or reanalysis set.
obs_src_username    = ""            # Username of account to download 'obs_src_era5' or 'obs_src_era5_land' data.
obs_src_password    = ""            # Password of account to download 'obs_src_era5' or 'obs_src_era5_land' data.

# Variables.
variables_cordex    = []            # CORDEX data.
variables_ra        = []            # Reanalysis data.
priority_timestep   = ["day"] * len(variables_cordex)

# System ---------------------------------------------------------------------------------------------------------------

# Directories.
# Base directories.
d_base_in1          = ""            # Base directory #1 (input).
d_base_in2          = ""            # Base directory #2 (input).
d_base_exec         = ""            # Base directory #3 (stations and output).
d_username          = ""            # Username on the machine running the script.
# Input-only files and directories.
d_bounds            = ""            # geog.json file comprising political boundaries.
d_ra_raw            = ""            # ERA5 reanalysis set (default frequency).
d_cordex            = ""            # Climate projections (CORDEX).
d_crcm5             = ""            # Climate projections (CRCM5).
# Output-only files and directories.
d_sim               = ""            # Climate projections.
d_stn               = ""            # Grid version of observations.
# Input and output files and directories.
d_ra_day            = ""            # Reanalysis set (day frequency).

# Log file.
p_log               = ""            # Log file (date and time).
log_n_blank         = 10            # Number of blanks at the beginning of a message.
log_sep_len         = 70            # Number of instances of the symbol "-" in a separator line.

# Calibration parameters file.
p_calib             = "calib.csv"   # Calibration file (bias adjustment parameters).

# Performance.
n_proc              = 4             # Number of processes (for multiprocessing).

# Step 2 - Download and aggregation ------------------------------------------------------------------------------------

# Download.
opt_download        = False         # If True, download a dataset.
lon_bnds_download   = [0, 0]        # Longitude boundaries.
lat_bnds_download   = [0, 0]        # Latitude boundaries.

# Aggregation.
opt_aggregate       = False         # If True, aggregate data.

# Steps 3-4 - Data extraction and scenarios ----------------------------------------------------------------------------

# Scenarios.
opt_scen                = True      # If True, produce climate scenarios.
opt_scen_load_obs       = True      # If True, converts observations to NetCDF files.
opt_scen_extract        = True      # If True, forces extraction.
opt_scen_itp_time       = True      # If True, performs temporal interpolation during extraction.
opt_scen_itp_space      = True      # If True, perform spatial interpolation during extraction.
opt_scen_regrid         = False     # If True, relies on the regrid for interpolation. Otherwise, takes nearest point.
opt_scen_preprocess     = True      # If True, forces pre-processing.
opt_scen_postprocess    = True      # If True, forces post-processing.
lon_bnds                = [0, 0]    # Longitude boundaries.
lat_bnds                = [0, 0]    # Latitude boundaries.
radius                  = 0.5       # Search radius (around any given location).
detrend_order           = None      # ???

# Patch.
sim_excepts         = []            # Simulation excluded from the analysis.
                                    # Ex: "RCA4_AFR-44_ICHEC-EC-EARTH_rcp85".
var_sim_excepts     = []            # Simulation-variable combinations excluded from the analysis.
                                    # Ex: var_cordex_pr + "_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85.nc".

# Step 5 - Bias adjustment and statistical downscaling -----------------------------------------------------------------

# Calibration options.
opt_calib           = True          # If True, explores the sensitivity to nq, up_qmf and time_win parameters.
opt_calib_auto      = True          # If True, calibrates for nq, up_qmf and time_win parameters.
opt_calib_bias      = True          # If True, examines bias correction.
opt_calib_bias_meth = "rrmse"       # Error quantification method (select one of the following methods).
opt_calib_coherence = False         # If True, examines physical coherence.
opt_calib_qqmap     = True          # If true, calculate qqmap.

# Bias parameters.
# The parameter 'time_win' is the number of days before and after any given day (15 days before and after = 30 days).
# This needs to be adjusted as there is period of adjustment between cold period and monsoon). It's possible that a
# very small precipitation amount be considered extreme. We need to limit correction factors.

# Default values.
nq_default          = 50            # Default 'nq' value.
up_qmf_default      = 3.0           # Default 'up_qmf' value.
time_win_default    = 30            # Default 'time_win' value.
bias_err_default    = -1            # Default 'bias_err' value.

# Array of values to test for each calibration parameter.
nq_calib            = [nq_default]          # List of 'nq' values to test during calibration.
up_qmf_calib        = [up_qmf_default]      # List of 'up_wmf' values to test during calibration.
time_win_calib      = [time_win_default]    # List of 'time_win' values to test during calibration.
bias_err_calib      = [bias_err_default]    # List of 'bias_err' values to test during calibration.

# Container for calibration parameters ().
df_calib            = None          # Pandas dataframe.

# Step 6 - Indices -----------------------------------------------------------------------------------------------------

# Indices.
opt_idx             = True          # If True, calculate indices.
idx_resol           = 0.05          # Spatial resolution for mapping.
idx_names           = []            # Index names.
idx_threshs         = []            # Index thresholds.

# Step 7 - Statistics --------------------------------------------------------------------------------------------------

opt_stat            = True          # If True, calculate statistics.
stat_quantiles      = [1.00, 0.99, 0.75, 0.50, 0.25, 0.01, 0.00]  # Quantiles.

# Step 8 - Visualization -----------------------------------------------------------------------------------------------

# Plots.
opt_plot            = True          # If True, actives plot generation.

# Color associated with specific datasets (for consistency).
col_sim_adj_ref     = "blue"        # Simulation (bias-adjusted) for the reference period.
col_sim_ref         = "orange"      # Simulation (non-adjusted) for the reference period
col_obs             = "green"       # Observations.
col_sim_adj         = "red"         # Simulation (bias-adjusted).
col_sim_fut         = "purple"      # Simulation (non-adjusted) for the future period.
col_ref             = "black"       # Reference period.
col_rcp26           = "blue"        # RCP 2.6.
col_rcp45           = "green"       # RCP 4.5.
col_rcp85           = "red"         # RCP 8.5.


def get_rank_inst():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the token corresponding to the institute.
    --------------------------------------------------------------------------------------------------------------------
    """

    return len(d_cordex.split("/"))


def get_rank_gcm():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the rank of token corresponding to the GCM.
    --------------------------------------------------------------------------------------------------------------------
    """

    return get_rank_inst() + 1


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


def get_idx_desc(idx_name, idx_threshs):

        """
        ----------------------------------------------------------------------------------------------------------------
        Gets the description of an index.

        Parameters
        ----------
        idx_name : str
            Climate index.
        idx_threshs : [[float]]]
            Thresholds
        ----------------------------------------------------------------------------------------------------------------
        """

        idx_desc = ""
        if idx_name == idx_tx_days_above:
            idx_desc = "Nombre de jours avec " + get_var_desc(var_cordex_tasmax).lower() + " > " + str(idx_threshs[0])

        return idx_desc


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
        Emission scenario, e.g., {cfg.rcp_ref, cfg.rcp_26, cfg.rcp_45, cfg.rcp_85}
    --------------------------------------------------------------------------------------------------------------------
    """

    rcp_desc = ""
    if rcp == rcp_ref:
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
        d = d + "stn/" + stn + "/"
    if category != "":
        d = d
        if (category == cat_raw) or (category == cat_regrid) or (category == cat_qqmap):
            d = d + "scen/"
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
