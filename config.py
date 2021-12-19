# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import os.path

import sys
sys.path.append("dashboard")
from dashboard import varidx_def as vi, rcp_def


# Constants ------------------------------------------------------------------------------------------------------------

# Projection data.
prj_src_cordex      = "cordex"      # CORDEX.

# Data array attributes.
attrs_units         = "units"
attrs_sname         = "standard_name"
attrs_lname         = "long_name"
attrs_axis          = "axis"
attrs_gmap          = "grid_mapping"
attrs_gmapname      = "grid_mapping_name"
attrs_bias          = "bias_corrected"
attrs_comments      = "comments"
attrs_stn           = "Station Name"
attrs_group         = "group"
attrs_kind          = "kind"

# Units (general).
unit_deg            = "°"
unit_pct            = "%"

# Units (NetCDF).
unit_C              = "C"
unit_K              = "K"
unit_kg_m2s1        = "kg m-2 s-1"
unit_m              = "m"
unit_mm             = "mm"
unit_mm_d           = "mm d-1"
unit_m_s            = "m s-1"
unit_J_m2           = "J m-2"
unit_Pa             = "Pa"
unit_d              = "d"
unit_km_h           = "km h-1"
unit_1              = "1"

# Units (description; for plots).
unit_C_desc         = unit_deg + unit_C
unit_K_desc         = unit_K
unit_kg_m2s1_desc   = "kg/m²s)"
unit_m_desc         = unit_m
unit_mm_desc        = unit_mm
unit_mm_d_desc      = "mm/jour"
unit_m_s_desc       = "m/s"
unit_J_m2_desc      = "J/m²"
unit_Pa_desc        = unit_Pa
unit_d_desc         = "jours"
unit_km_h_desc      = "km/h"

# Dataset dimensions.
dim_lon             = "lon"
dim_lat             = "lat"
dim_rlon            = "rlon"
dim_rlat            = "rlat"
dim_longitude       = "longitude"
dim_latitude        = "latitude"
dim_time            = "time"
dim_location        = "location"

# Directory names.
sep                 = "/"            # Separator.
# Reference data.
cat_stn             = "stn"          # At-a-station or reanalysis data.
# Scenario files (in order of generation).
cat_obs             = "obs"          # Observation.
cat_raw             = "raw"          # Raw.
cat_regrid          = "regrid"       # Reggrided.
cat_qqmap           = "qqmap"        # Adjusted simulation.
cat_qmf             = "qmf"          # Quantile mapping function.
# Scenarios vs. indices.
cat_scen            = "scen"         # Scenarios.
cat_idx             = "idx"          # Indices.
# Other.
cat_stat            = "stat"         # Statistics.
cat_fig             = "fig"          # Figures.
cat_fig_calibration = "calibration"  # Figures (calibration).
cat_fig_postprocess = "postprocess"  # Figures (postprocess).
cat_fig_workflow    = "workflow"     # Figures (workflow).
cat_fig_cycle_ms    = "cycle_ms"     # Figures (annual cycle, monthly).
cat_fig_cycle_d     = "cycle_d"      # Figures (annual cycle, daily).

# Calendar types.
cal_noleap          = "noleap"      # No-leap.
cal_360day          = "360_day"     # 360 days.
cal_365day          = "365_day"     # 365 days.

# Date type.
dtype_obj           = "object"
dtype_64            = "datetime64[ns]"

# Data frequency.
freq_D              = "D"           # Daily.
freq_MS             = "MS"          # Monthly.
freq_YS             = "YS"          # Annual.

# Scenarios.
group               = "time.dayofyear"  # Grouping period.

# Kind.
kind_add            = "+"               # Additive.
kind_mult           = "*"               # Multiplicative.

# Calibration.
opt_calib_bias_meth_r2     = "r2"       # Coefficient of determination.
opt_calib_bias_meth_mae    = "mae"      # Mean absolute error.
opt_calib_bias_meth_rmse   = "rmse"     # Root mean square error.
opt_calib_bias_meth_rrmse  = "rrmse"    # Relative root mean square error.

# Climate index parameters ---------------------------------------------------------------------------------------------

"""
Temperature ------------------------------------------------

etr: Extreme temperature range.
Requirements: tasmin, tasmax
Parameters:   [nan]

tx90p: Number of days with extreme maximum temperature (> 90th percentile).
Requirements: tasmax
Parameters:   [nan]

heat_wave_max_length: Maximum heat wave length.
Requirements: tasmin, tasmax
Parameters:   [tasmin_thresh: float, tasmax_thresh: float, n_days: int]
              tasmin_thresh: daily minimum temperature must be greater than a threshold value.
              tasmax_thresh: daily maximum temperature must be greater than a threshold value.
              n_days: minimum number of consecutive days. 

heat_wave_total_len: Total heat wave length.
Requirements: tasmin, tasmax
Parameters:   [tasmin_thresh: float, tasmax_thresh: float, n_days: int]
              tasmin_thresh: daily minimum temperature must be greater than a threshold value.
              tasmax_thresh: daily maximum temperature must be greater than a threshold value.
              n_days: Minimum number of consecutive days.

hot_spell_frequency: Number of hot spells.
Requirements: tasmax
Parameters:   [tasmax_thresh: float, n_days: int]
              tasmax_thresh: daily maximum temperature must be greater than a threshold value.
              n_days: minimum number of consecutive days.

hot_spell_max_length: Maximum hot spell length.
Requirements: tasmax
Parameters:   [tasmax_threshold: float, n_days: int]
              tasmax_threshold: daily maximum temperature must be greater than this threshold.
              n_days: minimum number of consecutive days.

tgg: Mean of mean temperature.
Requirements: tasmin, tasmax
Parameters:   [nan]

tng: Mean of minimum temperature.
Requirements: tasmin
Parameters:   [nan]

tnx: Maximum of minimum temperature.
Requirements: tasmin
Parameters:   [nan]

txg: Mean of maximum temperature.
Requirements: tasmax
Parameters:   [nan]

txx: Maximum of maximum temperature.
Parameters: [nan]

tng_months_below: Number of months per year with a mean minimum temperature below a threshold.
Requirements: tasmin
Parameters:   [tasmin_thresh: float]
              tasmin_thresh: daily minimum temperature must be greater than a threshold value.

tx_days_above: Number of days per year with maximum temperature above a threshold.
Requirements: tasmax
Parameters:   [tasmax_thresh: float]
              tasmax_thresh: daily maximum temperature must be greater than a threshold value.

tn_days_below: Number of days per year with a minimum temperature below a threshold.
Requirements: tasmin
Parameters:   [tasmin_thresh: float, doy_min: int, doy_max: int]
              tasmin_thresh: daily minimum temperature must be greater than a threshold value.
              doy_min: minimum day of year to consider.
              doy_max: maximum day of year to consider.

tropical_nights: Number of tropical nights, i.e. with minimum temperature above a threshold.
Requirements: tasmin
Parameters:   [tasmin_thresh: float]
              tasmin_thresh: daily minimum temperature must be greater than a threshold value.

wsdi: Warm spell duration index.
Requirements: tasmax
Parameters:   [tasmax_thresh=nan, n_days: int]
              tasmax_thresh: daily maximum temperature must be greater than a threshold value; this value is
              calculated automatically and corresponds to the 90th percentile of tasmax values.
              n_days: minimum number of consecutive days.

# Precipitation --------------------------------------------

rx1day: Largest 1-day precipitation amount.
Requirements: pr
Parameters:   [nan]

rx5day: Largest 5-day precipitation amount.
Requirements: pr
Parameters:   [nan]

cdd: Maximum number of consecutive dry days.
Requirements: pr
Parameters:   [pr_thresh: float]
              pr_thresh: a daily precipitation amount lower than a threshold value is considered dry.

cwd: Maximum number of consecutive wet days.
Requirements: pr
Parameters:   [pr_thresh: float]
              pr_thresh: a daily precipitation amount greater than or equal to a threshold value is considered wet.

dry_days: Number of dry days.
Requirements: pr
Parameters:   [pr_thresh: float]
              pr_thresh: a daily precipitation amount lower than a threshold value is considered dry.

wet_days: Number of wet days.
Requirements: pr
Parameters:   [pr_thresh: float]
              pr_thresh: a daily precipitation amount greater than or equal to a threshold value is considered wet.

prcptot: Accumulated total precipitation.
Requirements: pr
Parameters:   [pct=nan, doy_min: int, doy_max: int]
              pct: the default value is nan;
                   if a value is provided, an horizontal line is drawn on time series;
                   if a percentile is provided (ex: 90p), the equivalent value is calculated and an horizontal line is
                   drawn on time series.
              doy_min: minimum day of year to consider.
              doy_max: maximum day of year to consider.

r10mm: Number of days with precipitation greater than or equal to 10 mm.
Requirements: pr
Parameters:   [nan]

r20mm: Number of days with precipitation greater than or equal to 20 mm.
Requirements: pr
Parameters:   [nan]


rnnmm: Number of days with precipitation greater than or equal to a user-provided value.
Requirements: pr
Parameters:   [pr_thresh: float]
              pr_thresh: daily precipitation amount must be greater than or equal to a threshold value.

sdii: Mean daily precipitation intensity.
Requirements: pr
Parameters:   [pr_thresh: float]
              pr_thresh: daily precipitation amount must be greater than or equal to a threshold value.

rain_season: Rain season.
Requirements: pr (mandatory), evspsbl* (optional)
Parameters:   combination of the parameters of indices idx_rain_season_start and idx_rain_season_end.

rain_season_start: Day of year where rain season starts.
Requirements: pr
Parameters: [thresh_wet: str, window_wet: int, thresh_dry: str, dry_days_max: int, window_dry: int,
    start_date: str, end_date: str, freq: str]
    thresh_wet: Accumulated precipitation threshold associated with {window_wet}.
    window_wet: Number of days where accumulated precipitation is above {thresh_wet}.
    thresh_dry: Daily precipitation threshold associated with {window_dry}.
    dry_days:   Maximum number of dry days in {window_dry}.
    window_dry: Number of days, after {window_wet}, during which daily precipitation is not greater than or equal to
                {thresh_dry} for {dry_days} consecutive days.
    start_date: First day of year where season can start ("mm-dd").
    end_date:   Last day of year where season can start ("mm-dd").
    freq:       Resampling frequency.

rain_season_end: Day of year where rain season ends.
Requirements: pr (mandatory), rain_season_start_next (optional), evspsbl* (optional)
              will search for evspsblpot, then for evspsbl
Parameters: [method: str, thresh: str, window: int, etp_rate: str, start_date: str, end_date: str, freq: str]
    op: Resampling operator = {"max", "sum", "etp}
        If "max": based on the occurrence (or not) of an event during the last days of a rain season.
            The rain season ends when no daily precipitation greater than {thresh} have occurred over a period of
            {window} days.
        If "sum": based on a total amount of precipitation received during the last days of the rain season.
            The rain season ends when the total amount of precipitation is less than {thresh} over a period of
            {window} days.
        If "etp": calculation is based on the period required for a water column of height {thresh] to
            evaporate, considering that any amount of precipitation received during that period must evaporate as
            well. If {etp} is not available, the evapotranspiration rate is assumed to be {etp_rate}.
    thresh: maximum or accumulated precipitation threshold associated with {window}.
        If {op} == "max": maximum daily precipitation  during a period of {window} days.
        If {op} == "sum": accumulated precipitation over {window} days.
        If {op} == "etp": height of water column that must evaporate
    window: int
        If {op} in ["max", "sum"]: number of days used to verify if the rain season is ending.
    etp_rate:
        If {op} == "etp": evapotranspiration rate.
        Otherwise: not used.
    start_date: First day of year where season can end ("mm-dd").
    end_date: Last day of year where season can end ("mm-dd").
    freq: Resampling frequency.

rain_season_length: Duration of the rain season.
Requirements: rain_season_start, rain_season_end
Parameters:   [nan]

rain_season_prcptot: Quantity received during rain season.
Requirements: pr, rain_season_start, rain_season_end
Parameters:   [nan]

dry_spell_total_length: Total length of dry period.
Requirements: pr
Parameters:   [thresh: str, window: int, op: str, start_date: str, end_date: str]
    thresh: precipitation threshold
    op: period over which to combine data: "max" = one day, "sum" = cumulative over {window} days.
        If {op} == "max": daily precipitation amount under which precipitation is considered negligible.
        If {op} == "sum": sum of daily precipitation amounts under which the period is considered dry.
    window: minimum number of days required in a dry period.
    start_date: first day of year to consider ("mm-dd").
    end_date: last day of year to consider ("mm-dd").

# Wind -----------------------------------------------------

wg_days_above: Number of days per year with mean wind speed above a threshold value coming from a given direction.
Requirements: uas, vas
Parameters:   [speed_tresh: float, velocity_thresh_neg: float, dir_thresh: float, dir_thresh_tol: float,
               doy_min: int, doy_max: int]
              speed_tresh: wind speed must be greater than or equal to a threshold value.
                   if a percentile is provided (ex: 90p), the equivalent value is calculated.
              speed_tresh_neg: wind speed is considered negligible if smaller than or equal to a threshold value.
              dir_thresh: wind direction (angle, in degrees) must be close to a direction given by a threshold value.
              dir_thresh_tol: wind direction tolerance (angle, in degrees).
              doy_min: minimum day of year to consider (nan can be provided).
              doy_max: maximum day of year to consider (nan can be provided).

wx_days_above: Number of days per year with maximum wind speed above a threshold value.
Requirements: sfcWindmax
Parameters:   [speed_tresh: float, nan, nan, nan, doy_min: int, doy_max: int]
              speed_tresh: wind speed must be greater than or equal to a threshold value.
                   if a percentile is provided (ex: 90p), the equivalent value is calculated.
              doy_min: minimum day of year to consider (nan can be provided).
              doy_max: maximum day of year to consider (nan can be provided).

# Temperature-precipitation --------------------------------

drought_code: Drought code.
Requirements: tas, pr
Parameters:   [nan]
"""

# Statistics.
stat_mean           = "mean"        # Mean value.
stat_min            = "min"         # Minimum value.
stat_max            = "max"         # Maximum value.
stat_sum            = "sum"         # Sum of values.
stat_quantile       = "quantile"    # Value associated with a given quantile.

# Conversion coefficients.
spd                 = 86400         # Number of seconds per day.
d_KC                = 273.15        # Temperature difference between Kelvin and Celcius.
km_h_per_m_s        = 3.6           # Number of km/h per m/s.

# Files.
f_sep               = ","           # File separator (only in CSV files containing observations).
f_csv               = "csv"         # CSV file type (comma-separated values).
f_png               = "png"         # PNG file type (image).
f_tif               = "tif"         # TIF file type (image, potentially georeferenced).
f_nc                = "nc"          # NetCDF file type.
f_nc4               = "nc4"         # NetCDF v4 file type.
f_ext_csv           = "." + f_csv   # CSV file extension.
f_ext_png           = "." + f_png   # PNG file extension.
f_ext_tif           = "." + f_tif   # TIF file extension.
f_ext_nc            = "." + f_nc    # NetCDF file extension.
f_ext_nc4           = "." + f_nc4   # NetCDF v4 file extension.
f_ext_log           = ".log"        # LOG file extension.

# Context --------------------------------------------------------------------------------------------------------------

# Country and project.
country             = ""            # Country name.
project             = ""            # Project acronym.

# Emission scenarios, periods and horizons.
rcps                = [rcp_def.rcp_26, rcp_def.rcp_45, rcp_def.rcp_85]          # All emission scenarios.
per_ref             = [1981, 2010]                                              # Reference period.
per_fut             = [1981, 2100]                                              # Future period.
per_hors            = [[1981, 2010], [2021, 2040], [2041, 2060], [2061, 2080]]  # Horizons.

# Boundary box.
lon_bnds            = [0, 0]        # Longitude boundaries.
lat_bnds            = [0, 0]        # Latitude boundaries.
ctrl_pt             = None          # Control point: [longitude, latitude] (this is for statistics.

# Stations.
# Observations are located in directories /exec/<user_name>/<country>/<project>/obs/<obs_provider>/<var>/*.csv
stns                = []            # Station names.

# Reanalysis data.
obs_src             = ""            # Provider of observations or reanalysis set.
obs_src_username    = ""            # Username of account to download ERA5* data.
obs_src_password    = ""            # Password of account to download ERA5* data.

# Variables.
variables           = []            # Variables (based on CORDEX names).
variables_ra        = []            # Variables (based on the names in the reanalysis ensemble).
priority_timestep   = ["day"] * len(variables)

# System ---------------------------------------------------------------------------------------------------------------

# Input-only files and directories.
# The parameter 'p_bounds' is only used to compute statistics; this includes .csv files in the 'stat' directory and
# time series (.png and .csv)). The idea behind this is to export heat maps (.png and .csv) that covers values included
# in the box defined by 'lon_bnds' and 'lat_bnds'.
d_proj              = ""            # Projections.
d_ra_raw            = ""            # Reanalysis set (default frequency, usually hourly).
p_bounds            = ""            # .geogjson file comprising political boundaries.
p_locations         = ""            # .csv file comprising locations.
region              = ""            # Region name or acronym.
# Output-only files and directories.
d_stn               = ""            # Observations or reanalysis.
d_res               = ""            # Results.
# Input and output files and directories.
d_exec              = ""            # Base directory #3 (stations and output).
d_ra_day            = ""            # Reanalysis set (aggregated frequency, i.e. daily).

# File option.
opt_force_overwrite = False         # If True, all NetCDF files and calibration diagrams will be overwritten.

# Log file.
p_log               = ""            # Log file (date and time).
log_n_blank         = 10            # Number of blanks at the beginning of a message.
log_sep_len         = 110           # Number of instances of the symbol "-" in a separator line.
opt_trace           = False         # If True, additional traces are enabled/logged.

# Calibration parameters file.
p_calib             = "calib.csv"   # Calibration file (bias adjustment parameters).

# Performance.
n_proc              = 1             # Number of processes (for multiprocessing).
use_chunks          = False         # If True, use chunks.
pid                 = -1            # Process identifier (primary process)

# Algorithms.
opt_unit_tests      = False         # If True, launch unit tests.

# Step 2 - Download and aggregation ------------------------------------------------------------------------------------

# Download.
opt_download        = False         # If True, download a dataset.
variables_download  = []            # Variables to download.
lon_bnds_download   = [0, 0]        # Longitude boundaries.
lat_bnds_download   = [0, 0]        # Latitude boundaries.

# Aggregation.
opt_aggregate       = False         # If True, aggregate data.

# Steps 3-4 - Data extraction and scenarios ----------------------------------------------------------------------------

opt_ra                  = False      # If True, the analysis is based on reanalysis data.

# Scenarios.
opt_scen                = True      # If True, produce climate scenarios.
radius                  = 0.5       # Search radius (around any given location).
detrend_order           = None      # TODO.MAB: Seems to be not working.

# Patch.
# Simulation excluded from the analysis.
# Ex1: "RCA4_AFR-44_ICHEC-EC-EARTH_rcp85",
# Ex2: "RCA4_AFR-44_MPI-M-MPI-ESM-LR_rcp85",
# Ex3: "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp45",
# Ex4: "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp85"
sim_excepts = []
# Simulation-variable combinations excluded from the analysis.
# Ex1: "pr_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85",
# Ex2: "tasmin_REMO2009_AFR-44_MIROC-MIROC5_rcp26"
var_sim_excepts = []

# Step 5 - Bias adjustment and statistical downscaling -----------------------------------------------------------------

# Calibration options.
opt_calib           = True          # If True, explores the sensitivity to nq, up_qmf and time_win parameters.
opt_calib_auto      = True          # If True, calibrates for nq, up_qmf and time_win parameters.
opt_calib_bias      = True          # If True, examines bias correction.
opt_calib_bias_meth = "rrmse"       # Error quantification method (select one of the following methods).
opt_calib_coherence = False         # If True, examines physical coherence.
opt_calib_qqmap     = True          # If true, calculate qqmap.
opt_calib_perturb   = []            # Perturbation: list of [variable,value]. "*" applies to all variables.
opt_calib_quantiles  = [1.00, 0.99, 0.90, 0.50, 0.10, 0.01, 0.00]  # Quantiles.

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
idx_codes           = []            # Index codes.
idx_names           = []            # Index names.
idx_params          = []            # Index parameters.

# Step 7 - Statistics --------------------------------------------------------------------------------------------------

opt_stat            = [True] * 2        # If True, calculate statistics for [scenarios, indices].
opt_stat_quantiles  = [1.00, 0.90, 0.50, 0.10, 0.00]  # Quantiles.
opt_stat_clip       = False             # If True, clip according to 'p_bounds'.
opt_save_csv        = [False] * 2       # If True, save results to CSV files for [scenarios, indices].

# Step 8 - Visualization -----------------------------------------------------------------------------------------------

# Plots.
opt_plot            = [True] * 2   # If True, generate plots, except time series for [scenarios, indices].
opt_ts              = [True] * 2   # If True, generate time series for [scenarios, indices].
opt_cycle           = [True] * 2   # If Ture, generate cycle plots for [scenarios, indices].
opt_map             = [False] * 2  # If True, generate heat maps [for scenarios, for indices].
opt_map_delta       = [False] * 2  # If True, generate delta heat maps for [scenarios, indices].
opt_map_clip        = False        # If True, clip according to 'p_bounds'.
opt_map_quantiles   = []           # Quantiles for which a map is required.
opt_map_formats     = [f_png]      # Map formats.
opt_map_spat_ref    = ""           # Spatial reference (starts with: EPSG).
opt_map_res         = -1           # Map resolution.
opt_map_discrete    = False        # If true, discrete color scale in maps (rather than continuous).

# Color associated with specific datasets (calibration plots).
col_sim_adj_ref     = "blue"       # Simulation (bias-adjusted) for the reference period.
col_sim_ref         = "orange"     # Simulation (non-adjusted) for the reference period
col_obs             = "green"      # Observations.
col_sim_adj         = "red"        # Simulation (bias-adjusted).
col_sim_fut         = "purple"     # Simulation (non-adjusted) for the future period.

"""
Color maps apply to categories of variables and indices.
+----------------------------+------------+------------+
| Variable, category         |   Variable |      Index |
+----------------------------+------------+------------+
| temperature, high values   | temp_var_1 | temp_idx_1 |
| temperature, low values    |          - | temp_idx_2 |
| precipitation, high values | prec_var_1 | prec_idx_1 |
| precipitation, low values  |          - | prec_idx_2 |
| precipitation, dates       |          - | prec_idx_3 |
| wind                       | wind_var_1 | wind_idx_1 |
+----------------------------+------------+------------+

Notes:
- The 1st scheme is for absolute values.
- The 2nd scheme is divergent and his made to represent delta values when both negative and positive values are
  present.
  It combines the 3rd and 4th schemes.
- The 3rd scheme is for negative-only delta values.
- The 4th scheme is for positive-only delta values.
"""

opt_map_col_temp_var   = []        # Temperature variables.
opt_map_col_temp_idx_1 = []        # Temperature indices (high).
opt_map_col_temp_idx_2 = []        # Temperature indices (low).
opt_map_col_prec_var   = []        # Precipitation variables.
opt_map_col_prec_idx_1 = []        # Precipitation indices (high).
opt_map_col_prec_idx_2 = []        # Precipitation indices (low).
opt_map_col_prec_idx_3 = []        # Precipitation indices (other).
opt_map_col_wind_var   = []        # Wind variables.
opt_map_col_wind_idx_1 = []        # Wind indices.
opt_map_col_default    = []        # Other variables and indices.


def get_rank_inst():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the token corresponding to the institute.
    --------------------------------------------------------------------------------------------------------------------
    """

    return len(d_proj.split(sep))


def get_rank_gcm():

    """
    --------------------------------------------------------------------------------------------------------------------
    Get the rank of token corresponding to the GCM.
    --------------------------------------------------------------------------------------------------------------------
    """

    return get_rank_inst() + 1


def get_d_stn(
    var: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get directory of station data.

    Parameters
    ----------
    var : str
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    d = ""
    if var != "":
        d = d_stn + var + sep

    return d


def get_p_stn(
    var: str,
    stn: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get path of station data.

    Parameters
    ----------
    var : str
        Variable.
    stn : str
        Station.
    --------------------------------------------------------------------------------------------------------------------
    """

    p = d_stn + var + sep + var + "_" + stn + f_ext_nc

    return p


def get_d_scen(
    stn: str,
    cat: str,
    var: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get scenario directory.

    Parameters
    ----------
    stn : str
        Station.
    cat : str
        Category.
    var : str, optional
        Variable.
    --------------------------------------------------------------------------------------------------------------------
    """

    d = d_res
    if stn != "":
        d = d + cat_stn + sep + stn + ("_" + region if region != "" else "") + sep
    if cat != "":
        d = d
        if cat in [cat_obs, cat_raw, cat_regrid, cat_qqmap, cat_qmf, "*"]:
            d = d + cat_scen + sep
        d = d + cat + sep
    if var != "":
        d = d + var + sep

    return d


def get_d_idx(
    stn: str,
    idx_name: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get index directory.

    Parameters
    ----------
    stn : str
        Station.
    idx_name : str, optional
        Index name.
    --------------------------------------------------------------------------------------------------------------------
    """

    d = d_res
    if stn != "":
        d = d + cat_stn + sep + stn + ("_" + region if region != "" else "") + sep
    d = d + cat_idx + sep
    if idx_name != "":
        d = d + idx_name + sep

    return d


def get_p_obs(
    stn_name: str,
    var: str,
    cat: str = ""
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Get observation path (under scenario directory).

    Parameters
    ----------
    stn_name : str
        Localisation.
    var : str
        Variable.
    cat : str, optional
        Category.
    --------------------------------------------------------------------------------------------------------------------
    """

    p = get_d_scen(stn_name, cat_obs) + var + sep + var + "_" + stn_name
    if cat != "":
        p = p + "_4qqmap"
    p = p + f_ext_nc

    return p


def explode_idx_l(
    idx_group_l
) -> [str]:

    """
    --------------------------------------------------------------------------------------------------------------------
    Explode a list of index names or codes, e.g. [rain_season_1, rain_season_2] into
    [rain_season_start_1, rain_season_end_1, rain_season_length_1, rain_season_prcptot_1,
     rain_season_start_2, rain_season_end_2, rain_season_length_2, rain_season_prcptot_2].

    Parameters
    ----------
    idx_group_l : [str]
        List of climate index groups.
    --------------------------------------------------------------------------------------------------------------------
    """

    idx_l_new = []

    # Loop through index names or codes.
    for i in range(len(idx_group_l)):
        idx_code = idx_group_l[i]

        # Loop through index groups.
        in_group = False
        for j in range(len(vi.i_groups)):

            # Explode and add to list.
            if vi.i_groups[j][0] in idx_code:
                in_group = True

                # Extract instance number of index (ex: "_1").
                no = idx_code.replace(idx_code, "")

                # Loop through embedded indices.
                for k in range(len(vi.i_groups[j][1])):
                    idx_l_new.append(vi.i_groups[j][1][k] + no)

        if not in_group:
            idx_l_new.append(idx_code)

    return idx_l_new


def get_equivalent_idx_path(
    p: str,
    vi_code_a: str,
    vi_code_b: str,
    stn: str,
    rcp: str
) -> str:

    """
    --------------------------------------------------------------------------------------------------------------------
    Determine the equivalent path for another variable or index.

    Parameters
    ----------
    p : str
        Path associated with 'vi_code_a'.
    vi_code_a : str
        Climate variable or index to be replaced.
    vi_code_b : str
        Climate variable or index to replace with.
    stn : str
        Station name.
    rcp : str
        Emission scenario.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine if we have variables or indices.
    vi_a = vi.VarIdx(vi_code_a)
    vi_b = vi.VarIdx(vi_code_b)
    a_is_var = vi_a.get_name() in variables
    b_is_var = vi_b.get_name() in variables
    fn = os.path.basename(p)

    # No conversion required.
    if vi_code_a != vi_code_b:

        # Variable->Variable or Index->Index.
        if (a_is_var and b_is_var) or (not a_is_var and not b_is_var):
            p = p.replace(str(vi_a.get_name()), str(vi_b.get_name()))

        # Variable->Index (or the opposite)
        else:
            # Variable->Index.
            if a_is_var and not b_is_var:
                p = get_d_idx(stn, vi_code_b)
            # Index->Variable.
            else:
                if rcp == rcp_def.rcp_ref:
                    p = get_d_stn(vi_code_b)
                else:
                    p = get_d_scen(stn, cat_qqmap, vi_code_b)
            # Both.
            if rcp == rcp_def.rcp_ref:
                p += str(vi_b.get_name()) + "_" + rcp_def.rcp_ref + f_ext_nc
            else:
                p += fn.replace(str(vi_a.get_name()) + "_", str(vi_b.get_name()) + "_")

    return p
