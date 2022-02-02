# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script configuration.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import ast
import configparser
import datetime
import os
import os.path
import pandas as pd
import sys
from typing import Optional

# Workflow libraries.
from def_constant import const as c

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import def_context

"""
------------------------------------------------------------------------------------------------------------------------
File dependencies:

workflow
|
+- def_constant, def_context, def_varidx
|
+- aggregate
|
+- download
|  +- def_constant, def_context, def_varidx, file_utils*
|
+- file_utils*
|
+- indices <-----------------------------------------------------------------------------+
|  +- def_constant, def_context, def_varidx, def_rcp, file_utils*, statistics, utils*    |
|                                                                     |                  |
+- scenarios                                                          |                  |
|  |                                                                  |                  |
|  +- def_constant, def_context, def_varidx, file_utils*, plot ------ | ----+            |
|  |                                                                  |     |            |
|  +- quantile_mapping                                                |     |            |
|  |  +- def_constant                                                 |     |            |
|  |                                                                  |     |            |
|  +- statistics <----------------------------------------------------+     |            |
|  |  +- def_constant, def_context, def_delta, def_project,                 |            |
|  |  |  def_hor, def_lib, def_rcp, def_sim, def_stat,                      |            |
|  |  |  def_varidx, dev_view, dash_plot, file_utils*                       |            |
|  |  |                                                                     |            |
|  |  +- plot <-------------------------------------------------------------+            |
|  |  |  +- def_constant, def_context, def_varidx, file_utils*, utils*                   |
|  |  |                                                                                  |
|  |  +- utils*                                                                          |
|  |                                                                                     |
|  +- utils*                                    +----------------------------------------+
|                                               |
+- test                                         |
|  +- def_constant, def_varidx, file_utils*, indices, utils*
|
+- utils
   +- def_constant, def_context, file_utils*
   
def_context
|
+- def_constant, def_context(dash)

def_constant
|
+- def_constant(dash)
   
file_utils
|
+- def_constant, def_context, utils*

------------------------------------------------------------------------------------------------------------------------
Class hierarchy:

+- Context
|  |
|  +- Context(dash)
|
+- Constant
   |
   +- Constant(dash)
   
------------------------------------------------------------------------------------------------------------------------
"""


class Context(def_context.Context):

    """
    --------------------------------------------------------------------------------------------------------------------
    Class defining the object context.
    --------------------------------------------------------------------------------------------------------------------
    """

    def __init__(
        self,
        code: str
    ):
        """
        ----------------------------------------
        Constructor.
        ----------------------------------------
        """

        super(Context, self).__init__(code)

        """
        Context ------------------------------------
        """

        # Country.
        self.country = ""

        # Projet name.
        self.project = ""

        # Region name or acronym.
        self.region = ""

        # Emission scenarios to be considered.
        self.rcps = [c.rcp26, c.rcp45, c.rcp85]

        # Reference period.
        self.per_ref = [1981, 2010]

        # Future period.
        self.per_fut = [1981, 2100]

        # Horizons.
        self.per_hors = [[1981, 2010], [2021, 2040], [2041, 2060], [2061, 2080]]

        # Boundaries.
        self.lon_bnds = [0, 0]
        self.lat_bnds = [0, 0]

        # Control point: [longitude, latitude] (required for statistics).
        self.ctrl_pt = None

        # Station names.
        self.stns = []

        # Provider of observations or reanalysis data.
        self.obs_src = ""

        # Credentials to access ERA5* data.
        self.obs_src_username = ""
        self.obs_src_password = ""

        # Variables (based on CORDEX names).
        self.variables = []
        self.vars = None

        # Variables (based on the names in the reanalysis ensemble).
        self.vars_ra = None

        self.priority_timestep = ["day"] * 0

        """
        File system --------------------------------
        """

        # Directory of projections.
        self.d_proj = ""

        # Directory of reanalysis set (default frequency, usually hourly).
        self.d_ra_raw = ""

        # Directory of reference data (observations or reanalysis) are located in:
        # /exec/<user_name>/<country>/<project>/stn/<obs_src>/<var>/*.csv
        self._d_stn = ""

        # Directory of results.
        self.d_res = ""

        # Base directory.
        self.d_exec = ""

        # Directory of reanalysis data at aggregated frequency (i.e. daily).
        self.d_ra_day = ""

        # Directory separator (default corresponds to Linux/Unix).
        self.sep = "/"

        # Columns separator (only in CSV files containing observations).
        self.f_sep = ","

        # Tells whether files will be overwritten/recalculated.
        self.opt_force_overwrite = False

        # Path of log file.
        self.p_log = ""

        # Enable/disable additional traces.
        self.opt_trace = False

        # Name of file holding bias adjustment results.
        self.p_calib = "calib.csv"

        # Number of processes to be used for the analysis.
        self.n_proc = 1

        # Enable/disable chunking.
        self.use_chunks = False

        # Process identifier (of primary process).
        self.pid = os.getpid()

        # Enable/disable unit tests.
        self.opt_test = False

        """
        Download and aggregation -------------------
        """

        # Enable/disable download of reanalysis datasets.
        self.opt_download = False

        # Variables to download.
        self.opt_download_variables = []
        self.opt_download_vars = None

        # Boundaries.
        self.opt_download_lon_bnds = [0, 0]
        self.opt_download_lat_bnds = [0, 0]

        # Enable/disable data aggregation.
        self.opt_aggregate = False

        """
        Data extraction and scenarios --------------
        """

        # Tells wether the analysis is based on reanalysis data.
        self.opt_ra = False

        # Enable/disable the production of climate scenarios.
        self.opt_scen = True

        # Search radius (around any given location).
        self.radius = 0.5

        # Simulations excluded from the analysis.
        # Ex1: "RCA4_AFR-44_ICHEC-EC-EARTH_rcp85",
        # Ex2: "RCA4_AFR-44_MPI-M-MPI-ESM-LR_rcp85",
        # Ex3: "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp45",
        # Ex4: "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp85"
        self.sim_excepts = []

        # Simulation-variable combinations excluded from the analysis.
        # Ex1: "pr_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85",
        # Ex2: "tasmin_REMO2009_AFR-44_MIROC-MIROC5_rcp26"
        self.var_sim_excepts = []

        """
        Bias adjustment and statistical downscaling
    
        There are 3 calibration modes.
        Note that a CSV file containing calibration parameters (corresponding to the variable 'p_calib') is generated
        when using modes 2 and 3. This file will automatically be loaded the next time the script runs. In that 
        situation, calibration could be disabled if previous values are still valid.
    
        Calibration mode #1: no calibration.
        a. Set a value for each of the following variable:
           nq_default       = ...
           up_qmf_default   = ...
           time_win_default = ...
        b. Set the following options:
           opt_calib      = False
           opt_calib_auto = False
        c. Run the script.
    
        Calibration mode #2: manual (default mode):
        a. Run function 'scenarios.init_calib_params()' so that it generates a calibration file.
        b. Set a list of values to each of the following parameters:
           nq_calib       = [<value_i>, ..., <value_n>]
           up_qmf_calib   = [<value_i>, ..., <value_n>]
           time_win_calib = [<value_i>, ..., <value_n>]
           bias_err_calib = [<value_i>, ..., <value_n>]
        c. Set the following options:
           opt_calib      = True
           opt_calib_auto = False
           opt_idx        = False
        d. Run the script.
           This will adjust bias for each combination of three parameter values.
        e. Examine the plots that were generated under the following directory:
           d_scen(<station_name>, "fig")/calibration/
           and select the parameter values that produce the best fit between simulations and observations.
           There is one parameter value per simulation, per station, per variable.
        f. Set values for the following parameters:
           nq_default       = <value in cntx.nq_calib producing the best fit>
           up_qmf_default   = <value in cntx.up_qmf_calib producing the best fit>
           time_win_default = <value in cntx.time_win_calib producing the best fit>
           bias_err_default = <value in cntx.bias_err_calib producing the best fit>
        g. Set the following options:
           opt_calib      = False
           opt_calib_auto = False
           opt_idx        = True
        h. Run the script again.
    
        Calibration mode #3: automatic:
        This mode will take much longer to run, but it will fit observations much better.
        a. Set a list of values to each of the following parameters:
           nq_calib       = [<value_i>, ..., <value_n>]
           up_qmf_calib   = [<value_i>, ..., <value_n>]
           time_win_calib = [<value_i>, ..., <value_n>]
           bias_err_calib = [<value_i>, ..., <value_n>]
        b. Set the following options:
           opt_calib      = True
           opt_calib_auto = True
        c. Run the script.
    
        The parameter 'time_win' is the number of days before and after any given day (15 days before and after =
        30 days). This needs to be adjusted as there is period of adjustment between cold period and monsoon. It's
        possible that a very small precipitation amount be considered extreme. We need to limit correction factors.
        """

        # Enable/disable bias adjustment.
        self.opt_calib = True

        # Enable/disable automatic determination of 'nq', 'up_qmf' and 'time_win' parameters.
        self.opt_calib_auto = True

        # Enable/disable bias adjustment.
        self.opt_calib_bias = True

        # Error quantification method (see the other options in def_constant.py).
        self.opt_calib_bias_meth = "rrmse"

        # Enable/disable the calculation of qqmap.
        self.opt_calib_qqmap = True

        # Slight data perturbation: list of [variable, value]. "*" applies to all variables.
        self.opt_calib_perturb = []

        # Quantiles to show in the calibration figure.
        self.opt_calib_quantiles = [1.00, 0.99, 0.90, 0.50, 0.10, 0.01, 0.00]

        # Default 'nq' value.
        self.nq_default = 50

        # Default 'up_qmf' value.
        self.up_qmf_default = 3.0

        # Default 'time_win' value.
        self.time_win_default = 30

        # Default 'bias_err' value.
        self.bias_err_default = -1

        # List of 'nq' values to test during bias adjustment.
        self.nq_calib = [self.nq_default]

        # List of 'up_wmf' values to test during bias adjustment.
        self.up_qmf_calib = [self.up_qmf_default]

        # List of 'time_win' values to test during bias adjustment.
        self.time_win_calib = [self.time_win_default]

        # List of 'bias_err' values to test during bias adjustment.
        self.bias_err_calib = [self.bias_err_default]

        # Container for calibration parameters (pandas dataframe).
        self.df_calib = None

        """
        Indices
    
        => Temperature :
    
        etr: Extreme temperature range.
        Requirements: tasmin, tasmax
        Parameters:   [nan]
    
        tx90p: Number of days with extreme maximum temperature (> 90th percentile).
        Requirements: tasmax
        Parameters:   [nan]
    
        heat_wave_max_length: Maximum heat wave length.
        Requirements: tasmin, tasmax
        Parameters:   [tasmin_thresh: float, tasmax_thresh: float, n_days: int]
                      0.tasmin_thresh: daily minimum temperature must be greater than a threshold value.
                      1.tasmax_thresh: daily maximum temperature must be greater than a threshold value.
                      2.n_days: minimum number of consecutive days. 
    
        heat_wave_total_len: Total heat wave length.
        Requirements: tasmin, tasmax
        Parameters:   [tasmin_thresh: float, tasmax_thresh: float, n_days: int]
                      0.tasmin_thresh: daily minimum temperature must be greater than a threshold value.
                      1.tasmax_thresh: daily maximum temperature must be greater than a threshold value.
                      2.n_days: minimum number of consecutive days.
    
        hot_spell_frequency: Number of hot spells.
        Requirements: tasmax
        Parameters:   [tasmax_thresh: float, n_days: int]
                      0.tasmax_thresh: daily maximum temperature must be greater than a threshold value.
                      1.n_days: minimum number of consecutive days.
    
        hot_spell_max_length: Maximum hot spell length.
        Requirements: tasmax
        Parameters:   [tasmax_threshold: float, n_days: int]
                      0.tasmax_threshold: daily maximum temperature must be greater than this threshold.
                      1.n_days: minimum number of consecutive days.
    
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
                      0.tasmin_thresh: daily minimum temperature must be greater than a threshold value.
    
        tx_days_above: Number of days per year with maximum temperature above a threshold.
        Requirements: tasmax
        Parameters:   [tasmax_thresh: float]
                      0.tasmax_thresh: daily maximum temperature must be greater than a threshold value.
    
        tn_days_below: Number of days per year with a minimum temperature below a threshold.
        Requirements: tasmin
        Parameters:   [tasmin_thresh: float, doy_min: int, doy_max: int]
                      0.tasmin_thresh: daily minimum temperature must be greater than a threshold value.
                      1.doy_min: minimum day of year to consider.
                      2.doy_max: maximum day of year to consider.
    
        tropical_nights: Number of tropical nights, i.e. with minimum temperature above a threshold.
        Requirements: tasmin
        Parameters:   [tasmin_thresh: float]
                      0.tasmin_thresh: daily minimum temperature must be greater than a threshold value.
    
        wsdi: Warm spell duration index.
        Requirements: tasmax
        Parameters:   [tasmax_thresh=nan, n_days: int]
                      0.tasmax_thresh: daily maximum temperature must be greater than a threshold value; this value is
                        calculated automatically and corresponds to the 90th percentile of tasmax values.
                      1.n_days: minimum number of consecutive days.
    
        => Precipitation :
    
        rx1day: Largest 1-day precipitation amount.
        Requirements: pr
        Parameters:   [nan]
    
        rx5day: Largest 5-day precipitation amount.
        Requirements: pr
        Parameters:   [nan]
    
        cdd: Maximum number of consecutive dry days.
        Requirements: pr
        Parameters:   [pr_thresh: float]
                      0.pr_thresh: a daily precipitation amount lower than a threshold value is considered dry.
    
        cwd: Maximum number of consecutive wet days.
        Requirements: pr
        Parameters:   [pr_thresh: float]
                      0.pr_thresh: a daily precipitation amount greater than or equal to a threshold value is considered
                        wet.
    
        dry_days: Number of dry days.
        Requirements: pr
        Parameters:   [pr_thresh: float]
                      0.pr_thresh: a daily precipitation amount lower than a threshold value is considered dry.
    
        wet_days: Number of wet days.
        Requirements: pr
        Parameters:   [pr_thresh: float]
                      0.pr_thresh: a daily precipitation amount greater than or equal to a threshold value is considered
                        wet.
    
        prcptot: Accumulated total precipitation.
        Requirements: pr
        Parameters:   [pct=nan, doy_min: int, doy_max: int]
                      0.pct: the default value is nan; if a value is provided, an horizontal line is drawn on time
                        series; if a percentile is provided (ex: 90p), the equivalent value is calculated and an
                        horizontal line is drawn on time series.
                      1.doy_min: minimum day of year to consider.
                      2.doy_max: maximum day of year to consider.
    
        r10mm: Number of days with precipitation greater than or equal to 10 mm.
        Requirements: pr
        Parameters:   [nan]
    
        r20mm: Number of days with precipitation greater than or equal to 20 mm.
        Requirements: pr
        Parameters:   [nan]
    
    
        rnnmm: Number of days with precipitation greater than or equal to a user-provided value.
        Requirements: pr
        Parameters:   [pr_thresh: float]
                      0.pr_thresh: daily precipitation amount must be greater than or equal to a threshold value.
    
        sdii: Mean daily precipitation intensity.
        Requirements: pr
        Parameters:   [pr_thresh: float]
                      0.pr_thresh: daily precipitation amount must be greater than or equal to a threshold value.
    
        rain_season: Rain season.
        Requirements: pr (mandatory), evspsbl* (optional)
        Parameters:   Combination of the parameters of indices idx_rain_season_start and idx_rain_season_end.
    
        rain_season_start: Day of year where rain season starts.
        Requirements: pr
        Parameters:   [thresh_wet: str, window_wet: int, thresh_dry: str, dry_days_max: int, window_dry: int,
                      0.start_date: str, end_date: str, freq: str]
                      1.thresh_wet: accumulated precipitation threshold associated with {window_wet}.
                      2.window_wet: number of days where accumulated precipitation is above {thresh_wet}.
                      3.thresh_dry: daily precipitation threshold associated with {window_dry}.
                      4.dry_days: maximum number of dry days in {window_dry}.
                      5.window_dry: number of days, after {window_wet}, during which daily precipitation is not greater
                        than or equal to {thresh_dry} for {dry_days} consecutive days.
                      6.start_date: first day of year where season can start ("mm-dd").
                      7.end_date: last day of year where season can start ("mm-dd").
                      8.freq: resampling frequency.
    
        rain_season_end: Day of year where rain season ends.
        Requirements: pr (mandatory), rain_season_start_next (optional), evspsbl* (optional)
                      will search for evspsblpot, then for evspsbl
        Parameters:   [method: str, thresh: str, window: int, etp_rate: str, start_date: str, end_date: str, freq: str]
                      0.op: Resampling operator = {"max", "sum", "etp}
                        . if "max": based on the occurrence (or not) of an event during the last days of a rain season;
                          rain season ends when no daily precipitation greater than {thresh} have occurred over a period
                          of {window} days.
                        . if "sum": based on a total amount of precipitation received during the last days of the rain
                          season; rain season ends when the total amount of precipitation is less than {thresh} over a
                          period of {window} days.
                        . if "etp": calculation is based on the period required for a water column of height {thresh] to
                          evaporate, considering that any amount of precipitation received during that period must
                          evaporate as well; if {etp} is not available, evapotranspiration rate is assumed to be
                          {etp_rate}.
                      1.thresh: maximum or accumulated precipitation threshold associated with {window}.
                        . if {op} == "max": maximum daily precipitation  during a period of {window} days.
                        . if {op} == "sum": accumulated precipitation over {window} days.
                        . if {op} == "etp": height of water column that must evaporate
                      2.window: int
                        . if {op} in ["max", "sum"]: number of days used to verify if the rain season is ending.
                      3.etp_rate:
                        . if {op} == "etp": evapotranspiration rate.
                        . else: not used.
                      4.start_date: First day of year where season can end ("mm-dd").
                      5.end_date: Last day of year where season can end ("mm-dd").
                      6.freq: Resampling frequency.
    
        rain_season_length: Duration of the rain season.
        Requirements: rain_season_start, rain_season_end
        Parameters:   [nan]
    
        rain_season_prcptot: Quantity received during rain season.
        Requirements: pr, rain_season_start, rain_season_end
        Parameters:   [nan]
    
        dry_spell_total_length: Total length of dry period.
        Requirements: pr
        Parameters:   [thresh: str, window: int, op: str, start_date: str, end_date: str]
                      0.thresh: Precipitation threshold
                      1.op: Period over which to combine data: "max" = one day, "sum" = cumulative over {window} days.
                        . if {op} == "max": daily precipitation amount under which precipitation is considered
                          negligible.
                        . if {op} == "sum": sum of daily precipitation amounts under which the period is considered dry.
                      2.window: minimum number of days required in a dry period.
                      3.start_date: first day of year to consider ("mm-dd").
                      4.end_date: last day of year to consider ("mm-dd").
    
        => Temperature-precipitation :
    
        drought_code: Drought code.
        Requirements: tas, pr
        Parameters:   [nan]
    
        => Wind :
    
        wg_days_above: Number of days per year with mean wind speed above a threshold value coming from a given
                       direction.
        Requirements: uas, vas
        Parameters:   [speed_tresh: float, velocity_thresh_neg: float, dir_thresh: float, dir_thresh_tol: float,
                       doy_min: int, doy_max: int]
                      0.speed_tresh: Wind speed must be greater than or equal to a threshold value.
                        . if a percentile is provided (ex: 90p), the equivalent value is calculated.
                      1.speed_tresh_neg: wind speed is considered negligible if smaller than or equal to a threshold
                        value.
                      2.dir_thresh: wind direction (angle, in degrees) must be close to a direction given by a threshold
                        value.
                      3.dir_thresh_tol: wind direction tolerance (angle, in degrees).
                      4.doy_min: minimum day of year to consider (nan can be provided).
                      5.doy_max: maximum day of year to consider (nan can be provided).
    
        wx_days_above: Number of days per year with maximum wind speed above a threshold value.
        Requirements: sfcWindmax
        Parameters:   [speed_tresh: float, nan, nan, nan, doy_min: int, doy_max: int]
                      0.speed_tresh: Wind speed must be greater than or equal to a threshold value.
                        . if a percentile is provided (ex: 90p), the equivalent value is calculated.
                      1.doy_min: minimum day of year to consider (nan can be provided).
                      2.doy_max: maximum day of year to consider (nan can be provided).
        """

        # Enable/disable the calculation of climate indices.
        self.opt_idx = True

        # Spatial resolution for mapping.
        self.idx_resol = 0.05

        # Indices.
        self.idxs = None

        """
        Statistics -----------------------------
        """

        # Enable/disable the calculation of statistics for [scenarios, indices].
        self.opt_stat = [True] * 2

        # Quantiles.
        self.opt_stat_quantiles = [1.00, 0.90, 0.50, 0.10, 0.00]

        # Enable/disable clipping according to 'p_bounds'.
        self.opt_stat_clip = False

        # Enabe/disable exporting NetCDF to CSV files for [scenarios, indices].
        self.export_nc_to_csv = [False] * 2

        """
        Visualization --------------------------
        """

        # Enable/disable diagnostic plots (related to bias adjustment).
        self.opt_diagnostic = True

        # Format of diagnostic plots.
        self.opt_diagnostic_format = ["png", "csv"]

        # Enable/disable generation of annual and monthly cycle plots for [scenarios, indices].
        self.opt_cycle = True

        # Format of cycle plots.
        self.opt_cycle_format = ["png", "csv"]

        # Enable/diasble generation of time series for [scenarios, indices].
        self.opt_ts = [True] * 2

        # Enable/disable generation of bias plots for [scenarios].
        self.opt_ts_bias = [True]

        # Format of time series.
        self.opt_ts_format = ["png", "csv"]

        # Enable/disable generation of maps [for scenarios, for indices].
        self.opt_map = [False] * 2

        # Enable/disable generationg of delta maps for [scenarios, indices].
        self.opt_map_delta = [False] * 2

        # Tells whether map clipping should be done using 'p_bounds'.
        self.opt_map_clip = False

        # Quantiles for which a map is required.
        self.opt_map_quantiles = []

        # Format of maps.
        self.opt_map_format = ["png", "csv"]

        # Map locations (pd.DataFrame).
        self.opt_map_locations = None

        # Spatial reference (starts with: EPSG).
        self.opt_map_spat_ref = ""

        # Map resolution.
        self.opt_map_res = -1

        # Tells whether discrete color scale are used maps (rather than continuous).
        self.opt_map_discrete = False

        # Enable/diasble generation of cluster plots (only applicable to scenarios).
        self.opt_cluster = False

        # Variables used for clustering.
        self.opt_cluster_variables = []
        self.opt_cluster_vars = None

        # Format of cluster plots.
        self.opt_cluster_format = ["png", "csv"]

        # Color scale of cluster plots.
        self.opt_cluster_col = "Dark2"

        """
        Color maps apply to categories of variables and indices.
    
        Variable       |  Category     |  Variable    |  Index
        ---------------+---------------+--------------+------------
        temperature    |  high values  |  temp_var_1  |  temp_idx_1
        temperature    |  low values   |  -           |  temp_idx_2
        precipitation  |  high values  |  prec_var_1  |  prec_idx_1
        precipitation  |  low values   |  -           |  prec_idx_2
        precipitation  |  dates        |  -           |  prec_idx_3
        wind           |               |  wind_var_1  |  wind_idx_1
    
        Notes:
        - The 1st scheme is for absolute values.
        - The 2nd scheme is divergent and his made to represent delta values when both negative and positive values are
          present.
          It combines the 3rd and 4th schemes.
        - The 3rd scheme is for negative-only delta values.
        - The 4th scheme is for positive-only delta values.
        """

        # Temperature variables.
        self.opt_map_col_temp_var = []

        # Temperature indices (high).
        self.opt_map_col_temp_idx_1 = []

        # Temperature indices (low).
        self.opt_map_col_temp_idx_2 = []

        # Precipitation variables.
        self.opt_map_col_prec_var = []

        # Precipitation indices (high).
        self.opt_map_col_prec_idx_1 = []

        # Precipitation indices (low).
        self.opt_map_col_prec_idx_2 = []

        # Precipitation indices (other).
        self.opt_map_col_prec_idx_3 = []

        # Wind variables.
        self.opt_map_col_wind_var = []

        # Wind indices.
        self.opt_map_col_wind_idx_1 = []

        # Other variables and indices.
        self.opt_map_col_default = []

        # Number of processes
        self.n_proc = 1

    def load(
        self,
        p_ini: str = "config.ini"
    ):

        """
        ----------------------------------------
        Load parameters from an INI file.

        Parameters
        ----------
        p_ini : str
            Path of INI file.
        ----------------------------------------
        """

        config = configparser.ConfigParser()
        config.read(p_ini)

        # Track if a few variables were read.
        per_ref_read = False
        per_hors_read = False

        # Loop through sections.
        for section in config.sections():

            # Loop through keys.
            for key in config[section]:

                # Extract value.
                value = config[section][key]

                # Project.
                if key == "country":
                    self.country = ast.literal_eval(value)
                elif key == "project":
                    self.project = ast.literal_eval(value)

                # Observations or reanalysis.
                elif key == "obs_src":
                    self.obs_src = ast.literal_eval(value)
                    self.opt_ra = (self.obs_src == c.ens_era5) or (self.obs_src == c.ens_era5_land) or \
                                  (self.obs_src == c.ens_merra2) or (self.obs_src == c.ens_enacts)
                elif key == "obs_src_username":
                    self.obs_src_username = ast.literal_eval(value)
                elif key == "obs_src_password":
                    self.obs_src_password = ast.literal_eval(value)
                elif key == "file_sep":
                    self.f_sep = ast.literal_eval(value)
                elif key == "stns":
                    self.stns = def_context.str_to_arr_1d(value, str)

                # Context.
                elif key == "rcps":
                    self.rcps = ast.literal_eval(value)
                elif key == "per_ref":
                    self.per_ref = def_context.str_to_arr_1d(value, int)
                    if per_hors_read:
                        self.per_hors = [self.per_ref] + self.per_hors
                    per_ref_read = True
                elif key == "per_fut":
                    self.per_fut = def_context.str_to_arr_1d(value, int)
                elif key == "per_hors":
                    self.per_hors = def_context.str_to_arr_2d(value, int)
                    if per_ref_read:
                        self.per_hors = [self.per_ref] + self.per_hors
                    per_hors_read = True
                elif key == "lon_bnds":
                    self.lon_bnds = def_context.str_to_arr_1d(value, float)
                elif key == "lat_bnds":
                    self.lat_bnds = def_context.str_to_arr_1d(value, float)
                elif key == "ctrl_pt":
                    self.ctrl_pt = def_context.str_to_arr_1d(value, float)
                elif key == "variables":
                    self.variables = def_context.str_to_arr_1d(value, str)
                elif key == "p_bounds":
                    self.p_bounds = ast.literal_eval(value)
                elif key == "region":
                    self.region = ast.literal_eval(value)

                # Data:
                elif key == "opt_download":
                    self.opt_download = ast.literal_eval(value)
                elif key == "opt_download_variables":
                    self.opt_download_vars = def_context.str_to_arr_1d(value, str)
                elif key == "opt_download_lon_bnds":
                    self.opt_download_lon_bnds = def_context.str_to_arr_1d(value, float)
                elif key == "opt_download_lat_bnds":
                    self.opt_download_lat_bnds = def_context.str_to_arr_1d(value, float)
                elif key == "opt_aggregate":
                    self.opt_aggregate = ast.literal_eval(value)

                # Climate scenarios.
                elif key == "opt_scen":
                    self.opt_scen = ast.literal_eval(value)
                elif key == "radius":
                    self.radius = float(value)
                elif key == "sim_excepts":
                    self.sim_excepts = def_context.str_to_arr_1d(value, str)
                elif key == "var_sim_excepts":
                    self.var_sim_excepts = def_context.str_to_arr_1d(value, str)

                # Bias adjustment:
                elif key == "opt_calib":
                    self.opt_calib = ast.literal_eval(value)
                elif key == "opt_calib_auto":
                    self.opt_calib_auto = ast.literal_eval(value)
                elif key == "opt_calib_bias":
                    self.opt_calib_bias = ast.literal_eval(value)
                elif key == "opt_calib_bias_meth":
                    self.opt_calib_bias_meth = ast.literal_eval(value)
                elif key == "opt_calib_qqmap":
                    self.opt_calib_qqmap = ast.literal_eval(value)
                elif key == "opt_calib_perturb":
                    self.opt_calib_perturb = def_context.str_to_arr_2d(value, float)
                elif key == "opt_calib_quantiles":
                    self.opt_calib_quantiles = def_context.str_to_arr_1d(value, float)
                elif key == "nq_default":
                    self.nq_default = int(value)
                    self.nq_calib = [self.nq_default]
                elif key == "up_qmf_default":
                    self.up_qmf_default = float(value)
                    self.up_qmf_calib = [self.up_qmf_default]
                elif key == "time_win_default":
                    self.time_win_default = int(value)
                    self.time_win_calib = [self.time_win_default]

                # Climate indices.
                elif key == "opt_idx":
                    self.opt_idx = ast.literal_eval(value)
                elif key == "idx_codes":
                    self.idx_codes = def_context.str_to_arr_1d(value, str)
                elif key == "idx_params":
                    idx_params_tmp = def_context.str_to_arr_2d(value, float)
                    for i in range(len(self.idx_codes)):
                        if self.idx_codes[i] == c.i_r10mm:
                            self.idx_params.append([10])
                        elif self.idx_codes[i] == c.i_r20mm:
                            self.idx_params.append([20])
                        else:
                            self.idx_params.append(idx_params_tmp[i])

                # Statistics.
                elif key == "opt_stat":
                    self.opt_stat = ast.literal_eval(value)\
                        if ("," not in value) else def_context.str_to_arr_1d(value, bool)
                elif key == "opt_stat_quantiles":
                    self.opt_stat_quantiles = def_context.str_to_arr_1d(value, float)
                elif key == "opt_stat_clip":
                    self.opt_stat_clip = ast.literal_eval(value)
                elif key == "export_nc_to_csv":
                    self.export_nc_to_csv = ast.literal_eval(value)\
                        if ("," not in value) else def_context.str_to_arr_1d(value, bool)

                # Visualization:
                elif key == "opt_diagnostic":
                    self.opt_diagnostic = ast.literal_eval(value)
                elif key == "opt_diagnostic_format":
                    self.opt_diagnostic_format = def_context.str_to_arr_1d(value, str)
                elif key == "opt_cycle":
                    self.opt_cycle = ast.literal_eval(value)
                elif key == "opt_cycle_format":
                    self.opt_cycle_format = def_context.str_to_arr_1d(value, str)
                elif key == "opt_ts":
                    self.opt_ts = ast.literal_eval(value)\
                        if ("," not in value) else def_context.str_to_arr_1d(value, bool)
                elif key == "opt_ts_bias":
                    self.opt_ts_bias = ast.literal_eval(value)\
                        if ("," not in value) else def_context.str_to_arr_1d(value, bool)
                elif key == "opt_ts_format":
                    self.opt_ts_format = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map":
                    self.opt_map = [False, False]
                    if self.opt_ra:
                        self.opt_map = ast.literal_eval(value)\
                            if ("," not in value) else def_context.str_to_arr_1d(value, bool)
                elif key == "opt_map_delta":
                    self.opt_map_delta = [False, False]
                    if self.opt_ra:
                        if "," not in value:
                            self.opt_map_delta = ast.literal_eval(value)
                        else:
                            self.opt_map_delta = def_context.str_to_arr_1d(value, bool)
                elif key == "opt_map_clip":
                    self.opt_map_clip = ast.literal_eval(value)
                elif key == "opt_map_quantiles":
                    self.opt_map_quantiles = def_context.str_to_arr_1d(value, float)
                    if str(self.opt_map_quantiles).replace("['']", "") == "":
                        self.opt_map_quantiles = None
                elif key == "opt_map_format":
                    self.opt_map_format = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_locations":
                    self.opt_map_locations = pd.DataFrame(def_context.str_to_arr_2d(value, float),
                                                          columns=["longitude", "latitude", "desc"])
                elif key == "opt_map_spat_ref":
                    self.opt_map_spat_ref = ast.literal_eval(value)
                elif key == "opt_map_res":
                    self.opt_map_res = ast.literal_eval(value)
                elif key == "opt_map_discrete":
                    self.opt_map_discrete = ast.literal_eval(value)
                elif key == "opt_map_col_temp_var":
                    self.opt_map_col_temp_var = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_temp_idx_1":
                    self.opt_map_col_temp_idx_1 = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_temp_idx_2":
                    self.opt_map_col_temp_idx_2 = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_var":
                    self.opt_map_col_prec_var = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_idx_1":
                    self.opt_map_col_prec_idx_1 = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_idx_2":
                    self.opt_map_col_prec_idx_2 = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_idx_3":
                    self.opt_map_col_prec_idx_3 = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_wind_var":
                    self.opt_map_col_wind_var = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_wind_idx_1":
                    self.opt_map_col_wind_idx_1 = def_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_default":
                    self.opt_map_col_default = def_context.str_to_arr_1d(value, str)
                elif key == "opt_cluster":
                    self.opt_cluster = ast.literal_eval(value)
                elif key == "opt_cluster_variables":
                    self.opt_cluster_variables = def_context.str_to_arr_1d(value, str)
                elif key == "opt_cluster_format":
                    self.opt_cluster_format = def_context.str_to_arr_1d(value, str)
                elif key == "opt_cluster_col":
                    self.opt_cluster_col = ast.literal_eval(value)

                # Environment.
                elif key == "n_proc":
                    self.n_proc = int(value)
                    if self.n_proc > 1:
                        self.use_chunks = False
                elif key == "use_chunks":
                    self.use_chunks = ast.literal_eval(value)
                    if self.use_chunks:
                        self.n_proc = 1
                elif key == "d_exec":
                    self.d_exec = ast.literal_eval(value)
                    if "\\" in self.d_exec:
                        self.sep = "\\"
                elif key == "d_proj":
                    self.d_proj = ast.literal_eval(value)
                elif key == "d_ra_raw":
                    self.d_ra_raw = ast.literal_eval(value)
                elif key == "d_ra_day":
                    self.d_ra_day = ast.literal_eval(value)
                elif key == "opt_trace":
                    self.opt_trace = ast.literal_eval(value)
                elif key == "opt_force_overwrite":
                    self.opt_force_overwrite = ast.literal_eval(value)

                # Unit tests.
                elif key == "opt_test":
                    self.opt_test = ast.literal_eval(value)

        # Time step.
        self.priority_timestep = ["day"] * len(self.variables)

        # Directories and paths.
        d_base = self.d_exec + self.country + self.sep + self.project + self.sep
        obs_src_region = self.obs_src + ("_" + self.region if (self.region != "") and self.opt_ra else "")
        self._d_stn = d_base + c.cat_stn + self.sep + obs_src_region + self.sep
        self.d_res = self.d_exec + "sim_climat" + self.sep + self.country + self.sep + self.project + self.sep

        # Boundaries and locations.
        if self.p_bounds != "":
            self.p_bounds = d_base + "gis" + self.sep + self.p_bounds

        # Log file.
        dt = datetime.datetime.now()
        dt_str = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + "_" + \
            str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + str(dt.second).zfill(2)
        self.p_log = self.d_res + "stn" + self.sep + obs_src_region + self.sep + "log" + self.sep + dt_str + ".log"

        # Calibration file.
        self.p_calib = self.d_res + "stn" + self.sep + obs_src_region + self.sep + self.p_calib

    def d_stn(
        self,
        var_name: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get directory of station data.

        Parameters
        ----------
        var_name: Optional[str]
            Variable name.

        Returns
        -------
        str
            Directory of station data.
        ----------------------------------------
        """

        d = ""
        if var_name != "":
            d = self._d_stn + var_name + self.sep

        return d

    def p_stn(
        self,
        var_name: str,
        stn: str
    ) -> str:

        """
        ----------------------------------------
        Get path of station data.

        Parameters
        ----------
        var_name: str
            Variable name.
        stn: str
            Station.

        Returns
        -------
        str
            Path of station data.
        ----------------------------------------
        """

        p = str(self.d_stn(var_name)) + var_name + "_" + stn + c.f_ext_nc

        return p

    def d_scen(
        self,
        stn: str,
        cat: str,
        var_name: str = ""
    ) -> str:

        """
        ----------------------------------------
        Get directory of scenarios.

        Parameters
        ----------
        stn: str
            Station.
        cat: str
            Category.
        var_name: str, optional
            Variable.

        Returns
        -------
        str
            Directory of scenarios.
        ----------------------------------------
        """

        d = self.d_res
        if stn != "":
            d = d + c.cat_stn + self.sep + stn + ("_" + self.region if self.region != "" else "") + self.sep
        if cat != "":
            d = d
            if cat in [c.cat_obs, c.cat_raw, c.cat_regrid, c.cat_qqmap, c.cat_qmf, "*"]:
                d = d + c.cat_scen + self.sep
            d = d + cat + self.sep
        if var_name != "":
            d = d + var_name + self.sep

        return d

    def d_idx(
        self,
        stn: str,
        idx_name: str = ""
    ) -> str:

        """
        ----------------------------------------
        Get directory of indices.

        Parameters
        ----------
        stn : str
            Station.
        idx_name : str, optional
            Index name.

        Returns
        -------
        str
            Directory of indices.
        ----------------------------------------
        """

        d = self.d_res
        if stn != "":
            d = d + c.cat_stn + self.sep + stn + ("_" + self.region if self.region != "" else "") + self.sep
        d = d + c.cat_idx + self.sep
        if idx_name != "":
            d = d + idx_name + self.sep

        return d

    def p_obs(
        self,
        stn_name: str,
        var_name: str,
        cat: str = ""
    ) -> str:

        """
        ----------------------------------------
        Get observation path (under scenario directory).

        Parameters
        ----------
        stn_name: str
            Localisation.
        var_name: str
            Variable name.
        cat: str, optional
            Category.

        Returns
        -------
        str
            Observation path.
        ----------------------------------------
        """

        p = str(self.d_scen(stn_name, c.cat_obs)) + var_name + self.sep + var_name + "_" + stn_name
        if cat != "":
            p = p + "_4qqmap"
        p = p + c.f_ext_nc

        return p

    def rank_inst(
        self
    ) -> int:

        """
        ----------------------------------------
        Get the token corresponding to the institute.

        Returns
        -------
        int
            Token corresponding to the institute.
        ----------------------------------------
        """

        return len(self.d_proj.split(self.sep))

    def rank_gcm(
        self
    ) -> int:

        """
        ----------------------------------------
        Get the rank of token corresponding to the GCM.

        Returns
        -------
        int
            Rank of token corresponding to the GCM.
        ----------------------------------------
        """

        return self.rank_inst() + 1


# Configuration instance.
cntx = Context("script")
cntx.load()
