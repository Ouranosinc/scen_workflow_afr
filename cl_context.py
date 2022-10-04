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
import glob
import os
import os.path
import pandas as pd
import sys
from itertools import compress
from typing import List, Optional, Tuple

# Workflow libraries.
from cl_constant import const as c

# Dashboard libraries.
sys.path.append("scen_workflow_afr_dashboard")
from scen_workflow_afr_dashboard import cl_context

"""
------------------------------------------------------------------------------------------------------------------------
File dependencies:

workflow
|
+- cl_constant, cl_context, cl_varidx
|
+- wf_aggregate
|
+- wf_download
|  +- cl_constant, cl_context, cl_varidx, wf_file_utils*
|
+- wf_file_utils*
|
+- wf_indices <--------------------------------------------------------------------------+
|  +- cl_constant, cl_context, cl_varidx, cl_rcp, wf_file_utils*, wf_stats, wf_utils*    |
|                                                                    |                   |
+- wf_scenarios                                                      |                   |
|  |                                                                 |                   |
|  +- cl_constant, cl_context, cl_varidx, wf_file_utils*, wf_plot -- | -----+            |
|  |                                                                 |      |            |
|  +- wf_qm                                                          |      |            |
|  |  +- cl_constant                                                 |      |            |
|  |                                                                 |      |            |
|  +- wf_stats <-----------------------------------------------------+      |            |
|  |  +- cl_constant, cl_context, cl_delta, cl_project,                     |            |
|  |  |  cl_hor, cl_lib, cl_rcp, cl_sim, cl_stat,                           |            |
|  |  |  cl_varidx, cl_view, dash_plot, wf_file_utils*                      |            |
|  |  |                                                                     |            |
|  |  +- wf_plot <----------------------------------------------------------+            |
|  |  |  +- cl_constant, cl_context, cl_varidx, wf_file_utils*, wf_utils*                |
|  |  |                                                                                  |
|  |  +- wf_utils*                                                                       |
|  |                                                                                     |
|  +- wf_utils*                                +-----------------------------------------+
|                                              |
+- wf_test                                     |
|  +- cl_constant, cl_varidx, wf_file_utils*, wf_indices, wf_utils*
|
+- wf_utils
   +- cl_constant, cl_context, wf_file_utils*
   
cl_context
|
+- cl_constant, cl_context(dash)

cl_constant
|
+- cl_constant(dash)
   
wf_file_utils
|
+- cl_constant, cl_context, wf_utils*

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


class Context(cl_context.Context):

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
        self._project = ""

        # Region name or acronym.
        self.region = ""

        # Emission scenarios to be considered.
        self.emission_scenarios = [c.RCP26, c.RCP45, c.RCP85]
        self.rcps = None

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

        # Ensemble used in bias adjustment (gridded data has priority).
        self.ens_ref = ""

        # Ensemble used for at-a-station reference data, along with station names.
        self.ens_ref_stn = ""
        self.ens_ref_stns = []
        self.ens_ref_col_sep = ","

        # Ensemble used for gridded reference data, along with username and password.
        self.ens_ref_grid = ""
        self.ens_ref_grid_usr = ""
        self.ens_ref_grid_pwd = ""

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

        # Directory of reference data at hourly frequency.
        self.d_ref_hour = ""

        # Base directory.
        self.d_exec = ""

        # Directory of reference data at daily frequency.
        self.d_ref_day = ""

        # Directory separator (default corresponds to Linux/Unix).
        self.sep = "/"

        # Tells whether files will be overwritten/recalculated.
        self.opt_force_overwrite = False

        # Path of log file.
        self.p_log = ""

        # Name of boundary files.
        self.p_bounds = ""

        # Enable/disable additional traces.
        self.opt_trace = False

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

        # Boundaries.
        self.opt_download_lon_bnds = [0, 0]
        self.opt_download_lat_bnds = [0, 0]

        # Download period.
        self.opt_download_per = []

        # Enable/disable data aggregation.
        self.opt_aggregate = False

        """
        Data extraction and scenarios --------------
        """

        # Enable/disable the production of climate scenarios.
        self.opt_scen = True

        # Domains.
        self.domains = ["AFR-44"]

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
        Bias adjustment and statistical downscaling (activated with 'opt_bias')
    
        The CSV file containing bias adjustment parameters (corresponding to the variable 'opt_bias_fn') is generated
        and updated during the adjustment. This file will automatically be loaded the next time the script runs.
    
        'nq':       Number of quantiles.
        'up_qmf:    Upper limit of the quantile mapping function.
        'time_win': This is the number of days before and after any given day (15 days before + 15 days after =
                    30 days). This needs to be adjusted as there is period of adjustment between cold period and
                    monsoon. It's possible that a very small precipitation amount be considered extreme. We need to
                    limit correction factors.
        """

        # Enable/disable bias adjustment.
        self.opt_bias = True

        # Bias adjustment method (only one method is available).
        self.opt_bias_meth = "quantile_mapping"

        # Number of quantiles.
        self.opt_bias_nq = 50

        # Upper limit of quantile mapping function.
        self.opt_bias_up_qmf = 3.0

        # Time window.
        self.opt_bias_time_win = 30

        # Slight data perturbation: list of [variable, value]. "*" applies to all variables.
        self.opt_bias_perturb = []

        # Centiles to show in the bias adjustment figure.
        self.opt_bias_centiles = [100, 99, 90, 50, 10, 1, 0]

        # Error quantification method (see the other options in cl_constant.py).
        self.opt_bias_err_meth = "rrmse"

        # Name of file holding bias adjustment statistics.
        self.opt_bias_fn = "bias.csv"

        # Container for bias adjustment parameters (pd.DataFrame).
        self.opt_bias_df = None

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

        # Indices.
        self.idxs = None

        """
        Results --------------------------------
        """

        # Enable/diasble generation of cluster plots (only applicable to scenarios).
        self.opt_cluster = False

        # Centiles required to produce a cluster scatter plot (if a single variable is selected).
        self.opt_cluster_centiles = [10, 50, 90]

        # Color scale of cluster plots.
        self.opt_cluster_col = "Dark2"

        # Format of cluster plots.
        self.opt_cluster_format = ["png", "csv"]

        # Variables used for clustering.
        self.opt_cluster_variables = []
        self.opt_cluster_vars = None

        # Enable/disable generation of annual and monthly cycle plots for [scenarios, indices].
        self.opt_cycle = True

        # Format of cycle plots.
        self.opt_cycle_format = ["png", "csv"]

        # Enable/disable diagnostic plots (related to bias adjustment).
        self.opt_diagnostic = True

        # Format of diagnostic plots.
        self.opt_diagnostic_format = ["png", "csv"]

        # Enable/disable generation of maps [for scenarios, for indices].
        self.opt_map = [False] * 2

        # Centiles for which a map is required.
        self.opt_map_centiles = [10, 90]

        # Tells whether map clipping should be done using 'p_bounds'.
        self.opt_map_clip = False

        """
        Color maps apply to categories of variables and indices.
    
        Variable       |  Category     |  Variable    |  Index
        ---------------+---------------+--------------+------------
        temperature    |  high values  |  temp_var_1  |  temp_idx_1
        temperature    |  low values   |  -           |  temp_idx_2
        ---------------+---------------+--------------+------------
        precipitation  |  high values  |  prec_var_1  |  prec_idx_1
        precipitation  |  low values   |  -           |  prec_idx_2
        precipitation  |  dates        |  -           |  prec_idx_3
        ---------------+---------------+--------------+------------
        evaporation    |  high values  |  evap_var_1  |  evap_idx_1
        evaporation    |  low values   |  -           |  evap_idx_2
        evaporation    |  dates        |  -           |  evap_idx_3
        ---------------+---------------+--------------+------------
        wind           |               |  wind_var_1  |  wind_idx_1
    
        Notes:
        - The 1st scheme is for absolute values.
        - The 2nd scheme is divergent and his made to represent delta values when both negative and positive values are
          present.
          It combines the 3rd and 4th schemes.
        - The 3rd scheme is for negative-only delta values.
        - The 4th scheme is for positive-only delta values.
        """

        self.opt_map_col_temp_var   = super(Context, self).opt_map_col_temp_var
        self.opt_map_col_temp_idx_1 = super(Context, self).opt_map_col_temp_idx_1
        self.opt_map_col_temp_idx_2 = super(Context, self).opt_map_col_temp_idx_2
        self.opt_map_col_prec_var   = super(Context, self).opt_map_col_prec_var
        self.opt_map_col_prec_idx_1 = super(Context, self).opt_map_col_prec_idx_1
        self.opt_map_col_prec_idx_2 = super(Context, self).opt_map_col_prec_idx_2
        self.opt_map_col_prec_idx_3 = super(Context, self).opt_map_col_prec_idx_3
        self.opt_map_col_evap_var   = super(Context, self).opt_map_col_evap_var
        self.opt_map_col_evap_idx_1 = super(Context, self).opt_map_col_evap_idx_1
        self.opt_map_col_evap_idx_2 = super(Context, self).opt_map_col_evap_idx_2
        self.opt_map_col_evap_idx_3 = super(Context, self).opt_map_col_evap_idx_3
        self.opt_map_col_wind_var   = super(Context, self).opt_map_col_wind_var
        self.opt_map_col_wind_idx_1 = super(Context, self).opt_map_col_wind_idx_1
        self.opt_map_col_default    = super(Context, self).opt_map_col_default

        # Enable/disable generation of delta maps.
        self.opt_map_delta = False

        # Tells whether discrete color scale are used maps (rather than a continuous scale).
        self.opt_map_discrete = True

        # Format of maps.
        self.opt_map_format = ["png", "csv"]

        # Map locations (pd.DataFrame).
        self.opt_map_locations = None

        # Map resolution.
        self.opt_map_resolution = 0.05

        # Spatial reference (starts with: EPSG).
        self.opt_map_spat_ref = ""

        # Enable/diasble generation of Taylor diagrams for [scenarios, indices].
        self.opt_taylor = [True] * 2

        # Format of Taylor diagrams.
        self.opt_taylor_format = ["png", "csv"]

        # Enable/disable the calculation of statistics tables for [scenarios, indices].
        self.opt_tbl = [True] * 2

        # Centiles required to generate a statistical tables.
        self.opt_tbl_centiles = [0, 1, 10, 50, 90, 99, 100]

        # Enable/diasble generation of time series for [scenarios, indices].
        self.opt_ts = [True] * 2

        # Enable/disable generation of bias plots for [scenarios].
        self.opt_ts_bias = True

        # Centiles for which a time series is required.
        self.opt_ts_centiles = [10, 90]

        # Format of time series.
        self.opt_ts_format = [c.F_PNG, c.F_CSV]

        # Enabled/disable the comparison between reference data sources (at-a-station vs gridded).
        self.opt_valid_ref = False

        # Enabe/disable exporting dataset to CSV files for [scenarios, indices].
        self.export_data_to_csv = [False] * 2

        # Enable/disable exporting to dashboard.
        self.export_dashboard = False

        # Enable/disable clipping according to 'p_bounds' (for everything except maps).
        self.opt_stat_clip = False

        # Environment --------------------------

        # Number of processes
        self.n_proc = 1

        # Format of output data.
        self.f_data_out = c.F_NC

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
                    self._project = ast.literal_eval(value)

                # Reference data (at-a-station).
                elif key == "ens_ref_stn":
                    self.ens_ref_stn = ast.literal_eval(value)
                    if (self.ens_ref_stn != "") and (self.ens_ref_grid == ""):
                        self.ens_ref = self.ens_ref_stn
                elif key == "ens_ref_stns":
                    self.ens_ref_stns = cl_context.str_to_arr_1d(value, str)
                elif key == "ens_ref_col_sep":
                    self.ens_ref_col_sep = ast.literal_eval(value)

                # Reference data (gridded).
                elif key == "ens_ref_grid":
                    self.ens_ref_grid = ast.literal_eval(value)
                    if self.ens_ref_grid != "":
                        self.ens_ref = self.ens_ref_grid
                    if len(self.opt_download_per) == 0:
                        if self.ens_ref_grid == c.ENS_ERA5:
                            self.opt_download_per = [1979, 2020]
                        elif self.ens_ref_grid == c.ENS_ERA5_LAND:
                            self.opt_download_per = [1981, 2020]
                        elif self.ens_ref_grid == c.ENS_MERRA2:
                            self.opt_download_per = [1980, 2010]
                elif key == "ens_ref_grid_usr":
                    self.ens_ref_grid_usr = ast.literal_eval(value)
                elif key == "ens_ref_grid_pwd":
                    self.ens_ref_grid_usr = ast.literal_eval(value)

                # Context.
                elif key == "emission_scenarios":
                    self.emission_scenarios = ast.literal_eval(value)
                elif key == "per_ref":
                    self.per_ref = cl_context.str_to_arr_1d(value, int)
                    if per_hors_read:
                        self.per_hors = [self.per_ref] + self.per_hors
                    per_ref_read = True
                elif key == "per_fut":
                    self.per_fut = cl_context.str_to_arr_1d(value, int)
                elif key == "per_hors":
                    self.per_hors = cl_context.str_to_arr_2d(value, int)
                    if per_ref_read:
                        self.per_hors = [self.per_ref] + list(self.per_hors)
                    per_hors_read = True
                elif key == "lon_bnds":
                    self.lon_bnds = cl_context.str_to_arr_1d(value, float)
                elif key == "lat_bnds":
                    self.lat_bnds = cl_context.str_to_arr_1d(value, float)
                elif key == "ctrl_pt":
                    self.ctrl_pt = cl_context.str_to_arr_1d(value, float)
                elif key == "variables":
                    self.variables = cl_context.str_to_arr_1d(value, str)
                elif key == "p_bounds":
                    self.p_bounds = ast.literal_eval(value)
                elif key == "region":
                    self.region = ast.literal_eval(value)

                # Data:
                elif key == "opt_download":
                    self.opt_download = ast.literal_eval(value)
                elif key == "opt_download_lon_bnds":
                    self.opt_download_lon_bnds = cl_context.str_to_arr_1d(value, float)
                elif key == "opt_download_lat_bnds":
                    self.opt_download_lat_bnds = cl_context.str_to_arr_1d(value, float)
                elif key == "opt_download_per":
                    self.opt_download_per = cl_context.str_to_arr_1d(value, int)
                elif key == "opt_aggregate":
                    self.opt_aggregate = ast.literal_eval(value)

                # Climate scenarios.
                elif key == "opt_scen":
                    self.opt_scen = ast.literal_eval(value)
                elif key == "domains":
                    self.domains = cl_context.str_to_arr_1d(value, str)
                elif key == "radius":
                    self.radius = float(value)
                elif key == "sim_excepts":
                    self.sim_excepts = cl_context.str_to_arr_1d(value, str)
                elif key == "var_sim_excepts":
                    self.var_sim_excepts = cl_context.str_to_arr_1d(value, str)

                # Bias adjustment:
                elif key == "opt_bias":
                    self.opt_bias = ast.literal_eval(value)
                elif key == "opt_bias_meth":
                    self.opt_bias_meth = ast.literal_eval(value)
                elif key == "opt_bias_nq":
                    self.opt_bias_nq = int(value)
                elif key == "opt_bias_up_qmf":
                    self.opt_bias_up_qmf = float(value)
                elif key == "opt_bias_time_win":
                    self.opt_bias_time_win = int(value)
                elif key == "opt_bias_perturb":
                    self.opt_bias_perturb = cl_context.str_to_arr_2d(value, float)
                elif key == "opt_bias_err_meth":
                    self.opt_bias_err_meth = ast.literal_eval(value)

                # Climate indices.
                elif key == "opt_idx":
                    self.opt_idx = ast.literal_eval(value)
                elif key == "idx_codes":
                    self.idx_codes = cl_context.str_to_arr_1d(value, str)
                elif key == "idx_params":
                    idx_params_tmp = cl_context.str_to_arr_2d(value, float)
                    for i in range(len(self.idx_codes)):
                        if self.idx_codes[i] == c.I_R10MM:
                            self.idx_params.append([10])
                        elif self.idx_codes[i] == c.I_R20MM:
                            self.idx_params.append([20])
                        else:
                            self.idx_params.append(idx_params_tmp[i])

                # Results > Cluster:
                elif key == "opt_cluster":
                    self.opt_cluster = ast.literal_eval(value)
                elif key == "opt_cluster_centiles":
                    opt_cluster_centiles = cl_context.str_to_arr_1d(value, int)
                    if str(opt_cluster_centiles).replace("['']", "") == "":
                        self.opt_cluster_centiles = list(opt_cluster_centiles)
                        self.opt_cluster_centiles.sort()
                elif key == "opt_cluster_col":
                    self.opt_cluster_col = ast.literal_eval(value)
                elif key == "opt_cluster_format":
                    self.opt_cluster_format = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_cluster_variables":
                    self.opt_cluster_variables = cl_context.str_to_arr_1d(value, str)

                # Results > Cycle:
                elif key == "opt_cycle":
                    self.opt_cycle = ast.literal_eval(value)
                elif key == "opt_cycle_format":
                    self.opt_cycle_format = cl_context.str_to_arr_1d(value, str)

                # Results > Diagnostic:
                elif key == "opt_diagnostic":
                    self.opt_diagnostic = ast.literal_eval(value)
                elif key == "opt_diagnostic_format":
                    self.opt_diagnostic_format = cl_context.str_to_arr_1d(value, str)

                # Results > Map:
                elif key == "opt_map":
                    self.opt_map = [False, False]
                    if self.ens_ref_grid != "":
                        self.opt_map = ast.literal_eval(value)\
                            if ("," not in value) else cl_context.str_to_arr_1d(value, bool)
                elif key == "opt_map_centiles":
                    opt_map_centiles = cl_context.str_to_arr_1d(value, int)
                    if str(opt_map_centiles).replace("['']", "") == "":
                        self.opt_map_centiles = list(opt_map_centiles)
                        self.opt_map_centiles.sort()
                elif key == "opt_map_clip":
                    self.opt_map_clip = ast.literal_eval(value)
                elif key == "opt_map_col_temp_var":
                    self.opt_map_col_temp_var = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_temp_idx_1":
                    self.opt_map_col_temp_idx_1 = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_temp_idx_2":
                    self.opt_map_col_temp_idx_2 = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_var":
                    self.opt_map_col_prec_var = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_idx_1":
                    self.opt_map_col_prec_idx_1 = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_idx_2":
                    self.opt_map_col_prec_idx_2 = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_prec_idx_3":
                    self.opt_map_col_prec_idx_3 = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_wind_var":
                    self.opt_map_col_wind_var = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_wind_idx_1":
                    self.opt_map_col_wind_idx_1 = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_col_default":
                    self.opt_map_col_default = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_delta":
                    self.opt_map_delta = ast.literal_eval(value)
                elif key == "opt_map_discrete":
                    self.opt_map_discrete = ast.literal_eval(value)
                elif key == "opt_map_format":
                    self.opt_map_format = cl_context.str_to_arr_1d(value, str)
                elif key == "opt_map_locations":
                    self.opt_map_locations =\
                        pd.DataFrame(cl_context.str_to_arr_2d(value, float), columns=["longitude", "latitude", "desc"])
                elif key == "opt_map_resolution":
                    self.opt_map_resolution = ast.literal_eval(value)
                elif key == "opt_map_spat_ref":
                    self.opt_map_spat_ref = ast.literal_eval(value)

                # Results > Taylor diagram:
                elif key == "opt_taylor":
                    self.opt_taylor = ast.literal_eval(value)\
                        if ("," not in value) else cl_context.str_to_arr_1d(value, bool)
                elif key == "opt_taylor_format":
                    self.opt_taylor_format = cl_context.str_to_arr_1d(value, str)

                # Results > Statistics table:
                elif key == "opt_tbl":
                    self.opt_tbl = ast.literal_eval(value)\
                        if ("," not in value) else cl_context.str_to_arr_1d(value, bool)
                elif key == "opt_tbl_centiles":
                    opt_tbl_centiles = cl_context.str_to_arr_1d(value, int)
                    if str(opt_tbl_centiles).replace("['']", "") != "":
                        self.opt_tbl_centiles = list(opt_tbl_centiles)
                        self.opt_tbl_centiles.sort()

                # Results > Time series:
                elif key == "opt_ts":
                    self.opt_ts = ast.literal_eval(value)\
                        if ("," not in value) else cl_context.str_to_arr_1d(value, bool)
                elif key == "opt_ts_bias":
                    self.opt_ts_bias = ast.literal_eval(value)
                elif key == "opt_ts_centiles":
                    opt_ts_centiles = cl_context.str_to_arr_1d(value, int)
                    if str(opt_ts_centiles).replace("['']", "") == "":
                        self.opt_ts_centiles = list(opt_ts_centiles)
                        self.opt_ts_centiles.sort()
                elif key == "opt_ts_format":
                    self.opt_ts_format = cl_context.str_to_arr_1d(value, str)

                # Results > Validation of reference data.
                elif key == "opt_valid_ref":
                    self.opt_valid_ref = ast.literal_eval(value)

                # Results > General.
                elif key == "opt_stat_clip":
                    self.opt_stat_clip = ast.literal_eval(value)

                # Results > Export data to CSV files.
                elif key == "export_data_to_csv":
                    self.export_data_to_csv = ast.literal_eval(value)\
                        if ("," not in value) else cl_context.str_to_arr_1d(value, bool)

                # Results > Export to dashboard.
                elif key == "export_dashboard":
                    self.export_dashboard = ast.literal_eval(value)

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
                elif key == "d_ref_hour":
                    self.d_ref_hour = ast.literal_eval(value)
                elif key == "d_ref_day":
                    self.d_ref_day = ast.literal_eval(value)
                elif key == "opt_trace":
                    self.opt_trace = ast.literal_eval(value)
                elif key == "opt_force_overwrite":
                    self.opt_force_overwrite = ast.literal_eval(value)
                elif key == "f_data_out":
                    self.f_data_out = ast.literal_eval(value)

                # Unit tests.
                elif key == "opt_test":
                    self.opt_test = ast.literal_eval(value)

        # Time step.
        self.priority_timestep = ["day"] * len(self.variables)

        # File holding geographic boundaries.
        if self.p_bounds != "":
            self.p_bounds = self.d_exec + self.country + self.sep + self._project + self.sep + "gis" + self.sep +\
                            self.p_bounds

        # Log file.
        dt = datetime.datetime.now()
        dt_str = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + "_" + \
            str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + str(dt.second).zfill(2)
        self.p_log = self.d_res + "log" + self.sep + dt_str + ".log"

        # Bias adjustment file name.
        self.opt_bias_fn = self.d_res + self.opt_bias_fn

    def d_ref(
        self,
        var_name: Optional[str] = "",
        ens: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get directory of reference data.

        /exec/<user_name>/<country>/<project>/stn/<ens_ref_stn|ens_ref_grid>/<var>/*.csv

        Parameters
        ----------
        var_name: Optional[str]
            Variable name.
        ens: Optional[str]
            Ensemble.
            If this agrument is not specified, the directory holding gridded reference data is returned.
            If it's not available, the directory holding at-a-station reference data is returne.

        Returns
        -------
        str
            Directory of station data.
        ----------------------------------------
        """

        # Region.
        if ((ens == "") and (self.ens_ref_grid != "")) or\
           ((ens != "") and (data_type(ens) == c.DATA_TYPE_GRID)):
            region = self.ens_ref_grid + ("_" + self.region if self.region != "" else "")
        else:
            region = self.ens_ref_stn

        d = self.d_exec + self.country + self.sep + self._project + self.sep + c.CAT_STN + self.sep +\
            region + self.sep

        if var_name != "":
            d = d + var_name + self.sep

        return d

    def p_ref(
        self,
        var_name: str,
        stn: str
    ) -> str:

        """
        ----------------------------------------
        Get the path of reference data.

        Parameters
        ----------
        var_name: str
            Variable name.
        stn: str
            Station.

        Returns
        -------
        str
            Path of reference data.
        ----------------------------------------
        """

        p = str(self.d_ref(var_name)) + var_name + "_" + stn + "." + self.f_data_out

        return p

    def d_ref_raw(
        self,
        freq: Optional[str],
        var_name: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get the path of the directory holding reference data (original).

        Parameters
        ----------
        freq: Optional[str]
            Frequency = {"c.FREQ_H", "c.FREQ_D"}
        var_name: str
            Climate variable name.

        Returns
        -------
        str
            Path of the directory holding reference data (raw).
        ----------------------------------------
        """

        p = (self.d_ref_hour if freq == c.FREQ_H else self.d_ref_day)
        if var_name != "":
            p += var_name + self.sep

        return p

    def p_ref_raw(
        self,
        var_name: str,
        freq: str,
        year: int,
        stat_name: Optional[str] = ""
    ):

        """
        --------------------------------------------------------------------------------------------------------------------
        Get the path of ECMWF reanalysis data file.

        Parameters
        ----------
        var_name: str
            Variable name.
        freq: str
            Frequency = {"c.FREQ_H", "c.FREQ_D"}
        year: int
            Year.
        stat_name: Optional[str]
            Statistics name = {c.STAT_MEAN, c.STAT_MIN, c.STAT_MAX, c.STAT_SUM}
        --------------------------------------------------------------------------------------------------------------------
        """

        return (self.d_ref_hour if freq == c.FREQ_H else self.d_ref_day) + var_name + stat_name + cntx.sep + \
            var_name + stat_name + "_" + self.ens_ref_grid + "_" + ("day" if freq == c.FREQ_D else "hour") + "_" + \
            str(year) + "." + self.f_data_out

    def p_proj(
        self,
        var_name: Optional[str] = "*",
        domain: Optional[str] = "*",
        freq: Optional[str] = "day",
        rcp_code: Optional[str] = "*"
    ) -> str:

        """
        ----------------------------------------
        Get path of projection files (pattern).

        Parameters
        ----------
        var_name: Optional[str]
            Climate variable name.
        domain: Optional[str]
            Domain.
        freq: Optional[str]
            Frequency = {"day", "mon", "fx"}.
        rcp_code: Optional[str]
            Emission scenario.

        Returns
        -------
        str
            Path of projection files (pattern).
        ----------------------------------------
        """

        return self.d_proj + "*" + self.sep + "*" + self.sep + domain + "_*" + rcp_code + self.sep +\
            freq + self.sep + "atmos" + self.sep + "*" + self.sep + var_name + self.sep + "*"

    @property
    def d_res(
        self
    ) -> str:

        """
        ----------------------------------------
        Get path of directory holding results.

        Returns
        -------
        str
            Path of directory holding results.
        ----------------------------------------
        """

        return self.d_exec + "sim_climat" + self.sep + self.country + self.sep + self._project + self.sep +\
            "stn" + cntx.sep + (self.ens_ref_grid if self.ens_ref_grid != "" else self.ens_ref_stn) +\
            ("_" + self.region if self.region != "" else "") + self.sep

    def d_scen(
        self,
        cat: str,
        var_name: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get the path of the directory holding climate scenarios.

        Parameters
        ----------
        cat: str
            Category.
        var_name: Optional[str]
            Climate variable.

        Returns
        -------
        str
            Directory of climate scenarios.
        ----------------------------------------
        """

        d = self.d_res + c.CAT_SCEN + self.sep
        if cat != "":
            d = d + cat + self.sep
        if var_name != "":
            d = d + var_name + self.sep

        return d

    def p_scen(
        self,
        cat: str,
        var_name: str,
        sim_code: Optional[str] = "*"
    ) -> str:

        """
        ----------------------------------------
        Get the path of a dataset holding climate scenarios.

        Parameters
        ----------
        cat: str
            Category.
        var_name: str
            Climate variable name.
        sim_code: Optional[str]
            Simulation code.

        Returns
        -------
        str
            Path of climate scenarios.
        ----------------------------------------
        """

        suffix = ""
        if cat in [c.CAT_REGRID_REF, c.CAT_REGRID_FUT]:
            suffix = "4" + c.CAT_QQMAP
            if cat == c.CAT_REGRID_REF:
                suffix = c.REF + "_" + suffix
            cat = c.CAT_REGRID

        p = str(self.d_scen(cat, var_name)) + var_name + "_" + sim_code

        if suffix != "":
            p = p + "_" + suffix
        p = p + "." + cntx.f_data_out

        return p

    def d_idx(
        self,
        idx_code: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get the path of the directory holding climate indices.

        Parameters
        ----------
        idx_code : Optional[str]
            Climate index code.

        Returns
        -------
        str
            Directory of climate indices.
        ----------------------------------------
        """

        d = self.d_res + c.CAT_IDX + self.sep
        if idx_code != "":
            d = d + idx_code + self.sep

        return d

    def p_idx(
        self,
        idx_code: str,
        idx_name: str,
        sim_code: Optional[str] = "*"
    ) -> str:

        """
        ----------------------------------------
        Get the path of a dataset holding climate indices.

        Parameters
        ----------
        idx_code: str
            Climate index code.
        idx_name: str
            Climate index name.
        sim_code: Optional[str]
            Simulation code.

        Returns
        -------
        str
            Directory of climate indices.
        ----------------------------------------
        """

        p = self.d_res + c.CAT_IDX + self.sep + idx_code + self.sep + idx_name + "_" + sim_code + "." + cntx.f_data_out

        return p

    def d_fig(
        self,
        view_code: Optional[str] = "",
        vi_name: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get the path of the directory holding figures and tables.

        Parameters
        ----------
        view_code: Optional[str]
            View code = {c.VIEW_BIAS, c.VIEW_CLUSTER, c.VIEW_CYCLE*, c.VIEW_MAP, c.VIEW_POSTPROCESS, c.VIEW_TAYLOR,
                         c.VIEW_TBL, c.VIEW_TS, c.VIEW_TS_BIAS}
        vi_name: Optional[str]
            Climate variable or index name.

        Returns
        -------
        str
            Directory of figure data.
        ----------------------------------------
        """

        d = self.d_res + c.CAT_FIG + self.sep
        if view_code != "":
            d = d + view_code + self.sep
        if vi_name != "":
            d = d + vi_name + self.sep

        return d

    def p_fig(
        self,
        view_code: str,
        vi_code: str,
        vi_name: str,
        f_type: str,
        sim_code: Optional[str] = "",
        per: Optional[List[str]] = None,
        suffix: Optional[str] = ""
    ) -> str:

        """
        ----------------------------------------
        Get path of a figure or table.

        Parameters
        ----------
        view_code: str
            View code = {c.VIEW_BIAS, c.VIEW_CLUSTER, c.VIEW_CYCLE*, c.VIEW_MAP, c.VIEW_POSTPROCESS, c.VIEW_TAYLOR,
                         c.VIEW_TBL, c.VIEW_TS, c.VIEW_TS_BIAS}
        vi_code: str
            Climate variable or index code.
        vi_name: str
            Climate variable or index name.
        f_type: str
            File type = {c.F_CSV, c.F_PNG, c.F_HTML, c.F_TIF}
        sim_code: Optional[str]
            Simulation code.
        per: Optional[List[int]]
            Period (ex: [1981, 2010])
        suffix: Optional[str]
            Suffix.

        Returns
        -------
        str
            Path of figure.
        ----------------------------------------
        """

        p = ""

        # Directory suffix (if data).
        d_suffix = "_" + f_type if f_type != c.F_PNG else ""

        # Period.
        per_str = ""
        if per is not None:
            per_str = str(per[0])
            if len(per) > 1:
                per_str += "-" + str(per[1])

        # View-specific portion of the path.
        if view_code in [c.VIEW_TAYLOR, c.VIEW_TBL, c.VIEW_TS, c.VIEW_TS_BIAS]:
            p = self.d_fig(view_code, vi_code + d_suffix) + vi_name
        elif view_code in [c.VIEW_BIAS, c.VIEW_POSTPROCESS, c.VIEW_WORKFLOW]:
            p = self.d_fig(view_code, vi_code + d_suffix) + vi_name + "_"  + sim_code + "_" + view_code
        elif view_code in [c.VIEW_CYCLE_D, c.VIEW_CYCLE_MS]:
            p = self.d_fig(view_code, vi_code + d_suffix) +\
                per_str + cntx.sep + vi_name + "_" + sim_code + "_" + per_str.replace("-", "_") + "_" + view_code
        elif view_code == c.VIEW_CLUSTER:
            p = self.d_fig(view_code, vi_code + d_suffix) + vi_name
        elif view_code == c.VIEW_MAP:
            p = self.d_fig(view_code, vi_code + d_suffix) +\
                per_str + cntx.sep + vi_name + "_" + sim_code + "_" + per_str.replace("-", "_")

        # Add suffix.
        if suffix != "":
            p = p + suffix

        # Add file extension.
        p = p + "." + f_type

        return p

    def list_cordex(
        self,
        var_name: str,
        rcp_code: Optional[str] = "*",
        sim_code: Optional[str] = "*"
    ) -> Tuple[List[str], List[str]]:

        """
        --------------------------------------------------------------------------------------------------------------------
        List paths of directories holding simulations.

        Parameters
        ----------
        var_name: str
            Climate variable name.
        rcp_code: Optional[str]
            Emission scenario.
        sim_code: Optional[str]
            Climate simulation.

        Returns
        -------
        Tuple[List[str]]
            List of paths.
        --------------------------------------------------------------------------------------------------------------------
        """

        # Frequency.
        freq = "fx" if var_name == c.V_SFTLF else "day"

        # Emission scenarios.
        if sim_code != "*":
            tokens = sim_code.split("_")
            rcp_code_l = [tokens[len(tokens) - 1]]
        elif rcp_code != "*":
            rcp_code_l = [rcp_code]
        else:
            rcp_code_l = self.rcps.code_l

        # List all simulations.
        cordex = {}
        for rcp_code in rcp_code_l:

            # List simulation files.
            p_sim_l = []
            for domain in cntx.domains:
                p_sim_l = p_sim_l +\
                    list(glob.glob(self.p_proj(var_name, domain, freq, rcp_code) + c.F_EXT_NC)) +\
                    list(glob.glob(self.p_proj(var_name, domain, freq, rcp_code) + c.F_EXT_ZARR))

            # Remove timestep information.
            for i in range(0, len(p_sim_l)):
                tokens = p_sim_l[i].split(self.sep)
                p_sim_l[i] = "".join([token + "/" for token in tokens[0:len(tokens) - 5]]).rstrip(cntx.sep)

            # Keep only the unique simulation folders (with a * as the timestep).
            d_l = list(set(p_sim_l))
            d_l_valid = [True] * len(p_sim_l)

            # Keep only the simulations with all the variables we need.
            d_l = list(compress(d_l, d_l_valid))

            cordex[rcp_code + "_historical"] = [w.replace(rcp_code, "historical") for w in d_l]
            cordex[rcp_code] = d_l

        p_ref_l, p_rcp_l = [], []

        # Simulation specified.
        if sim_code != "":
            for rcp_code in rcp_code_l:
                p_ref_l = cordex[rcp_code + "_historical"]
                p_rcp_l = cordex[rcp_code]
                for i_sim in range(len(p_ref_l)):
                    if sim_code.replace("_", self.sep) in p_rcp_l[i_sim].replace("_", self.sep).replace("_", self.sep):
                        p_ref_l = [p_ref_l[i_sim]]
                        p_rcp_l = [p_rcp_l[i_sim]]
                        break

        # RCP specified.
        elif (sim_code == "") and (rcp_code != ""):
            p_ref_l = cordex[rcp_code + "_historical"]
            p_rcp_l = cordex[rcp_code]

        # Neither the simulation or RCP is specified.
        else:
            for rcp_code in self.emission_scenarios:
                p_ref_l = p_ref_l + cordex[rcp_code + "_historical"]
                p_rcp_l = p_rcp_l + cordex[rcp_code]

        return p_ref_l, p_rcp_l

    def exclude_simulation(
        self,
        var_name: str,
        sim_code: str,
        cat: Optional[str] = ""
    ) -> bool:

        """
        ----------------------------------------
        Tells whether a simulation needs to be excluded or not, based on:
        - exception lists
        - the availability of dependencies (CORDEX files).

        Parameters
        ----------
        var_name: str
            Climate variable name.
        sim_code: str
            Simulation code.
        cat: Optional[str]
            Category = {c.CAT_SCEN, c.CAT_IDX}

        Returns
        -------
        bool
            If True, a simulation needs to be considered.
        ----------------------------------------
        """

        # Verify if simulation is in the exception list.
        is_sim_except = False
        for sim_except in self.sim_excepts:
            if sim_except in sim_code:
                is_sim_except = True
                break

        # Verify if variable-simulation is in the exception list.
        is_var_sim_except = False
        for var_sim_except in self.var_sim_excepts:
            if var_sim_except in var_name + "_" + sim_code:
                is_var_sim_except = True
                break

        # Verify if simulation (CORDEX) files exist.
        files_avail = True
        if cat == c.CAT_SCEN:

            # Directories containing simulations.
            sim_ref_l, sim_fut_l = self.list_cordex(var_name, sim_code=sim_code)
            # d_sim_ref = sim_ref_l[0] if len(sim_ref_l) > 0 else ""
            # d_sim_fut = sim_fut_l[0] if len(sim_fut_l) > 0 else ""

            # Skip iteration if the variable 'var' is not available in the current directory.
            # p_sim_ref_base = d_sim_ref + "/*/*/*/".replace("/", cntx.sep) + var_name + cntx.sep + "*"
            # p_sim_ref_l =\
            #     list(glob.glob(p_sim_ref_base + c.F_EXT_NC)) +\
            #     list(glob.glob(p_sim_ref_base + c.F_EXT_ZARR))
            # p_sim_fut_base = d_sim_fut + "/*/*/*/".replace("/", cntx.sep) + var_name + cntx.sep + "*"
            # p_sim_fut_l =\
            #     list(glob.glob(p_sim_fut_base + c.F_EXT_NC)) +\
            #     list(glob.glob(p_sim_fut_base + c.F_EXT_ZARR))

            files_avail = (len(sim_ref_l) > 0) and (len(sim_fut_l) > 0)

        return is_sim_except or is_var_sim_except or not files_avail


def data_type(
    ens: str
) -> str:

    """
    ----------------------------------------
    Determine the type of a data ensemble.

    Parameters
    ----------
    ens: Optional[str]
        Ensemble.

    Returns
    -------
    str
        Type of data ensemble = {c.DATA_TYPE_STN, c.DATA_TYPE_GRID}
    ----------------------------------------
    """

    if ens in [c.ENS_ECMWF, c.ENS_ERA5, c.ENS_ERA5_LAND, c.ENS_ENACTS, c.ENS_MERRA2, c.ENS_CHIRPS]:
        return c.REF_TYPE_GRID
    else:
        return c.REF_TYPE_STN


# Configuration instance.
cntx = Context("script")
