# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Constants.
#
# Contact information:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import sys

# Dashboard libraries.
sys.path.append("scen_workflow_afr_dashboard")
from scen_workflow_afr_dashboard import cl_constant as const


class Constant(const.Constant):

    # Data categories (in terms of the order of scenario production).
    CAT_STN             = "stn"             # At-a-station or reanalysis data.
    CAT_OBS             = "obs"             # Observation.
    CAT_RAW             = "raw"             # Raw.
    CAT_REGRID          = "regrid"          # Reggrided.
    CAT_REGRID_REF      = "regrid_ref"      # Regridded (reference period only): "_ref_4qqmap"
    CAT_REGRID_FUT      = "regrid_fut"      # Regridded (all years included): "_4qqmap"
    CAT_QMF             = "qmf"             # Quantile mapping function.
    CAT_QQMAP           = "qqmap"           # Adjusted simulation.

    # Data categories (in terms of what they represent).
    CAT_SIM             = "sim"             # Simulation (not bias-adjusted).
    CAT_SIM_REF         = "sim_ref"         # Simulation (not bias-adjusted, reference period).
    CAT_SIM_ADJ         = "sim_adj"         # Simulation (bias-adjusted).
    CAT_SIM_ADJ_REF     = "sim_adj_ref"     # Simulation (bias-adjusted, reference period).

    # Main directories: scenarios, indices and figures.
    CAT_SCEN            = "scen"            # Scenarios.
    CAT_IDX             = "idx"             # Indices.
    CAT_FIG             = "fig"             # Figures.

    # Conversion coefficients.
    SPD                 = 86400             # Number of seconds per day.
    d_KC                = 273.15            # Temperature difference between Kelvin and Celcius.
    KM_H_PER_M_S        = 3.6               # Number of km/h per m/s.

    # Calendar types.
    CAL_NOLEAP          = "noleap"          # No-leap.
    CAL_360DAY          = "360_day"         # 360 days.
    CAL_365DAY          = "365_day"         # 365 days.

    # Date type.
    DTYPE_OBJ           = "object"
    DTYPE_64            = "datetime64[ns]"

    # Data frequency.
    FREQ_H              = "H"               # Hourly.
    FREQ_D              = "D"               # Daily.
    FREQ_MS             = "MS"              # Monthly.
    FREQ_YS             = "YS"              # Annual.

    # Scenarios.
    GROUP               = "time.dayofyear"  # Grouping period.
    DETREND_ORDER       = None              # Detrend order (disabled).

    # Kind.
    KIND_ADD            = "+"               # Additive.
    KIND_MULT           = "*"               # Multiplicative.

    # Bias adjustment error.
    OPT_BIAS_ERR_METH_R2    = "r2"        # Coefficient of determination.
    OPT_BIAS_ERR_METH_MAE   = "mae"       # Mean absolute error.
    OPT_BIAS_ERR_METH_RMSE  = "rmse"      # Root mean square error.
    OPT_BIAS_ERR_METH_RRMSE = "rrmse"     # Relative root mean square error.

    # Dataset dimensions.
    DIM_LON             = "lon"
    DIM_LAT             = "lat"
    DIM_RLON            = "rlon"
    DIM_RLAT            = "rlat"
    DIM_X               = "x"
    DIM_Y               = "y"
    DIM_TIME            = "time"
    DIM_LOCATION        = "location"
    DIM_STATION         = "station"
    DIM_REALIZATION     = "realization"
    DIM_PERCENTILES     = "percentiles"

    # Data array attributes.
    ATTRS_UNITS         = "units"
    ATTRS_SNAME         = "standard_name"
    ATTRS_LNAME         = "long_name"
    ATTRS_AXIS          = "axis"
    ATTRS_GMAP          = "grid_mapping"
    ATTRS_GMAPNAME      = "grid_mapping_name"
    ATTRS_BIAS          = "bias_corrected"
    ATTRS_COMMENTS      = "comments"
    ATTRS_STATION_NAME  = "station_name"
    ATTRS_GROUP         = "group"
    ATTRS_KIND          = "kind"
    ATTRS_ROT_LAT_LON   = "rotated_latitude_longitude"
    ATTRS_G_NP_LON       = "grid_north_pole_longitude"
    ATTRS_G_NP_LAT       = "grid_north_pole_latitude"

    # Units (general).
    UNIT_DEG            = "°"
    UNIT_PCT            = "%"

    # Units.
    UNIT_C              = "C"
    UNIT_K              = "K"
    UNIT_kg_m2s1        = "kg m-2 s-1"
    UNIT_m              = "m"
    UNIT_mm             = "mm"
    UNIT_mm_d           = "mm d-1"
    UNIT_m_s            = "m s-1"
    UNIT_J_m2           = "J m-2"
    UNIT_Pa             = "Pa"
    UNIT_d              = "d"
    UNIT_km_h           = "km h-1"
    UNIT_1              = "1"

    # Units (description; for plots).
    UNIT_C_DESC         = UNIT_DEG + UNIT_C
    UNIT_K_DESC         = UNIT_K
    UNIT_kg_m2s1_DESC   = "kg/m²s)"
    UNIT_m_DESC         = UNIT_m
    UNIT_mm_DESC        = UNIT_mm
    UNIT_mm_d_DESC      = "mm/jour"
    UNIT_m_s_DESC       = "m/s"
    UNIT_J_m2_DESC      = "J/m²"
    UNIT_Pa_DESC        = UNIT_Pa
    UNIT_d_DESC         = "jours"
    UNIT_km_h_DESC      = "km/h"

    # Log file.
    LOG_N_BLANK         = 10                # Number of blanks at the beginning of a message.
    LOG_SEP_LEN         = 110               # Number of instances of the symbol "-" in a separator line.

    # Color associated with specific datasets (bias plots).
    COL_OBS             = "green"           # Observed data.
    COL_SIM             = "purple"          # Simulation data (non-adjusted).
    COL_SIM_REF         = "orange"          # Simulation data (non-adjusted, reference period).
    COL_SIM_ADJ         = "red"             # Simulation data (bias-adjusted).
    COL_SIM_ADJ_REF     = "blue"            # Simulation data (bias-adjusted, reference period).


const = Constant()
