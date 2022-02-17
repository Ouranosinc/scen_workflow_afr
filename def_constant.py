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
sys.path.append("dashboard")
from dashboard import def_constant as const


class Constant(const.Constant):

    # Data categories (in terms of the order of scenario production).
    cat_stn             = "stn"             # At-a-station or reanalysis data.
    cat_obs             = "obs"             # Observation.
    cat_raw             = "raw"             # Raw.
    cat_regrid          = "regrid"          # Reggrided.
    cat_qmf             = "qmf"             # Quantile mapping function.
    cat_qqmap           = "qqmap"           # Adjusted simulation.

    # Data categories (in terms of what they represent).
    cat_sim             = "sim"             # Simulation (not bias-adjusted).
    cat_sim_ref         = "sim_ref"         # Simulation (not bias-adjusted, reference period).
    cat_sim_adj         = "sim_adj"         # Simulation (bias-adjusted).
    cat_sim_adj_ref     = "sim_adj_ref"     # Simulation (bias-adjusted, reference period).

    # Scenarios vs. indices.
    cat_scen            = "scen"            # Scenarios.
    cat_idx             = "idx"             # Indices.

    # Categories of figures.
    cat_fig             = "fig"             # Figures.
    cat_fig_bias        = "bias"            # Figures (bias).
    cat_fig_postprocess = "postprocess"     # Figures (postprocess).
    cat_fig_workflow    = "workflow"        # Figures (workflow).

    # Conversion coefficients.
    spd                 = 86400             # Number of seconds per day.
    d_KC                = 273.15            # Temperature difference between Kelvin and Celcius.
    km_h_per_m_s        = 3.6               # Number of km/h per m/s.

    # Calendar types.
    cal_noleap          = "noleap"          # No-leap.
    cal_360day          = "360_day"         # 360 days.
    cal_365day          = "365_day"         # 365 days.

    # Date type.
    dtype_obj           = "object"
    dtype_64            = "datetime64[ns]"

    # Data frequency.
    freq_D              = "D"               # Daily.
    freq_MS             = "MS"              # Monthly.
    freq_YS             = "YS"              # Annual.

    # Scenarios.
    group               = "time.dayofyear"  # Grouping period.
    detrend_order       = None              # Detrend order (disabled).

    # Kind.
    kind_add            = "+"               # Additive.
    kind_mult           = "*"               # Multiplicative.

    # Bias adjustment error.
    opt_bias_err_meth_r2    = "r2"        # Coefficient of determination.
    opt_bias_err_meth_mae   = "mae"       # Mean absolute error.
    opt_bias_err_meth_rmse  = "rmse"      # Root mean square error.
    opt_bias_err_meth_rrmse = "rrmse"     # Relative root mean square error.

    # Dataset dimensions.
    dim_lon             = "lon"
    dim_lat             = "lat"
    dim_rlon            = "rlon"
    dim_rlat            = "rlat"
    dim_longitude       = "longitude"
    dim_latitude        = "latitude"
    dim_time            = "time"
    dim_location        = "location"
    dim_realization     = "realization"
    dim_percentiles     = "percentiles"

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

    # Log file.
    log_n_blank         = 10                # Number of blanks at the beginning of a message.
    log_sep_len         = 110               # Number of instances of the symbol "-" in a separator line.


const = Constant()
