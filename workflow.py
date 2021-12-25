# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script launcher.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2021 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import aggregate
import download
import file_utils as fu
import indices
import scenarios
import unit_tests
import utils
from config import cfg

import sys
sys.path.append("dashboard")
from dashboard import def_varidx as vi


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Step #1: Header.

    fu.log("=")
    fu.log("PRODUCTION OF CLIMATE SCENARIOS & CALCULATION OF CLIMATE INDICES                ")
    fu.log("Python Script created by Ouranos, based on xclim and xarray libraries.          ")
    fu.log("Script launched: " + utils.get_datetime_str())

    # Display configuration.
    fu.log("=")
    fu.log("Country                : " + cfg.country)
    fu.log("Project                : " + cfg.project)
    fu.log("Variables (CORDEX)     : " + str(cfg.variables).replace("'", ""))
    n_i = len(cfg.idx_names)
    for i in range(n_i):
        params_i = str(cfg.idx_params[i]).replace("'", "").replace("(", "[").replace(")", "]").replace("\\n", "")
        fu.log("Climate index #" + ("0" if i < 9 else "") + str(i + 1) + "      : " + params_i)
    if cfg.opt_ra:
        fu.log("Reanalysis set         : " + cfg.obs_src)
        fu.log("Variables (reanalysis) : " + str(cfg.variables_ra).replace("'", ""))
    else:
        fu.log("Stations               : " + str(cfg.stns))
    fu.log("Emission scenarios     : " + str(cfg.rcps).replace("'", ""))
    fu.log("Reference period       : " + str(cfg.per_ref))
    fu.log("Future period          : " + str(cfg.per_fut))
    fu.log("Horizons               : " + str(cfg.per_hors).replace("'", ""))
    if (cfg.region != "") and cfg.opt_ra:
        fu.log("Region                 : " + cfg.region)

    # Step #2: Download and aggregation

    # Download data.
    fu.log("=")
    msg = "Step #2a  Downloading climate data"
    if cfg.opt_download:
        fu.log(msg)
        download.run()
    else:
        fu.log(msg + " (not required)")

    # Aggregate reanalysis data to daily frequency
    fu.log("=")
    msg = "Step #2b  Aggregating hourly data"
    if cfg.opt_aggregate and cfg.opt_ra:
        fu.log(msg)
        aggregate.run()
    else:
        fu.log(msg + " (not required)")

    # Steps #3-5: Data extraction, scenarios, bias adjustment and statistical downscaling.

    # Clean NetCDF files.
    if cfg.opt_scen:
        for vi_code in cfg.variables:
            fu.clean_netcdf(cfg.d_stn + vi_code + cfg.sep)
            fu.clean_netcdf(cfg.get_d_scen(cfg.obs_src, "scen" + cfg.sep + "*", vi_code))
    if cfg.opt_idx:
        for vi_code in cfg.idx_codes:
            fu.clean_netcdf(cfg.get_d_idx(cfg.obs_src, str(vi.VarIdx(vi_code).get_name())))

    # Initialization.
    scenarios.init_calib_params()

    # Launch units tests.
    if cfg.opt_unit_tests:
        unit_tests.run()

    # Steps #2-5,8: Production of scenarios, plots and statistics.
    scenarios.run()

    # Steps #6,8: Calculation of indices, plots and statistics.
    indices.run()

    fu.log("=")
    fu.log("Script completed: " + utils.get_datetime_str())


if __name__ == "__main__":
    main()
