# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script launcher.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import sys

# Workflow libraries.
import aggregate
import download
import file_utils as fu
import indices
import scenarios
import test
import utils

# Dashboard libraries.
from def_constant import const as c
from def_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import def_varidx as vi


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Step #0: Structure variables and indices.

    # Variables.
    if len(cntx.variables) > 0:
        cntx.vars = vi.VarIdxs(cntx.variables)

        # Reanalysis variables.
        if cntx.obs_src in [c.ens_era5, c.ens_era5_land, c.ens_enacts]:
            variables_ra = []
            for var in cntx.vars.items:
                variables_ra.append(var.convert_name(cntx.obs_src))
            cntx.vars_ra = vi.VarIdxs(variables_ra)

    if len(cntx.opt_download_variables) > 0:
        cntx.opt_download_vars = vi.VarIdxs(cntx.opt_download_variables)

    if len(cntx.opt_cluster_variables) > 0:
        cntx.cluster_vars = vi.VarIdxs(cntx.opt_cluster_variables)

    # Indices.
    cntx.idxs = vi.VarIdxs()
    for i in range(cntx.idxs.count):
        idx = vi.VarIdx(cntx.idx_codes[i])
        idx.params = cntx.idx_params[i]
        cntx.idxs.add(idx)

    # Step #1: Header.

    fu.log("=")
    fu.log("PRODUCTION OF CLIMATE SCENARIOS & CALCULATION OF CLIMATE INDICES                ")
    fu.log("Python Script created by Ouranos, based on xclim and xarray libraries.          ")
    fu.log("Script launched: " + utils.datetime_str())

    # Display configuration.
    fu.log("=")
    fu.log("Country                : " + cntx.country)
    fu.log("Project                : " + cntx.project)
    fu.log("Variables (CORDEX)     : " + str(cntx.vars.code_l).replace("'", ""))
    for i in range(cntx.idxs.count):
        params_i =\
            str(cntx.idxs.items.params[i]).replace("'", "").replace("(", "[").replace(")", "]").replace("\\n", "")
        fu.log("Climate index #" + ("0" if i < 9 else "") + str(i + 1) + "      : " + params_i)
    if cntx.opt_ra:
        fu.log("Reanalysis set         : " + cntx.obs_src)
        fu.log("Variables (reanalysis) : " + str(cntx.vars_ra.code_l).replace("'", ""))
    else:
        fu.log("Stations               : " + str(cntx.stns))
    fu.log("Emission scenarios     : " + str(cntx.rcps).replace("'", ""))
    fu.log("Reference period       : " + str(cntx.per_ref))
    fu.log("Future period          : " + str(cntx.per_fut))
    fu.log("Horizons               : " + str(cntx.per_hors).replace("'", ""))
    if (cntx.region != "") and cntx.opt_ra:
        fu.log("Region                 : " + cntx.region)

    # Step #2: Download and aggregation

    # Download data.
    fu.log("=")
    msg = "Step #2a  Downloading climate data"
    if cntx.opt_download:
        fu.log(msg)
        download.run()
    else:
        fu.log(msg + " (not required)")

    # Aggregate reanalysis data to daily frequency
    fu.log("=")
    msg = "Step #2b  Aggregating hourly data"
    if cntx.opt_aggregate and cntx.opt_ra:
        fu.log(msg)
        aggregate.run()
    else:
        fu.log(msg + " (not required)")

    # Steps #3-5: Data extraction, scenarios, bias adjustment and statistical downscaling.

    # Clean NetCDF files.
    if cntx.opt_scen:
        for var in cntx.vars.items:
            fu.clean_netcdf(cntx.d_stn + var.name + cntx.sep)
            fu.clean_netcdf(cntx.d_scen(cntx.obs_src, c.cat_scen + cntx.sep + "*", var.name))
    if cntx.opt_idx:
        for idx in cntx.idxs.items:
            fu.clean_netcdf(cntx.d_idx(cntx.obs_src, idx.name))

    # Initialization.
    scenarios.init_calib_params()

    # Launch units tests.
    if cntx.opt_test:
        test.run()

    # Steps #2-5,8: Production of scenarios, plots and statistics.
    scenarios.run()

    # Steps #6,8: Calculation of indices, plots and statistics.
    indices.run()

    fu.log("=")
    fu.log("Script completed: " + utils.datetime_str())


if __name__ == "__main__":
    main()
