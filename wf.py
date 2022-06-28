# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Climate scenarization tool.
#
# Script launcher.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020-2022 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# External libraries.
import sys

# Workflow libraries.
import wf_aggregate
import wf_download
import wf_file_utils as fu
import wf_indices
import wf_scenarios
import wf_test
import wf_utils

# Dashboard libraries.
from cl_constant import const as c
from cl_context import cntx

# Dashboard libraries.
sys.path.append("dashboard")
from dashboard import cl_project
from dashboard.cl_rcp import RCPs
from dashboard.cl_varidx import VarIdx, VarIdxs


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Migrate project.
    # fu.migrate(c.PLATFORM_SCRIPT, "", 1.2, 1.4)

    # Step #0: Structure: project, variables and indices ---------------------------------------------------------------

    # Project.
    cntx.code = c.PLATFORM_SCRIPT
    cntx.project = cl_project.Project(str(cntx.project))
    cntx.project.p_bounds = cntx.p_bounds

    # Emission scenarios.
    cntx.rcps = RCPs(cntx.emission_scenarios)

    # CORDEX variables.
    if len(cntx.variables) > 0:

        # Remove variables that are not compatible with the reanalysis ensemble.
        if (c.V_SFTLF in cntx.variables) and (cntx.obs_src != c.ENS_ERA5):
            cntx.variables.remove(c.V_SFTLF)

        # Convert list of variables to instances.
        cntx.vars = VarIdxs(cntx.variables)

        # Reanalysis variables.
        if cntx.obs_src in [c.ENS_ERA5, c.ENS_ERA5_LAND, c.ENS_ENACTS, c.ENS_CHIRPS]:
            variables_ra = []
            for var in cntx.vars.items:
                variables_ra.append(var.convert_name(cntx.obs_src))
            cntx.vars_ra = VarIdxs(variables_ra)

    # CORDEX variables used for clustering.
    if len(cntx.opt_cluster_variables) > 0:
        cntx.cluster_vars = VarIdxs(cntx.opt_cluster_variables)

    # Indices.
    if len(cntx.idx_codes):
        cntx.idxs = VarIdxs()
        for i in range(len(cntx.idx_codes)):
            idx = VarIdx(cntx.idx_codes[i])
            idx.params = cntx.idx_params[i]
            cntx.idxs.add(idx)

    # Step #1: Header --------------------------------------------------------------------------------------------------

    fu.log("=")
    fu.log("PRODUCTION OF CLIMATE SCENARIOS & CALCULATION OF CLIMATE INDICES                ")
    fu.log("Python Script created by Ouranos, based on xclim and xarray libraries.          ")
    fu.log("Script launched: " + wf_utils.datetime_str())

    # Display configuration.
    fu.log("=")
    fu.log("Country                : " + cntx.country)
    fu.log("Project                : " + cntx.project.code)
    fu.log("Variables (CORDEX)     : " + str(cntx.vars.code_l).replace("'", ""))
    if cntx.idxs is not None:
        for i in range(cntx.idxs.count):
            params_i =\
                str(cntx.idxs.items[i].params).replace("'", "").replace("(", "[").replace(")", "]").replace("\\n", "")
            fu.log("Climate index #" + ("0" if i < 9 else "") + str(i + 1) + "      : " + params_i)
    if cntx.opt_ra:
        fu.log("Reanalysis set         : " + cntx.obs_src)
        fu.log("Variables (reanalysis) : " + str(cntx.vars_ra.code_l).replace("'", ""))
    else:
        fu.log("Stations               : " + str(cntx.stns))
    fu.log("Emission scenarios     : " + str(cntx.rcps.code_l).replace("'", ""))
    fu.log("Reference period       : " + str(cntx.per_ref))
    fu.log("Future period          : " + str(cntx.per_fut))
    fu.log("Horizons               : " + str(cntx.per_hors).replace("'", ""))
    if (cntx.region != "") and cntx.opt_ra:
        fu.log("Region                 : " + cntx.region)

    # Step #2: Download and aggregation --------------------------------------------------------------------------------

    # Download data.
    fu.log("=")
    msg = "Step #2a  Downloading climate data"
    if cntx.opt_download:
        fu.log(msg)
        wf_download.run()
    else:
        fu.log(msg + " (not required)")

    # Aggregate reanalysis data to daily frequency.
    fu.log("=")
    msg = "Step #2b  Aggregating hourly data"
    if cntx.opt_aggregate and cntx.opt_ra:
        fu.log(msg)
        wf_aggregate.run()
    else:
        fu.log(msg + " (not required)")

    # Steps #3-5: Data extraction, scenarios, bias adjustment and statistical downscaling ------------------------------

    # Clean NetCDF files.
    if cntx.opt_scen:
        for var in cntx.vars.items:
            fu.clean_netcdf(cntx.d_stn(var.name))
            fu.clean_netcdf(cntx.d_scen("*", var.name))
    if cntx.opt_idx:
        for idx in cntx.idxs.items:
            fu.clean_netcdf(cntx.d_idx(idx.code))

    # Initialization.
    wf_scenarios.init_bias_params()

    # Launch units tests.
    if cntx.opt_test:
        wf_test.run()

    # Steps #2-5,8: Production of scenarios, plots and statistics ------------------------------------------------------
    wf_scenarios.run()

    # Steps #6,8: Calculation of indices, plots and statistics ---------------------------------------------------------
    wf_indices.run()

    # Step #9: Export dashboard ----------------------------------------------------------------------------------------

    fu.log("=")
    msg = "Step #9   Exporting dashboard"
    if cntx.export_dashboard:
        fu.log(msg)
        fu.deploy()
    else:
        fu.log(msg + " (not required)")

    fu.log("=")
    fu.log("Script completed: " + wf_utils.datetime_str())


if __name__ == "__main__":

    main()
