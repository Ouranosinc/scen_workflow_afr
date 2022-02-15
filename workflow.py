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
from dashboard import def_project
from dashboard.def_rcp import RCPs
from dashboard.def_varidx import VarIdx, VarIdxs


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Rename files.
    # project_l = ["bf-co", "bf-hb"]
    # for i in range(len(project_l)):
    #     p = "/home/yrousseau/Documents/dev/scen_workflow_afr/dashboard/data/" + project_l[i] + "/"
    #     fu.renames_files(p, "q10", "c010", recursive=True)
    #     fu.renames_files(p, "q90", "c090", recursive=True)
    #     fu.renames_files(p, "drydays", "dry_days", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "heatwavemaxlen", "heat_wave_max_length", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "heatwavetotlen", "heat_wave_total_length", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "hotspellfreq", "hot_spell_frequency", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "hotspellmaxlen", "hot_spell_max_length", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "tropicalnights", "tropical_nights", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "txdaysabove", "tx_days_above", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "wetdays", "wet_days", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "wgdaysabove", "wg_days_above", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "wxdaysabove", "wx_days_above", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "rainstart", "rain_season_start", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "rainend", "rain_season_end", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "raindur", "rain_season_length", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "daily", "cycle_d", rename_directories=True, recursive=True)
    #     fu.renames_files(p, "monthly", "cycle_ms", rename_directories=True, recursive=True)

    # Step #0: Structure: project, variables and indices.

    # Project.
    cntx.code = c.platform_script
    cntx.project = def_project.Project(str(cntx.project))

    # Emission scenarios.
    cntx.rcps = RCPs(cntx.emission_scenarios)

    # CORDEX variables.
    if len(cntx.variables) > 0:
        cntx.vars = VarIdxs(cntx.variables)

        # Reanalysis variables.
        if cntx.obs_src in [c.ens_era5, c.ens_era5_land, c.ens_enacts]:
            variables_ra = []
            for var in cntx.vars.items:
                variables_ra.append(var.convert_name(cntx.obs_src))
            cntx.vars_ra = VarIdxs(variables_ra)

    # ERA5* variables to download.
    if len(cntx.opt_download_variables) > 0:
        cntx.opt_download_vars = VarIdxs(cntx.opt_download_variables)

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

    # Step #1: Header.

    fu.log("=")
    fu.log("PRODUCTION OF CLIMATE SCENARIOS & CALCULATION OF CLIMATE INDICES                ")
    fu.log("Python Script created by Ouranos, based on xclim and xarray libraries.          ")
    fu.log("Script launched: " + utils.datetime_str())

    # Display configuration.
    fu.log("=")
    fu.log("Country                : " + cntx.country)
    fu.log("Project                : " + cntx.project.code)
    fu.log("Variables (CORDEX)     : " + str(cntx.vars.code_l).replace("'", ""))
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
            fu.clean_netcdf(cntx.d_stn(var.name))
            fu.clean_netcdf(cntx.d_scen("*", var.name))
    if cntx.opt_idx:
        for idx in cntx.idxs.items:
            fu.clean_netcdf(cntx.d_idx(idx.code))

    # Initialization.
    scenarios.init_bias_params()

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
