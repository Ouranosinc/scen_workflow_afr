# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script launcher.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import aggregate
import config as cfg
import datetime
import download
import indices
import os
import scenarios
import scenarios_calib as scen_calib
# import scenarios_verif as scen_verif
import statistics as stat
import utils


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Step #1: Parameters ----------------------------------------------------------------------------------------------

    # ==========================================================
    # TODO.CUSTOMIZATION.BEGIN
    # Update project parameters in the following block code.
    # ==========================================================

    # Country.
    cfg.country = "burkina"
    # cfg.country = "coteivoire"

    # Project-dependent parameters.
    if cfg.country == "burkina":
        # Project name.
        cfg.project = "pcci"
        # Observations (from measurements).
        cfg.obs_src = "anam"
        cfg.stns = ["bereba", "boromo", "boura", "diebougou", "farakoba", "gao", "gaoua", "hounde", "kassoum",
                    "koumbia", "leo", "nasso", "po", "sapouy", "valleedukou"]
        # RCPs, and reference and future periods.
        cfg.rcps    = [cfg.rcp_26, cfg.rcp_45, cfg.rcp_85]
        cfg.per_ref = [1988, 2017]
        cfg.per_fut = [cfg.per_ref[0], 2095]
        # Variables.
        cfg.variables_cordex = [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax, cfg.var_cordex_pr,
                                cfg.var_cordex_uas, cfg.var_cordex_vas]
        # Boundaries.
        # https://datacatalog.worldbank.org/dataset/burkina-faso-administrative-boundaries-2017
        cfg.d_bounds          = "bf_boundaries.geojson"
        cfg.lon_bnds          = [-6, 3]
        cfg.lat_bnds          = [8, 16]
        # Steps 3-4 - Data extraction and scenarios.
        cfg.opt_scen          = True        # True by default.
        cfg.opt_scen_regrid   = False
        # Step #5 - Bias adjustment and statistical downscaling.
        cfg.opt_calib         = True        # True by default.
        cfg.opt_calib_auto    = False
        # Step 6 - Index options.
        cfg.opt_idx           = True        # True by default.
        cfg.idx_names         = [cfg.idx_tx_days_above]
        cfg.idx_threshs       = [[36.0]]
        # Step 7 - Calculate statistics.
        cfg.opt_stat          = True        # True by default.
        # Step 8 - Visualization.
        cfg.opt_plot          = True        # True by default.
        # Files.
        cfg.file_sep          = ";"

    elif cfg.country == "coteivoire":
        # Project name.
        cfg.project = "adaptcoop"
        # Observations (from reanalysis).
        cfg.obs_src = cfg.obs_src_era5
        # RCPs, and reference and future periods.
        cfg.rcps    = [cfg.rcp_45, cfg.rcp_85]
        cfg.per_ref = [1981, 2010]
        cfg.per_fut = [cfg.per_ref[0], 2095]
        # Variables.
        cfg.variables_cordex = [cfg.var_cordex_tas, cfg.var_cordex_tasmin, cfg.var_cordex_tasmax, cfg.var_cordex_pr,
                                cfg.var_cordex_uas, cfg.var_cordex_vas]
        cfg.variables_ra = [cfg.var_era5_d2m, cfg.var_era5_e, cfg.var_era5_pev, cfg.var_era5_sp, cfg.var_era5_ssrd,
                            cfg.var_era5_t2m, cfg.var_era5_tp, cfg.var_era5_u10, cfg.var_era5_v10]
        # Boundaries.
        # https://datacatalog.worldbank.org/dataset/cote-divoire-administrative-boundaries-2016
        cfg.d_bounds          = "ci_boundaries.geojson"
        cfg.lon_bnds          = [-9, -2]
        cfg.lat_bnds          = [4, 11]
        # Step 2 - Download and aggregation options.
        cfg.opt_download      = False
        cfg.lon_bnds_download = [-29, 65]
        cfg.lat_bnds_download = [-50, 47]
        cfg.opt_aggregate     = False
        # Steps 3-4 - Data extraction and scenarios.
        cfg.opt_scen          = True        # True by default.
        cfg.opt_scen_regrid   = False
        # Step #5 - Bias adjustment and statistical downscaling.
        cfg.opt_calib         = True        # True by default.
        cfg.opt_calib_auto    = False
        # Step 6 - Index options.
        cfg.opt_idx           = True        # True by default.
        cfg.idx_names         = [cfg.idx_tx_days_above]
        cfg.idx_threshs       = [[36.0]]
        # Step 7 - Calculate statistics.
        cfg.opt_stat          = True        # True by default.
        # Step 8 - Visualization.
        cfg.opt_plot          = True        # True by default.

    # ==========================================================
    # TODO.CUSTOMIZATION.END
    # ==========================================================

    # List of simulation and var-simulation combinations that must be avoided to avoid a crash.
    cfg.sim_excepts = ["RCA4_AFR-44_ICHEC-EC-EARTH_rcp85",
                       "RCA4_AFR-44_MPI-M-MPI-ESM-LR_rcp85",
                       "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp45.nc",
                       "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp85.nc"]
    cfg.var_sim_excepts = [cfg.var_cordex_pr + "_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85.nc",
                           cfg.var_cordex_tasmin + "_REMO2009_AFR-44_MIROC-MIROC5_rcp26.nc"]

    # Variables.
    cfg.priority_timestep = ["day"] * len(cfg.variables_cordex)

    # System.
    # Two base directories are used in the configuration below owing to limitations in the capacity of USB drives.
    # Ideally, all the datasets should go under a single hierarchy. Alternatively, input files could be read from one
    # location and saved to a second location to prevent overwriting input files by accident.
    # Location #1: CORDEX-AFR (daily frequency) (input)
    #              ERA5-Land (hourly frequency) (input)
    # Location #2: ERA5-Land (daily frequency) (output and input)
    #              ERA5 (hourly frequency) (input)
    #              ERA5 (daily frequency) (output and input)
    #              Geographic files (input)
    #              Gridded version of observations (output)
    #              Climate scenarios (output)
    # The following block of lines (in particular the 4 next lines) can be modified to fit the configuration of the
    # system on which the script is run. If all input and output files are located under the same directory, modify
    # the code to have: cfg.d_base_in2 = cfg.d_base_in1

    # ==========================================================
    # TODO.CUSTOMIZATION.BEGIN
    # Update system parameters in the following block code.
    # ==========================================================

    # The following variables are specific to the system on which the script is run.
    cfg.d_username = "yrousseau"
    cfg.d_base_in1 = "/media/" + cfg.d_username + "/ROCKET-XTRM/"
    cfg.d_base_in2 = "/media/" + cfg.d_username + "/wd/"
    cfg.d_base_exec = "/exec/" + cfg.d_username + "/"

    # ==========================================================
    # TODO.CUSTOMIZATION.END
    # ==========================================================

    # The following variables are determined automatically.
    if (cfg.obs_src == cfg.obs_src_era5) or (cfg.obs_src == cfg.obs_src_era5_land):
        d_suffix_raw = "scenario/external_data/ecmwf/" + cfg.obs_src + "/hour/"
        if cfg.obs_src == cfg.obs_src_era5_land:
            cfg.d_ra_raw = cfg.d_base_in1 + d_suffix_raw
            cfg.d_ra_day = (cfg.d_base_in2 + d_suffix_raw).replace("hour", "day")
        else:
            cfg.d_ra_raw = cfg.d_base_in2 + d_suffix_raw
            cfg.d_ra_day = cfg.d_ra_raw.replace("hour", "day")
    elif cfg.obs_src == cfg.obs_src_merra2:
        d_suffix_raw = "scenario/external_data/nasa/merra2/raw/"
        cfg.d_ra_raw = cfg.d_base_in2 + d_suffix_raw
    else:
        cfg.d_stn = cfg.d_base_exec + cfg.country + "/" + cfg.project + "/" + cfg.cat_obs + "/" + cfg.obs_src + "/"
    cfg.d_cordex = cfg.d_base_in1 + "scenario/external_data/CORDEX-AFR/"
    if cfg.d_bounds != "":
        cfg.d_bounds = cfg.d_base_in2 + "scenario/external_data/gis/" + cfg.d_bounds
    cfg.d_sim = cfg.d_base_exec + "sim_climat/" + cfg.country + "/" + cfg.project + "/"

    # Log file.
    dt = datetime.datetime.now()
    cfg.p_log = cfg.d_sim + "log/" + str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + "_" +\
        str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + str(dt.second).zfill(2) + ".log"

    # Calibration file.
    cfg.p_calib = cfg.d_sim + cfg.p_calib
    d = os.path.dirname(cfg.p_calib)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # ------------------------------------------------------------------------------------------------------------------

    utils.log("=")
    utils.log("PRODUCTION OF CLIMATE SCENARIOS & CALCULATION OF CLIMATE INDICES                ")
    utils.log("Python Script created by Ouranos, based on xclim and xarray libraries.          ")

    # Display configuration.
    utils.log("=")
    utils.log("Country            : " + cfg.country)
    utils.log("Project            : " + cfg.project)
    utils.log("CORDEX variables   : " + str(cfg.variables_cordex))
    for i in range(len(cfg.idx_names)):
        utils.log("Climate index #" + str(i + 1) + "   : " + cfg.idx_names[i] + str(cfg.idx_threshs[i]))
    if (cfg.obs_src == cfg.obs_src_era5) or (cfg.obs_src == cfg.obs_src_era5_land):
        utils.log("Reanalysis         : " + cfg.obs_src)
    else:
        utils.log("Stations           : " + str(cfg.stns))
    utils.log("Emission scenarios : " + str(cfg.rcps))
    utils.log("Reference period   : " + str(cfg.per_ref))
    utils.log("Future period      : " + str(cfg.per_fut))
    utils.log("Horizons           : " + str(cfg.per_hors))

    # Step #2: Download and aggregation --------------------------------------------------------------------------------

    # Download data.
    utils.log("=")
    msg = "Step #2a  Downloading climate data is "
    if cfg.opt_download:
        msg = msg + "running"
        utils.log(msg)
        download.run()
    else:
        msg = msg + "not required"
        utils.log(msg)

    # Aggregate reanalysis data to daily frequency
    utils.log("=")
    msg = "Step #2b  Aggregation of hourly data is "
    if cfg.opt_aggregate and ((cfg.obs_src == cfg.obs_src_era5) or (cfg.obs_src == cfg.obs_src_era5_land)):
        msg = msg + "running"
        utils.log(msg)
        aggregate.run()
    else:
        msg = msg + "not required"
        utils.log(msg)

    # Steps #3-5: Data extraction, scenarios, bias adjustment and statistical downscaling ------------------------------

    # There are 3 calibration modes.
    # Note that a CSV file containing calibration parameters (corresponding to the variable 'p_calib') is generated when
    # using modes 2 and 3. This file will automatically be loaded the next time the script runs. In that situation,
    # calibration could be disabled if previous values are still valid.

    # Calibration mode #1: no calibration.
    # a. Set a value for each of the following variable:
    #    cfg.nq_default       = ...
    #    cfg.up_qmf_default   = ...
    #    cfg.time_win_default = ...
    # b. Set the following options:
    #    cfg.opt_calib      = False
    #    cfg.opt_calib_auto = False
    # c. Run the script.

    # Calibration mode #2: manual:
    # a. Run function 'scen_calib.init_calib_params()' so that it generates a calibration file.
    # b. Set a list of values to each of the following parameters:
    #    cfg.nq_calib       = [<value_i>, ..., <value_n>]
    #    cfg.up_qmf_calib   = [<value_i>, ..., <value_n>]
    #    cfg.time_win_calib = [<value_i>, ..., <value_n>]
    #    cfg.bias_err_calib = [<value_i>, ..., <value_n>]
    # c. Set the following options:
    #    cfg.opt_calib      = True
    #    cfg.opt_calib_auto = False
    #    cfg.opt_idx        = False
    # d. Run the script.
    #    This will adjust bias for each combination of three parameter values.
    # e. Examine the plots that were generated under the following directory:
    #    cfg.get_d_sim(<station_name>, "fig")/calib/
    #    and select the parameter values that produce the best fit between simulations and observations.
    #    There is one parameter value per simulation, per station, per variable.
    # f. Set values for the following parameters:
    #    cfg.nq_default       = <value in cfg.nq_calib producing the best fit>
    #    cfg.up_qmf_default   = <value in cfg.up_qmf_calib producing the best fit>
    #    cfg.time_win_default = <value in cfg.time_win_calib producing the best fit>
    #    cfg.bias_err_default = <value in cfg.bias_err_calib producing the best fit>
    # g. Set the following options:
    #    cfg.opt_calib      = False
    #    cfg.opt_calib_auto = False
    #    cfg.opt_idx        = True
    # h. Run the script again.

    # Calibration mode #3: automatic:
    # This mode will take much longer to run, but it will fit observations much better.
    # a. Set a list of values to each of the following parameters:
    #    cfg.nq_calib       = [<value_i>, ..., <value_n>]
    #    cfg.up_qmf_calib   = [<value_i>, ..., <value_n>]
    #    cfg.time_win_calib = [<value_i>, ..., <value_n>]
    #    cfg.bias_err_calib = [<value_i>, ..., <value_n>]
    # b. Set the following options:
    #    cfg.opt_calib      = True
    #    cfg.opt_calib_auto = True
    # c. Run the script.

    # Default values.
    cfg.nq_default       = 50
    cfg.up_qmf_default   = 3
    cfg.time_win_default = 30

    # List of parameters to be tested.
    cfg.nq_calib       = [cfg.nq_default]
    cfg.up_qmf_calib   = [cfg.up_qmf_default]
    cfg.time_win_calib = [cfg.time_win_default]

    # Initialization.
    scen_calib.init_calib_params()

    # Calculation of scenarios.
    scenarios.run()

    # DEBUG: This following statement is optional. It is useful to verify the generated NetCDF files.
    # DEBUG: scen_verif.run()

    # Step #6: Indices -------------------------------------------------------------------------------------------------

    # Calculation of indices.
    indices.run()

    # Step #7: Statistics ----------------------------------------------------------------------------------------------

    # Calculation of statistics.
    stat.run()

    utils.log("=")
    utils.log("Script completed successfully.")


if __name__ == "__main__":
    main()
