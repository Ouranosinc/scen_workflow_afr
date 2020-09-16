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
import ast
import config as cfg
import configparser
import datetime
import download
import indices
import os
import scenarios
import scenarios_calib as scen_calib
# import scenarios_verif as scen_verif
import statistics as stat
import utils


def load_params(p_ini):

    """
    --------------------------------------------------------------------------------------------------------------------
    Load parameters from an INI file.

    Parameters
    ----------
    p_ini : str
        Path of INI file.
    --------------------------------------------------------------------------------------------------------------------
    """

    config = configparser.ConfigParser()
    config.read(p_ini)

    def convert_to_1d(value, type):

        if type == str:
            value_new = ast.literal_eval(value)
        else:
            value = value.replace("[", "").replace("]", "").split(",")
            value_new = [type(i) for i in value]

        return value_new

    def convert_to_2d(value, type):

        value_new = []
        value = value[1:(len(value) - 1)].split("],")
        for i in range(len(value)):
            value_new.append(convert_to_1d(value[i], type))

        return value_new

    # Loop through sections.
    for section in config.sections():

        # Loop through keys.
        for key in config[section]:

            # Extract value.
            value = config[section][key]

            # PROJECT:
            if key == "country":
                cfg.country = ast.literal_eval(value)
            elif key == "project":
                cfg.project = ast.literal_eval(value)

            # OBSERVATIONS:
            elif key == "obs_src":
                cfg.obs_src = ast.literal_eval(value)
            elif key == "obs_src_username":
                cfg.obs_src_username = ast.literal_eval(value)
            elif key == "obs_src_password":
                cfg.obs_src_password = ast.literal_eval(value)
            elif key == "file_sep":
                cfg.file_sep = ast.literal_eval(value)
            elif key == "stns":
                cfg.stns = convert_to_1d(value, str)

            # CONTEXT:
            elif key == "rcps":
                cfg.rcps = ast.literal_eval(value)
            elif key == "per_ref":
                cfg.per_ref = convert_to_1d(value, int)
            elif key == "per_fut":
                cfg.per_fut = convert_to_1d(value, int)
            elif key == "per_hors":
                cfg.per_hors = convert_to_2d(value, int)
            elif key == "variables_cordex":
                cfg.variables_cordex = convert_to_1d(value, str)

            # DATA:
            elif key == "opt_download":
                cfg.opt_download = ast.literal_eval(value)
            elif key == "lon_bnds_download":
                cfg.lon_bnds_download = convert_to_1d(value, float)
            elif key == "lat_bnds_download":
                cfg.lat_bnds_download = convert_to_1d(value, float)
            elif key == "opt_aggregate":
                cfg.opt_aggregate = ast.literal_eval(value)

            # SCENARIOS:
            elif key == "opt_scen":
                cfg.opt_scen = ast.literal_eval(value)
            elif key == "opt_scen_regrid":
                cfg.opt_scen_regrid = ast.literal_eval(value)
            elif key == "lon_bnds":
                cfg.lon_bnds = convert_to_1d(value, float)
            elif key == "lat_bnds":
                cfg.lat_bnds = convert_to_1d(value, float)
            elif key == "radius":
                cfg.radius = value
            elif key == "sim_excepts":
                cfg.sim_excepts = convert_to_1d(value, str)
            elif key == "var_sim_excepts":
                cfg.var_sim_excepts = convert_to_1d(value, str)

            # CALIBRATION:
            elif key == "opt_calib":
                cfg.opt_calib = ast.literal_eval(value)
            elif key == "opt_calib_auto":
                cfg.opt_calib_auto = ast.literal_eval(value)
            elif key == "opt_calib_bias":
                cfg.opt_calib_bias = ast.literal_eval(value)
            elif key == "opt_calib_bias_meth":
                cfg.opt_calib_bias_meth = ast.literal_eval(value)
            elif key == "opt_calib_qqmap":
                cfg.opt_calib_qqmap = ast.literal_eval(value)
            elif key == "nq_default":
                cfg.nq_default = value
            elif key == "up_qmf_default":
                cfg.up_qmf_default = value
            elif key == "time_win_default":
                cfg.time_win_default = value

            # INDICES:
            elif key == "opt_idx":
                cfg.opt_idx = ast.literal_eval(value)
            elif key == "idx_names":
                cfg.idx_names = convert_to_1d(value, str)
            elif key == "idx_threshs":
                cfg.idx_threshs = convert_to_2d(value, float)

            # STATISTICS:
            elif key == "opt_stat":
                cfg.opt_stat = ast.literal_eval(value)
            elif key == "stat_quantiles":
                cfg.stat_quantiles = convert_to_1d(value, float)

            # VISUALIZATION:
            elif key == "opt_plot":
                cfg.opt_plot = ast.literal_eval(value)
            elif key == "d_bounds":
                cfg.d_bounds = ast.literal_eval(value)

            # ENVIRONMENT:
            elif key == "n_proc":
                cfg.n_proc = value
            elif key == "d_exec":
                cfg.d_exec = ast.literal_eval(value)
            elif key == "d_proj":
                cfg.d_proj = ast.literal_eval(value)
            elif key == "d_ra_raw":
                cfg.d_ra_raw = ast.literal_eval(value)
            elif key == "d_ra_day":
                cfg.d_ra_day = ast.literal_eval(value)


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Step #1: Parameters ----------------------------------------------------------------------------------------------

    # Load parameters from INI file.
    load_params("config_bf.ini")

    # Variables.
    cfg.priority_timestep = ["day"] * len(cfg.variables_cordex)

    # The following variables are determined automatically.
    if (cfg.obs_src != cfg.obs_src_era5) and (cfg.obs_src != cfg.obs_src_era5_land):
        cfg.d_stn = cfg.d_exec + cfg.country + "/" + cfg.project + "/" + cfg.cat_obs + "/" + cfg.obs_src + "/"
    cfg.d_sim = cfg.d_exec + "sim_climat/" + cfg.country + "/" + cfg.project + "/"
    if cfg.d_bounds != "":
        cfg.d_bounds = cfg.d_sim + "gis/" + cfg.d_bounds

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
