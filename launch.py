# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script launcher.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# (C) 2021 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import aggregate
import ast
import config as cfg
import configparser
import download
import indices
import os
import scenarios
import scenarios_calib as scen_calib
import utils


def load_params(
    p_ini: str
):

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

    def replace_right(s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def convert_to_1d(vals, type_inner):

        if type_inner == str:
            vals = ast.literal_eval(vals)
        elif type_inner == bool:
            vals = replace_right(vals.replace("[", "", 1), "]", "", 1).split(",")
            vals = [True if val == 'True' else False for val in vals]
        else:
            vals = replace_right(vals.replace("[", "", 1), "]", "", 1).split(",")
            for i_val in range(len(vals)):
                try:
                    vals[i_val] = int(vals[i_val])
                except ValueError:
                    try:
                        vals[i_val] = float(vals[i_val])
                    except ValueError:
                        pass

        return vals

    def convert_to_2d(vals, type_inner):

        vals_new = []
        vals = vals[1:(len(vals) - 1)].split("],")
        for i_val in range(len(vals)):
            val_i = convert_to_1d(vals[i_val], type_inner)
            vals_new.append(val_i)

        return vals_new

    # Track if a few variables were read.
    per_ref_read = False
    per_hors_read = False

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
                cfg.opt_ra = (cfg.obs_src == cfg.obs_src_era5) or \
                             (cfg.obs_src == cfg.obs_src_era5_land) or \
                             (cfg.obs_src == cfg.obs_src_merra2)
            elif key == "obs_src_username":
                cfg.obs_src_username = ast.literal_eval(value)
            elif key == "obs_src_password":
                cfg.obs_src_password = ast.literal_eval(value)
            elif key == "file_sep":
                cfg.f_sep = ast.literal_eval(value)
            elif key == "stns":
                cfg.stns = convert_to_1d(value, str)

            # CONTEXT:
            elif key == "rcps":
                cfg.rcps = ast.literal_eval(value)
            elif key == "per_ref":
                cfg.per_ref = convert_to_1d(value, int)
                if per_hors_read:
                    cfg.per_hors = [cfg.per_ref] + cfg.per_hors
                per_ref_read = True
            elif key == "per_fut":
                cfg.per_fut = convert_to_1d(value, int)
            elif key == "per_hors":
                cfg.per_hors = convert_to_2d(value, int)
                if per_ref_read:
                    cfg.per_hors = [cfg.per_ref] + cfg.per_hors
                per_hors_read = True
            elif key == "lon_bnds":
                cfg.lon_bnds = convert_to_1d(value, float)
            elif key == "lat_bnds":
                cfg.lat_bnds = convert_to_1d(value, float)
            elif key == "ctrl_pt":
                cfg.ctrl_pt = convert_to_1d(value, float)
            elif key == "variables_cordex":
                cfg.variables_cordex = convert_to_1d(value, str)
                for var in cfg.variables_cordex:
                    cfg.variables_ra.append(cfg.convert_var_name(var))
            elif key == "d_bounds":
                cfg.d_bounds = ast.literal_eval(value)
            elif key == "region":
                cfg.region = ast.literal_eval(value)

            # DATA:
            elif key == "opt_download":
                cfg.opt_download = ast.literal_eval(value)
            elif key == "variables_download":
                cfg.variables_download = convert_to_1d(value, str)
            elif key == "lon_bnds_download":
                cfg.lon_bnds_download = convert_to_1d(value, float)
            elif key == "lat_bnds_download":
                cfg.lat_bnds_download = convert_to_1d(value, float)
            elif key == "opt_aggregate":
                cfg.opt_aggregate = ast.literal_eval(value)

            # SCENARIOS:
            elif key == "opt_scen":
                cfg.opt_scen = ast.literal_eval(value)
            elif key == "radius":
                cfg.radius = float(value)
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
            elif key == "opt_calib_perturb":
                cfg.opt_calib_perturb = convert_to_2d(value, float)
            elif key == "opt_calib_quantiles":
                cfg.opt_calib_quantiles = convert_to_1d(value, float)
            elif key == "nq_default":
                cfg.nq_default = int(value)
                cfg.nq_calib = [cfg.nq_default]
            elif key == "up_qmf_default":
                cfg.up_qmf_default = float(value)
                cfg.up_qmf_calib = [cfg.up_qmf_default]
            elif key == "time_win_default":
                cfg.time_win_default = int(value)
                cfg.time_win_calib = [cfg.time_win_default]

            # INDICES:
            elif key == "opt_idx":
                cfg.opt_idx = ast.literal_eval(value)
            elif key == "idx_codes":
                cfg.idx_codes = convert_to_1d(value, str)
                for i in range(len(cfg.idx_codes)):
                    idx_name = cfg.extract_idx(str(cfg.idx_codes[i]))
                    cfg.idx_names.append(idx_name)
            elif key == "idx_params":
                cfg.idx_params = convert_to_2d(value, float)
                for i in range(len(cfg.idx_names)):
                    if cfg.idx_names[i] == cfg.idx_r10mm:
                        cfg.idx_params[i] = [10]
                    elif cfg.idx_names[i] == cfg.idx_r20mm:
                        cfg.idx_params[i] = [20]

            # STATISTICS:
            elif key == "opt_stat":
                cfg.opt_stat = ast.literal_eval(value) if ("," not in value) else convert_to_1d(value, bool)
            elif key == "opt_stat_quantiles":
                cfg.opt_stat_quantiles = convert_to_1d(value, float)
            elif key == "opt_stat_clip":
                cfg.opt_stat_clip = ast.literal_eval(value)
            elif key == "opt_save_csv":
                cfg.opt_save_csv = ast.literal_eval(value) if ("," not in value) else convert_to_1d(value, bool)

            # VISUALIZATION:
            elif key == "opt_plot":
                cfg.opt_plot = ast.literal_eval(value) if ("," not in value) else convert_to_1d(value, bool)
            elif key == "opt_plot_col_2cla_temp":
                cfg.opt_plot_col_2cla_temp = convert_to_1d(value, str)
            elif key == "opt_plot_col_2cla_prec":
                cfg.opt_plot_col_2cla_prec = convert_to_1d(value, str)
            elif key == "opt_plot_col_2cla_wind":
                cfg.opt_plot_col_2cla_wind = convert_to_1d(value, str)
            elif key == "opt_map":
                cfg.opt_map = [False, False]
                if cfg.opt_ra:
                    cfg.opt_map = ast.literal_eval(value) if ("," not in value) else convert_to_1d(value, bool)
            elif key == "opt_map_delta":
                cfg.opt_map_delta = [False, False]
                if cfg.opt_ra:
                    cfg.opt_map_delta = ast.literal_eval(value) if ("," not in value) else convert_to_1d(value, bool)
            elif key == "opt_map_clip":
                cfg.opt_map_clip = ast.literal_eval(value)
            elif key == "opt_map_quantiles":
                cfg.opt_map_quantiles = convert_to_1d(value, float)
                if str(cfg.opt_map_quantiles).replace("['']", "") == "":
                    cfg.opt_map_quantiles = None
            elif key == "opt_map_formats":
                cfg.opt_map_formats = convert_to_1d(value, str)
            elif key == "opt_map_spat_ref":
                cfg.opt_map_spat_ref = ast.literal_eval(value)
            elif key == "opt_map_res":
                cfg.opt_map_res = ast.literal_eval(value)
            elif key == "opt_map_discrete":
                cfg.opt_map_discrete = ast.literal_eval(value)
            elif key == "opt_map_col_temp_var":
                cfg.opt_map_col_temp_var = convert_to_1d(value, str)
            elif key == "opt_map_col_temp_idx_1":
                cfg.opt_map_col_temp_idx_1 = convert_to_1d(value, str)
            elif key == "opt_map_col_temp_idx_2":
                cfg.opt_map_col_temp_idx_2 = convert_to_1d(value, str)
            elif key == "opt_map_col_prec_var":
                cfg.opt_map_col_prec_var = convert_to_1d(value, str)
            elif key == "opt_map_col_prec_idx_1":
                cfg.opt_map_col_prec_idx_1 = convert_to_1d(value, str)
            elif key == "opt_map_col_prec_idx_2":
                cfg.opt_map_col_prec_idx_2 = convert_to_1d(value, str)
            elif key == "opt_map_col_prec_idx_3":
                cfg.opt_map_col_prec_idx_3 = convert_to_1d(value, str)
            elif key == "opt_map_col_wind_var":
                cfg.opt_map_col_wind_var = convert_to_1d(value, str)
            elif key == "opt_map_col_wind_idx_1":
                cfg.opt_map_col_wind_idx_1 = convert_to_1d(value, str)

            # ENVIRONMENT:
            elif key == "n_proc":
                cfg.n_proc = int(value)
            elif key == "d_exec":
                cfg.d_exec = ast.literal_eval(value)
            elif key == "d_proj":
                cfg.d_proj = ast.literal_eval(value)
            elif key == "d_ra_raw":
                cfg.d_ra_raw = ast.literal_eval(value)
            elif key == "d_ra_day":
                cfg.d_ra_day = ast.literal_eval(value)
            elif key == "opt_trace":
                cfg.opt_trace = ast.literal_eval(value)
            elif key == "opt_force_overwrite":
                cfg.opt_force_overwrite = ast.literal_eval(value)


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Step #1: Parameters ----------------------------------------------------------------------------------------------

    # Load parameters from INI file.
    load_params("config.ini")

    # Process identifier.
    cfg.pid = os.getpid()

    # Variables.
    cfg.priority_timestep = ["day"] * len(cfg.variables_cordex)

    # The following variables are determined automatically.
    d_base = cfg.d_exec + cfg.country + "/" + cfg.project + "/"
    obs_src_region = cfg.obs_src + ("_" + cfg.region if (cfg.region != "") and cfg.opt_ra else "")
    cfg.d_stn = d_base + cfg.cat_stn + "/" + obs_src_region + "/"
    cfg.d_res = cfg.d_exec + "sim_climat/" + cfg.country + "/" + cfg.project + "/"
    if cfg.d_bounds != "":
        cfg.d_bounds = d_base + "gis/" + cfg.d_bounds

    # Log file.
    cfg.p_log = cfg.d_res + "stn/" + obs_src_region + "/log/" + utils.get_datetime_str() + ".log"

    # Calibration file.
    cfg.p_calib = cfg.d_res + "stn/" + obs_src_region + "/" + cfg.p_calib
    d = os.path.dirname(cfg.p_calib)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # ------------------------------------------------------------------------------------------------------------------

    utils.log("=")
    utils.log("PRODUCTION OF CLIMATE SCENARIOS & CALCULATION OF CLIMATE INDICES                ")
    utils.log("Python Script created by Ouranos, based on xclim and xarray libraries.          ")
    utils.log("Script launched: " + utils.get_datetime_str())

    # Display configuration.
    utils.log("=")
    utils.log("Country                : " + cfg.country)
    utils.log("Project                : " + cfg.project)
    utils.log("Variables (CORDEX)     : " + str(cfg.variables_cordex).replace("'", ""))
    n_i = len(cfg.idx_names)
    for i in range(n_i):
        params_i = str(cfg.idx_params[i]).replace("'", "").replace("(", "[").replace(")", "]").replace("\\n", "")
        utils.log("Climate index #" + ("0" if i < 9 else "") + str(i + 1) + "      : " + params_i)
    if cfg.opt_ra:
        utils.log("Reanalysis set         : " + cfg.obs_src)
        utils.log("Variables (reanalysis) : " + str(cfg.variables_ra).replace("'", ""))
    else:
        utils.log("Stations               : " + str(cfg.stns))
    utils.log("Emission scenarios     : " + str(cfg.rcps).replace("'", ""))
    utils.log("Reference period       : " + str(cfg.per_ref))
    utils.log("Future period          : " + str(cfg.per_fut))
    utils.log("Horizons               : " + str(cfg.per_hors).replace("'", ""))
    if (cfg.region != "") and cfg.opt_ra:
        utils.log("Region                 : " + cfg.region)

    # Step #2: Download and aggregation --------------------------------------------------------------------------------

    # Download data.
    utils.log("=")
    msg = "Step #2a  Downloading climate data"
    if cfg.opt_download:
        utils.log(msg)
        download.run()
    else:
        utils.log(msg + " (not required)")

    # Aggregate reanalysis data to daily frequency
    utils.log("=")
    msg = "Step #2b  Aggregating hourly data"
    if cfg.opt_aggregate and cfg.opt_ra:
        utils.log(msg)
        aggregate.run()
    else:
        utils.log(msg + " (not required)")

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
    #    cfg.get_d_scen(<station_name>, "fig")/calib/
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

    # Clean NetCDF files.
    utils.clean_netcdf(cfg.d_stn)
    utils.clean_netcdf(cfg.get_d_scen(cfg.obs_src, "", ""))

    # Initialization.
    scen_calib.init_calib_params()

    # Steps #2-5,8: Production of scenarios, plots and statistics.
    scenarios.run()

    # Steps #6,8: Calculation of indices, plots and statistics.
    indices.run()

    utils.log("=")
    utils.log("Script completed: " + utils.get_datetime_str())


if __name__ == "__main__":
    main()
