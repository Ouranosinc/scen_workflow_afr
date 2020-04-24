# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script launcher.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import calib
import config as cfg
import verif
import wflow


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Module launcher launched.")

    # Step #1: Set parameters ------------------------------------------------------------------------------------------

    # The parameter values below override 'config' values.

    # Structure of file system.
    cfg.country      = "burkina"
    cfg.project      = "pcci"
    cfg.user_name    = "yrousseau"
    cfg.obs_provider = "anam"

    # Reference and future periods.
    cfg.per_ref = [1988, 2017]
    cfg.per_fut = [1988, 2095]

    # Stations.
    # cfg.stn_names = ["boromo", "diebougou", "farakoba", "gaoua", "kassoum", "leo", "po", "valleedukou"]
    cfg.stn_names = ["boromo"]

    # Variables.
    # cfg.variables = [cfg.var_tas, cfg.var_tasmin, cfg.var_tasmax, cfg.var_pr, cfg.var_uas, cfg.var_vas]
    cfg.variables         = [cfg.var_uas, cfg.var_vas]
    cfg.priority_timestep = ["day"] * len(cfg.variables)

    # List of simulation and var-simulation combinations that must be avoided to avoid a crash.
    sim_excepts = ["RCA4_AFR-44_ICHEC-EC-EARTH_rcp85",
                   "RCA4_AFR-44_MPI-M-MPI-ESM-LR_rcp85",
                   "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp45.nc",
                   "HIRHAM5_AFR-44_ICHEC-EC-EARTH_rcp85.nc"]
    var_sim_excepts = [cfg.var_pr + "_RCA4_AFR-44_CSIRO-QCCCE-CSIRO-Mk3-6-0_rcp85.nc",
                       cfg.var_tasmin + "_REMO2009_AFR-44_MIROC-MIROC5_rcp26.nc"]

    # Bias correction.
    # For calibration.
    cfg.nq_calib       = [40]
    cfg.up_qmf_calib   = [2.5] # range(2,3,1)
    cfg.time_int_calib = [cfg.time_int] #range(1, 37, 5)
    # For workflow.
    cfg.nq             = 50
    cfg.up_qmf         = 3
    cfg.time_int       = 30

    # Step #2: Calibration ---------------------------------------------------------------------------------------------

    # This step is mandatory.

    calib.main()

    # Step #3: Workflow (mandatory) ------------------------------------------------------------------------------------

    # This step is mandatory.
    # Calibration is currently manual. It is based on looking at the plots generated in step #2. This involves that
    # steps are to be run individually.

    wflow.main()

    # Step #4: Verification --------------------------------------------------------------------------------------------

    # This step is optional. It is useful to verify generated NetCDF files.

    verif.main()

    print("Module launcher completed successfully.")


if __name__ == "__main__":
    main()
