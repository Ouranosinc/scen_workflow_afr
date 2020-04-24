# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Script launcher
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import calib
import verif
import workflow


def main():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Step #1: Calibration (mandatory).
    calib.main()

    # Step #2: Workflow (mandatory).
    workflow.main()

    # Step #3: Verification (optional, useful to verify generated NetCDF files).
    verif.main()


if __name__ == "__main__":
    main()
