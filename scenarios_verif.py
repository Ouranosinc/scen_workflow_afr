# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions that allow to verify generated NetCDF files.
#
# Authors:
# 1. rousseau.yannick@ouranos.ca
# 2. bourgault.marcandre@ouranos.ca (original)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import config as cfg
import plot


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    print("Module verif launched.")

    # Loop through station names.
    for stn in cfg.stns:

        # Loop through variables.
        for var in cfg.variables_cordex:

            # Produce plots.
            try:
                plot.plot_ts_single(stn, var)
                plot.plot_ts_mosaic(stn, var)
                plot.plot_monthly(stn, var)
            except FileExistsError:
                print("Unable to locate the required files!")
                pass

    print("Module verif completed successfully.")


if __name__ == "__main__":
    run()
