# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Functions that allow to verify generated NetCDF files.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca (current)
# 2. marc-andre.bourgault@ggr.ulaval.ca (second)
# 3. rondeau-genesse.gabriel@ouranos.ca (original)
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

# Current package.
import config as cfg
import plot
import utils


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Loop through station names.
    for stn in cfg.stns:

        # Loop through variables.
        for vi_name in cfg.variables:

            # Produce plots.
            try:
                plot.plot_ts_single(stn, vi_name)
                plot.plot_ts_mosaic(stn, vi_name)
            except FileExistsError:
                utils.log("Unable to locate the required files!", True)
                pass
