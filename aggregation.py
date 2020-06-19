# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Aggregation of a dataset (hourly to daily).
#
# Contact information:
# 1. rousseau.yannick@ouranos.ca
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

from datetime import datetime
import glob
import matplotlib.pyplot as plt
import os
import xarray as xr


def aggregate(path, var_code):

    """
    --------------------------------------------------------------------------------------------------------------------
    Convert a NetCDF file by aggregating hourly to daily frequency.

    Parameters
    ----------
    path : str
        Path of a NetCDF file.
    var_code : str
        Variable code.
    --------------------------------------------------------------------------------------------------------------------
    """

    opt_debug = True
    date = "1979-06-01"

    # Output file.
    dir_day = os.path.dirname(os.path.dirname(path)) + "/"
    fn_hour = os.path.basename(path)
    ds_hour = xr.open_dataset(path)[var_code]

    # Loop through statistics.
    for stat in ["mean", "min", "max", "sum"]:

        # Output file name.
        var_code_day = var_code + "_" + stat
        fn_day = dir_day + var_code_day + "/" + fn_hour.replace(var_code + "_", var_code_day + "_")
        fn_day = fn_day.replace("hour", "day")

        # Aggregate only if output file does not exist.
        if not(os.path.exists(fn_day)):

            # Aggregation.
            ds_day = None
            save     = False
            if stat == "mean":
                if var_code == "t2":
                    ds_day = ds_hour.groupby(ds_hour.time.dt.dayofyear).mean()
                    save = True
            elif stat == "min":
                if var_code == "t2":
                    ds_day = ds_hour.groupby(ds_hour.time.dt.dayofyear).min()
                    save = True
            elif stat == "max":
                if (var_code == "t2") or (var_code == "u10") or (var_code == "v10"):
                    ds_day = ds_hour.groupby(ds_hour.time.dt.dayofyear).max()
                    save = True
            elif stat == "sum":
                if (var_code == "tp") or (var_code == "e") or (var_code == "pev") or (var_code == "ssrd"):
                    ds_day = ds_hour.groupby(ds_hour.time.dt.dayofyear).sum()
                    save = True

            # Save NetCDF file.
            if save:
                dir_day = os.path.dirname(fn_day)
                if not (os.path.isdir(dir_day)):
                    os.makedirs(dir_day)
                ds_day.to_netcdf(fn_day)

        # DEBUG: Numerical test and plot for a given day of year.
        if opt_debug and (os.path.exists(fn_day)):

            # Day of year.
            dayofyear = datetime(int(date[0:4]), int(date[5:7]), int(date[8:10])).timetuple().tm_yday

            # Hourly data.
            ds_hour = xr.open_dataset(path)[var_code]
            ds_hour_sel = ds_hour.sel(time=date).values
            val_hour = -1.0
            if (var_code == "tp") or (var_code == "e") or (var_code == "pev") or (var_code == "ssrd"):
                val_hour = ds_hour_sel.sum() / (ds_hour_sel.shape[1] * ds_hour_sel.shape[2])
            elif (var_code == "t2") or (var_code == "u10") or (var_code == "v10"):
                val_hour = ds_hour_sel.mean()

            # Daily data.
            ds_day  = xr.open_dataset(fn_day)[var_code]
            val_day = ds_day.isel(dayofyear=dayofyear).values.mean()

            # DEBUG: Plot.
            plot_dayofyear(fn_day, var_code, dayofyear)


def plot_dayofyear(path, var_code, day_of_year):

    """
    --------------------------------------------------------------------------------------------------------------------
    Map.

    Parameters
    ----------
    path : str
        Path of a NetCDF file.
    var_code : str
        Variable code.
    day_of_year : int
        Day of year.
    --------------------------------------------------------------------------------------------------------------------
    """

    ds = xr.open_dataset(path)[var_code]

    # Data.
    var_desc = get_var_desc_era5(var_code)
    year_str = os.path.basename(path).replace(".nc", "").split("_")[-1]
    ds.isel(dayofyear=day_of_year).plot.pcolormesh(add_colorbar=True, add_labels=True,
                                                   cbar_kwargs=dict(orientation='vertical', pad=0.05, shrink=1, label=var_desc))

    # Format.
    fs = 10
    plt.title(year_str)
    plt.suptitle("", fontsize=fs)
    plt.xlabel("Longitude (º)", fontsize=fs)
    plt.ylabel("Latitude (º)", fontsize=fs)
    plt.tick_params(axis="x", labelsize=fs)
    plt.tick_params(axis="y", labelsize=fs)

    plt.close()


def get_var_desc_era5(var_code):

    """
    --------------------------------------------------------------------------------------------------------------------
    Gets the description of a variable.

    Parameters
    ----------
    var_code : str
        Variable code.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Determine description.
    var_desc = ""
    if var_code == "2t":
        var_desc = "Température ($^{\circ}$C)"
    elif var_code == "tp":
        var_desc = "Précipitation (m)"
    elif var_code == "u10":
        var_desc = "Vent (dir. est) (m/s)"
    elif var_code == "v10":
        var_desc = "Vent (dir. nord) (m/s)"
    elif var_code == "ssrd":
        var_desc = "Radiation solaire (J/m²)"
    elif var_code == "e":
        var_desc = "Évaporation (m)"
    elif var_code == "pev":
        var_desc = "Évapotranspiration potentielle (m)"

    return var_desc


def run():

    """
    --------------------------------------------------------------------------------------------------------------------
    Entry point.
    --------------------------------------------------------------------------------------------------------------------
    """

    set_code  = "era5"
    path_base = "/media/yrousseau/wd/"

    # ERA5 or ERA5_LAND.
    if (set_code == "era5_land") or (set_code == "era5"):
        path_base = path_base + "scenario/external_data/ecmwf/" + set_code + "/hour/"

    # MERRA2.
    elif set_code == "merra2":
        path_base = path_base + "scenario/external_data/nasa/merra2/hour/"

    # Loop through variables.
    vars = os.listdir(path_base)
    for var in vars:

        # Loop through files.
        paths = glob.glob(path_base + var + "/*.nc")
        paths.sort()
        for path in paths:

            # Perform aggregation.
            aggregate(path, var)


if __name__ == "__main__":
    run()