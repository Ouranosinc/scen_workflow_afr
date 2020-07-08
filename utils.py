# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Utility functions.
#
# Contact information:
# 1. bourgault.marcandre@ouranos.ca (original author)
# 2. rousseau.yannick@ouranos.ca (pimping agent)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

# Other packages.
import config as cfg
import glob
from math import sqrt
import numpy as np
import numpy.matlib
import os
import pandas as pd
import plot
import re
import scipy.stats
import utils
import xarray as xr
from cmath import rect, phase
from collections import defaultdict
from itertools import compress
from math import radians, degrees
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def scen_cluster(data, max_clusters=None, rsq_cutoff=None, n_clusters=None, variable_weights=None, sample_weights=None,
                 model_weights=None, make_graph=True):

    """
    --------------------------------------------------------------------------------------------------------------------
    Selects a subset of models located near the centroid of the K groups found by clustering.
    TODO: Support other formats, such as pandas dataframes.

    Parameters
    ----------
    data : numpy array NxP
        These are the values used for clustering. N is the number of models and P the number of variables/indicators.
    max_clusters : int, optional
        Maximum number of desired clusters (integer).
        MUTUALLY EXCLUSIVE TO N_CLUSTERS!
    rsq_cutoff : float (between 0 and 1), optional
        If a maximum number of clusters has been defined, this controls how many clusters are to be used.
    n_clusters : int, optional
        Number of desired clusters (integer).
        MUTUALLY EXCLUSIVE TO MAX_CLUSTERS!
        If neither of these parameters are set, default values are a max_cluster = N and rsq_cutoff = 0.90.
    variable_weights : numpy array of size P, optional
        This weighting can be used to influence of weight of the climate indices on the clustering itself.
    sample_weights : numpy array of size N, optional
        This weighting can be used to influence of weight of simulations on the clustering itself.
        For example, putting a weight of 0 on a simulation will completely exclude it from the clustering).
    model_weights : numpy array of size N, optional
        This weighting can be used to influence which model is selected within each cluster.
        This parameter has no influence whatsoever on the clustering itself!
    make_graph : boolean, optional
        Displays a plot of the R^2 vs. the number of clusters.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Number of simulations.
    n_sim = np.shape(data)[0]

    # Number of indicators.
    n_idx = np.shape(data)[1]

    if sample_weights is None:
        sample_weights = np.ones(n_sim)
    if model_weights is None:
        model_weights = np.ones(n_sim)
    if variable_weights is None:
        variable_weights = np.ones(shape=(1, n_idx))

    # Perform verifications.
    if max_clusters is not None and n_clusters is not None:
        log("Both max_clusters and n_clusters have been defined! Giving priority to n_clusters.", True)
        sel_method = "n_clusters"
    elif max_clusters is None and n_clusters is None:
        log("Neither max_clusters or n_clusters has been defined! Assuming that selection method = max_clusters", True)
        sel_method = "max_clusters"
        max_clusters = n_sim
    elif n_clusters is not None:
        sel_method = "n_clusters"
    else:
        sel_method = "max_clusters"

    if sel_method == "max_clusters" and rsq_cutoff is None:
        log("rsq_cutoff has not been defined! Using a default value of 0.90.", True)
        rsq_cutoff = 0.90

    if (rsq_cutoff is not None and rsq_cutoff >= 1) or (rsq_cutoff is not None and rsq_cutoff <= 0):
        log("rsq_cutoff must be between 0 and 1! Using a default value of 0.90.", True)
        rsq_cutoff = 0.90

    # Normalize the data matrix.
    # ddof=1 to be the same as Matlab's zscore.
    z = scipy.stats.zscore(data, axis=0, ddof=1)

    # Normalize weights.
    # Note: I don't know if this is really useful; this was in the MATLAB code.
    sample_weights = sample_weights / np.sum(sample_weights)
    model_weights = model_weights / np.sum(model_weights)
    variable_weights = variable_weights / np.sum(variable_weights)

    # Apply the variable weights to the Z matrix.
    variable_weights = numpy.matlib.repmat(variable_weights, n_sim, 1)
    z = z * variable_weights

    if sel_method == "max_clusters" or make_graph is True:
        sumd = np.empty(shape=n_sim)
        for nclust in range(n_sim):

            # This is k-means with only 10 iterations, to limit the computation times
            kmeans = KMeans(n_clusters=nclust + 1, n_init=10).fit(z, sample_weight=sample_weights)
            kmeans.fit(z)

            # Sum of the squared distance between each simulation and the
            # nearest cluster centroid.
            sumd[nclust] = kmeans.inertia_

        # R² of the groups vs. the full ensemble.
        rsq = (sumd[0] - sumd) / sumd[0]

        # Plot of rsq.
        plot.plot_rsq(rsq, n_sim)

        # If we actually need to find the optimal number of clusters, this is
        # where it is done.
        if sel_method == "max_clusters":
            # 'argmax' finds the first occurrence of rsq > rsq_cutoff but we need
            # to add 1 b/c of python indexing.
            n_clusters = np.argmax(rsq > rsq_cutoff) + 1

            if n_clusters <= max_clusters:
                log("Using " + str(n_clusters) +
                    " clusters has been found to be the optimal number of clusters.", True)
            else:
                n_clusters = max_clusters
                log("Using " + str(n_clusters) +
                    " clusters has been found to be the optimal number of clusters, but limiting to " +
                    str(max_clusters) + " as required by max_clusters.", True)

    # k-means clustering with 1000 iterations to avoid instabilities in the
    # choice of final scenarios.
    kmeans = KMeans(n_clusters=n_clusters, n_init=1000)

    # We use 'fit_' only once, otherwise it computes everything again.
    clusters = kmeans.fit_predict(z, sample_weight=sample_weights)

    # Squared distance between each point and each centroid.
    d = np.square(kmeans.transform(z))

    # Prepare an empty array in which to store the results.
    out = np.empty(shape=n_clusters)
    r = np.arange(n_sim)

    # In each cluster, find the closest (weighted) simulation and select it.
    for i in range(n_clusters):
        # Distance to the centroid for all simulations within the cluster 'i'.
        d_i = d[clusters == i, i]
        if d_i.shape[0] > 2:
            sig = np.std(d_i, ddof=1)
            # Standard deviation of those distances (ddof = 1 gives the same as Matlab's std function).
            # Weighted likelihood.
            like = scipy.stats.norm.pdf(d_i, 0, sig) * model_weights[clusters == i]
            # Index of maximum likelihood.
            argmax = np.argmax(like)
        elif d_i.shape[0] == 2:
            # Standard deviation would be 0 for a 2-simulation cluster, meaning
            # that model_weights would be ignored.
            sig = 1
            # Weighted likelihood.
            like = scipy.stats.norm.pdf(d_i, 0, sig) * model_weights[clusters == i]
            # Index of the maximum likelihood.
            argmax = np.argmax(like)
        else:
            argmax = 0
        # Index of the cluster simulations within the full ensemble.
        r_clust = r[clusters == i]
        out[i] = r_clust[argmax]

    out = out.astype(int)

    return out, clusters


def regrid_cdo(ds, new_grid, d_tmp, method="cubic"):

    """
    --------------------------------------------------------------------------------------------------------------------
    Regrids a dataset using CDO.
    TODO: Right now, this assumes that longitude and latitude are 1D.

    Parameters
    ----------
    ds : xarray
        Dataset.
    new_grid : xarray dataset
        ???
    tmpdir : str
        Location where to put the temporary netCDF files (string).
    method : [str]
        {"cubic"}
    --------------------------------------------------------------------------------------------------------------------
    """

    # Write a txt file with the mapping information.
    file = open(d_tmp + "new_grid.txt", "w")
    file.write("gridtype = lonlat\n")
    file.write("xsize = " + str(len(new_grid.lon.values.tolist())) + "\n")
    file.write("xvals = " + ",".join(str(num) for num in new_grid.lon.values) + "\n")
    file.write("ysize = " + str(len(new_grid.lat.values.tolist())) + "\n")
    file.write("yvals = " + ",".join(str(num) for num in new_grid.lat.values) + "\n")
    file.close()

    # Write the xarray dataset to a NetCDF.
    utils.save_dataset(ds, d_tmp + "old_data.nc")

    # Remap to the new grid.
    # TODO: To be fixed. It is not working well.
    if method == "cubic":
        os.system("cdo remapbic," + d_tmp + "new_grid.txt " + d_tmp + "old_data.nc " + d_tmp + "new_data.nc")
        # import subprocess
        # subprocess.call()
    elif method == "linear":
        os.system("cdo remapbil," + d_tmp + "new_grid.txt " + d_tmp + "old_data.nc " + d_tmp + "new_data.nc")
    else:
        "Invalid method!"

    # Load the new data.
    ds2 = xr.open_dataset(d_tmp + "new_data.nc")

    # Remove created files.
    os.system("rm " + d_tmp + "new_grid.txt")
    os.system("rm " + d_tmp + "old_data.nc")
    os.system("rm " + d_tmp + "new_data.nc")

    return ds2


def natural_sort(values):

    """
    --------------------------------------------------------------------------------------------------------------------
    Sorts a list numerically ('sorted([1,2,10,11]) would result in [1,10,11,2]).

    Parameters
    ----------
    values : List of values that require numerical sorting.
    --------------------------------------------------------------------------------------------------------------------
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(values, key=alphanum_key)


def uas_vas_2_sfc(uas, vas):

    """
    --------------------------------------------------------------------------------------------------------------------
    Transforms wind components to sfcwind and direction.
    TODO: Support other data formats.

    Parameters
    ----------
    uas : Wind component (x-axis; xr.DataArray).
    vas : Wind component (y-axis; xr.DataArray).
    --------------------------------------------------------------------------------------------------------------------
    """

    sfcwind = np.sqrt(np.square(uas) + np.square(vas))

    # Calculate the angle, then aggregate to daily.
    sfcwind_dirmath = np.degrees(np.arctan2(vas, uas))
    sfcwind_dirmet = (270 - sfcwind_dirmath) % 360.0

    # Calm winds have a direction of 0°. Northerly winds have a direction of 360°.
    # According to the Beaufort scale, "calm" winds are < 0.5 m/s
    sfcwind_dirmet.values[(sfcwind_dirmet.round() == 0) & (sfcwind >= 0.5)] = 360
    sfcwind_dirmet.values[sfcwind < 0.5] = 0

    return sfcwind, sfcwind_dirmet


def sfcwind_2_uas_vas(sfcwind, winddir, resample=None, nb_per_day=None):

    """
    --------------------------------------------------------------------------------------------------------------------
    Transforms sfcWind and direction as the wind components uas and vas
    TODO: Support other data formats.

    Parameters
    ----------
    sfcwind : xr.DataArray
        Wind speed.
    winddir : numpy array
        Direction from which the wind blows (using the meteorological standard).
    resample : str
        Whether or not the data needs to be resampled.
    nb_per_day : int
        Number of time steps per day.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Transform wind direction from the meteorological standard to the mathematical standard.
    sfcwind_dirmath = (-winddir + 270) % 360.0

    daily_avg_angle = 0.0
    if resample is not None:

        sfcwind = sfcwind.resample(time=resample).mean(dim="time", keep_attrs=True)

        # TODO.MAB: Remove nb_per_day and calculate it.

        # TODO.MAB: Improve the following line because it is very dirty.
        sfcwind_angle_per_day = sfcwind_dirmath.reshape((len(sfcwind.time), nb_per_day))

        # TODO.MAB: Improve the following line because it is also very dirty.
        daily_avg_angle = np.concatenate([[degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))]
                                          for angles in sfcwind_angle_per_day])

    uas = sfcwind.values * np.cos(np.radians(daily_avg_angle))
    vas = sfcwind.values * np.sin(np.radians(daily_avg_angle))

    return uas, vas


def list_cordex(p_ds, rcps):

    """
    --------------------------------------------------------------------------------------------------------------------
     Lists CORDEX simulations.

    Parameters
    ----------
    p_ds : str
        Path of data_source.
    rcps : [str]
        List of RCP scenarios.
    --------------------------------------------------------------------------------------------------------------------
    """

    list_f = {}

    # Find all the available simulations for a given RCP.
    for r in range(len(rcps)):

        d_format = p_ds + "*/*/AFR-*{r}".format(r=rcps[r]) + "/*/atmos/*/"
        d_list = glob.glob(d_format)
        d_list = [i for i in d_list if "day" in i]
        d_list.sort()

        # Remove timestep information.
        for i in range(0, len(d_list)):
            tokens = d_list[i].split("/")
            d_list[i] = d_list[i].replace(tokens[len(tokens) - 4], "*")

        # Keep only the unique simulation folders (with a * as the timestep).
        d_list = list(set(d_list))
        d_list_valid = [True] * len(d_list)

        # Keep only the simulations with all the variables we need.
        d_list = list(compress(d_list, d_list_valid))

        list_f[rcps[r] + "_historical"] = [w.replace(rcps[r], "historical") for w in d_list]
        list_f[rcps[r]] = d_list

    return list_f


def info_cordex(d_ds):

    """
    --------------------------------------------------------------------------------------------------------------------
     Creates an array that contains information about CORDEX simulations.
     A verification is made to ensure that there is at least one NetCDF file available for each variable.

    Parameters
    ----------
    d_ds : str
        Directory of data_source.

    Return
    ------
    sets : [str, str, str, str, [str]]
        List of simulation sets.
        sets[0] is the institute that created this simulation.
        sets[1] is the regional circulation model (RCM).
        sets[2] is the global circulation model (GCM).
        sets[3] is the simulation name.
        sets[4] is a list of variables available.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Results.
    sets = []

    # List directories containing simulation sets.
    d_format = d_ds + "*/*/AFR-*/day/atmos/*/"
    n_token = len(d_ds.split("/")) - 2

    # Loop through simulations sets.
    for i in glob.glob(d_format):
        tokens_i = i.split("/")

        # Extract institute, RGM, CGM and emission scenario.
        inst = tokens_i[n_token + 1]
        rgm  = tokens_i[n_token + 2]
        cgm  = tokens_i[n_token + 3].split("_")[1]
        scen = tokens_i[n_token + 3].split("_")[2]

        # Extract variables and ensure that there is at least one NetCDF file available for each  one.
        vars_i = []
        for j in glob.glob(i + "*/"):
            n_netcdf = len(glob.glob(j + "*.nc"))
            if n_netcdf > 0:
                tokens_j = j.split("/")
                var      = tokens_j[len(tokens_j) - 2]
                vars_i.append(var)

        sets.append([inst, rgm, cgm, scen, vars_i])

    return sets


def calendar(x, n_days_old=360, n_days_new=365):

    """
    --------------------------------------------------------------------------------------------------------------------
    Interpolates data from a 360- to 365-day calendar.

    Parameters
    ----------
    x : xarray
        Dataset.
    n_days_old: int
        Number of days in the old calendar.
    n_days_new: int
        Number of days in the new calendar.
    --------------------------------------------------------------------------------------------------------------------
    """

    x["backup"] = x.time

    # Put 360 on 365 calendar.
    ts         = x.assign_coords(time=x.time.dt.dayofyear / n_days_old * n_days_new)
    ts_year    = (ts.backup.dt.year.values - ts.backup.dt.year.values[0]) * n_days_new
    ts_time    = ts.time.values
    ts["time"] = ts_year + ts_time

    nb_year  = (ts.backup.dt.year.values[-1] - ts.backup.dt.year.values[0])+1
    time_new = np.arange(1, (nb_year*n_days_new)+1)

    # Create new times series.
    year_0  = str(ts.backup.dt.year.values[0])
    month_0 = str(ts.backup.dt.month.values[0]).zfill(2)
    day_0   = str(min(ts.backup.dt.day.values[0], 31)).zfill(2)
    year_1  = str(ts.backup.dt.year[-1].values)
    month_1 = str(ts.backup.dt.month[-1].values).zfill(2)
    if int(month_1) == 11:
        day_1 = str(min(ts.backup.dt.day[-1].values, 31)).zfill(2)
    else:
        day_1 = str(min(ts.backup.dt.day[-1].values + 1, 31)).zfill(2)
    date_0 = year_0 + "-" + month_0 + "-" + day_0
    date_1 = year_1 + "-" + month_1 + "-" + day_1
    time_date = pd.date_range(start=date_0, end=date_1)

    # Remove February 29th.
    time_date = time_date[~((time_date.month == 2) & (time_date.day == 29))]

    # Interpolate 365 days time series.
    ref_365 = ts.interp(time=time_new, kwargs={"fill_value": "extrapolate"}, method="nearest")

    # Recreate 365 time series.
    ref_365["time"] = time_date

    # DEBUG: Plot data.
    # DEBUG: plt.plot(np.arange(1,n_days_new+1),ref_365[:n_days_new].values)
    # DEBUG: plt.plot((np.arange(1, n_days_old+1)/n_days_old*n_days_new), ts[0:n_days_old].values,alpha=0.5)
    # DEBUG: plt.show()

    return ref_365


def list_files(p):

    """
    --------------------------------------------------------------------------------------------------------------------
    Lists files in a directory.

    Parameters
    ----------
    p : str
        Path of directory.
    --------------------------------------------------------------------------------------------------------------------
    """

    # List.
    p_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for p in f:
            if ".nc" in p:
                p_list.append(os.path.join(r, p))

    # Sort.
    p_list.sort()

    return p_list


def create_multi_dict(n, data_type):

    """
    --------------------------------------------------------------------------------------------------------------------
    Create directory.

    Parameters
    ----------
    n : int
        Number of dimensions.
    data_type : type
        Data type.
    --------------------------------------------------------------------------------------------------------------------
    """

    if n == 1:
        return defaultdict(data_type)
    else:
        return defaultdict(lambda: create_multi_dict(n - 1, data_type))


def calc_error(values_obs, values_pred):
    """
    -------------------------------------------------------------------------------------------------------------------
    Calculate the error between observed and predicted values.
    The methods and equations are presented in the following thesis:
    Parishkura D (2009) Évaluation de méthodes de mise à l'échelle statistique: reconstruction des extrêmes et de la
    variabilité du régime de mousson au Sahel (mémoire de maîtrise). UQAM.

    Parameters
    ----------
    values_obs : [float]
        Observed values.
    values_pred : [float]
        Predicted values.
    -------------------------------------------------------------------------------------------------------------------
    """

    error = -1

    # Method #1: Coefficient of determination.
    if cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_r2:
        error = r2_score(values_obs, values_pred)

    # Method #2: Mean absolute error.
    elif cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_mae:
        error = mean_absolute_error(values_obs, values_pred)

    # Method #3: Root mean square error.
    elif cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_rmse:
        error = sqrt(mean_squared_error(values_obs, values_pred))

    # Method #4: Relative root mean square error.
    elif cfg.opt_calib_bias_meth == cfg.opt_calib_bias_meth_rrmse:
        error = np.sqrt(np.sum(np.square((values_obs - values_pred) / np.std(values_obs))) / len(values_obs))

    return error


def log(msg, indent=False):

    """
    --------------------------------------------------------------------------------------------------------------------
    Log message.

    Parameters
    ----------
    msg : str
        Message.
    indent : bool
        If True, indent text.
    --------------------------------------------------------------------------------------------------------------------
    """

    ln = ""

    if indent:
        ln += " " * cfg.log_n_blank
    if (msg == "-") or (msg == "="):
        if indent:
            ln += msg * cfg.log_sep_len
        else:
            ln += msg * (cfg.log_sep_len + cfg.log_n_blank)
    else:
        ln += msg

    print(ln)


def save_dataset(ds, p):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save an xarray dataset as a NetCDF file.

    Parameters
    ----------
    ds : xarray
        Dataset.
    p : str
        Path of file to be created.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Create NetCDF file.
    ds.to_netcdf(p)


def save_plot(plot, p):

    """
    --------------------------------------------------------------------------------------------------------------------
    Save an xarray dataset as a NetCDF file.

    Parameters
    ----------
    plot : xarray
        Plot.
    path : str
        Path of file to be created.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Recursively create directories if the path does not exist.
    d = os.path.dirname(p)
    if not (os.path.isdir(d)):
        os.makedirs(d)

    # Discard file if it already exists.
    if os.path.exists(p):
        os.remove(p)

    # Create PNG file.
    plot.savefig(p)
