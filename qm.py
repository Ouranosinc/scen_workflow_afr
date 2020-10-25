# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Basic univariate quantile mapping post-processing algorithms.
#
# Contributors:
# 1. rousseau.yannick@ouranos.ca
# 2. marc-andre.bourgault@ggr.ulaval.ca (original)
# (C) 2020 Ouranos Inc., Canada
# ----------------------------------------------------------------------------------------------------------------------

import config as cfg
import dask.array
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def train(da_x: xr.DataArray, da_y: xr.DataArray, nq: int, group="time.dayofyear", kind=cfg.kind_add, time_win=0,
          detrend_order=0):

    """
    --------------------------------------------------------------------------------------------------------------------
    Compute quantile bias-adjustment factors.

    Parameters
    ----------
    da_x : xr.DataArray
        Training data, usually a model output whose biases are to be corrected.
    da_y : xr.DataArray
        Training target, usually an observed at-site time-series.
    nq : int
        Number of quantiles.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time.time'}
        Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    kind : {'+', '*'}
        The transfer operation, + for additive and * for multiplicative.
    time_win : int
        Number of days before and after any given day.
    detrend_order : int, None
        Polynomial order of detrending curve. Set to None to skip detrending.

    Returns
    -------
    xr.DataArray
        Delta factor computed over time grouping and quantile bins.
    --------------------------------------------------------------------------------------------------------------------
    """

    prop = ""
    if "." in group:
        dim, prop = group.split(".")
    else:
        dim = group

    # Detrend.
    cx = None
    cy = None
    if detrend_order is not None:
        da_x, _, cx = detrend(da_x, dim=dim, deg=detrend_order)
        da_y, _, cy = detrend(da_y, dim=dim, deg=detrend_order)

    # Define nodes. Here 'n' equally spaced points within [0, 1].
    # E.g. for nq=4 :  0---x------x------x------x---1
    dq = 1 / nq / 2
    q = np.linspace(dq, 1 - dq, nq)
    q = sorted(np.append([0.0001, 0.9999], q))

    # Group values by time, then compute quantiles. The resulting array will have new time and quantile dimensions.
    da_xq = None
    da_yq = None
    if "." in group:
        if prop == "dayofyear":
            if time_win == 0:
                da_xq = da_x.groupby(group).quantile(q)
                da_yq = da_y.groupby(group).quantile(q)
            else:
                da_xq = da_x.rolling(time=time_win, center=True).construct(window_dim="values").groupby(group).\
                    quantile(q, dim=["values", dim])
                da_yq = da_y.rolling(time=time_win, center=True).construct(window_dim="values").groupby(group).\
                    quantile(q, dim=["values", dim])
        elif prop == "month":
            da_xq = da_x.groupby(group).quantile(q)
            da_yq = da_y.groupby(group).quantile(q)
    else:
        da_xq = da_x.quantile(q, dim=dim)
        da_yq = da_y.quantile(q, dim=dim)

    # Compute correction factor.
    if kind == cfg.kind_add:
        da_train = xr.DataArray(da_yq - da_xq)
    elif kind == cfg.kind_mult:
        da_train = xr.DataArray(da_yq / da_xq)
    else:
        raise ValueError("kind must be + or *.")

    # Save input parameters as attributes of output DataArray.
    da_train.attrs[cfg.attrs_kind] = kind
    da_train.attrs[cfg.attrs_group] = group

    if detrend_order is not None:
        da_train.attrs["detrending_poly_coeffs_x"] = cx
        da_train.attrs["detrending_poly_coeffs_y"] = cy

    return da_train


def predict(da_x: xr.DataArray, da_qmf: xr.DataArray, interp=False, detrend_order=4):

    """
    --------------------------------------------------------------------------------------------------------------------
    Apply quantile mapping delta to an array.

    Parameters
    ----------
    da_x : xr.DataArray
        Data to predict on.
    da_qmf : xr.DataArray
        Quantile mapping factors computed by the `train` function.
    interp : bool
        Whether to interpolate between the groupings.
    detrend_order : int, None
        Polynomial order of detrending curve. Set to None to skip detrending.

    Returns
    -------
    xr.DataArray
        Input array with delta applied.
    --------------------------------------------------------------------------------------------------------------------
    """

    if "." in da_qmf.group:
        dim, prop = da_qmf.group.split(".")
    else:
        dim, prop = da_qmf.group, None

    if prop == "season" and interp:
        raise NotImplementedError

    # Detrend.
    trend = None
    coeffs = None
    if detrend_order is not None:
        da_x, trend, coeffs = detrend(da_x, dim=dim, deg=detrend_order)

    coord = da_x.coords[dim]

    # Add cyclical values to the scaling factors for interpolation.
    if interp:
        da_qmf = add_cyclic(da_qmf, prop)
        da_qmf = add_q_bounds(da_qmf)

    # Compute the percentile time series of the input array.
    da_q = da_x.load().groupby(da_qmf.group).apply(xr.DataArray.rank, pct=True, dim=dim)
    da_iq = xr.DataArray(da_q, dims=da_q.dims, coords=da_q.coords, name=cfg.stat_quantile + " index")

    # Create DataArrays for indexing
    # TODO.MAB: Adjust for different calendars if necessary.
    if interp:
        ind = da_q.indexes[dim]
        if prop == "month":
            y = ind.month - 0.5 + ind.day / ind.daysinmonth
        elif prop == "dayofyear":
            y = ind.dayofyear
        else:
            raise NotImplementedError

    else:
        y = getattr(da_q.indexes[dim], prop)

    da_it = xr.DataArray(y, dims=dim, coords={dim: coord}, name=dim + " group index")

    # Extract the correct quantile for each time step.
    # Interpolate both the time group and the quantile.
    if interp:
        da_factor = da_qmf.interp({prop: da_it, cfg.stat_quantile: da_iq})
    # Find quantile for nearest time group and quantile.
    else:
        da_factor = da_qmf.sel({prop: da_it, cfg.stat_quantile: da_iq}, method="nearest")

    # Apply correction factors.
    da_predict = da_x.copy()

    if da_qmf.kind == cfg.kind_add:
        da_predict = da_predict + da_factor
    elif da_qmf.kind == cfg.kind_mult:
        da_predict = da_predict * da_factor

    da_predict.attrs[cfg.attrs_bias] = True
    if detrend_order is not None:
        da_predict.attrs["detrending_poly_coeffs"] = coeffs

    # Add trend back.
    if detrend_order is not None:
        da_predict = da_predict + trend

    # Remove time grouping and quantile coordinates.
    da_predict = da_predict.drop_vars([cfg.stat_quantile, prop])

    return da_predict


def add_cyclic(da_qmf: xr.DataArray, att: str):

    """
    --------------------------------------------------------------------------------------------------------------------
    Reindex the scaling factors to include the last time grouping at the beginning and the first at the end.
    This is done to allow interpolation near the end-points.
    TODO.MAB: Use pad?

    Parameters
    ----------
    da_qmf : xr.DataArray
        Quantile mapping factors computed by the `train` function.
    att : str
        Attribute.
    --------------------------------------------------------------------------------------------------------------------
    """

    gc = da_qmf.coords[att]
    i = np.concatenate(([-1], range(len(gc)), [0]))
    da_qmf = da_qmf.reindex({att: gc[i]})
    da_qmf.coords[att] = range(len(da_qmf))

    return da_qmf


def add_q_bounds(da_qmf: xr.DataArray):

    """
    --------------------------------------------------------------------------------------------------------------------
    Reindex the scaling factors to set the quantile at 0 and 1 to the first and last quantile respectively.
    This is a naive approach that won't work well for extremes.
    TODO.MAB: Use pad?

    Parameters
    ----------
    da_qmf : xr.DataArray
        Quantile mapping factors computed by the `train` function.
    --------------------------------------------------------------------------------------------------------------------
    """

    att = cfg.stat_quantile
    q = da_qmf.coords[att]
    i = np.concatenate(([0], range(len(q)), [-1]))
    da_qmf = da_qmf.reindex({att: q[i]})
    da_qmf.coords[att] = np.concatenate(([0], q, [1]))

    return da_qmf


def calc_slope(x: [float], y: [float]):

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper that returns the slope from a linear regression fit of x and y values.

    Parameters
    ----------
    x : [float]
        X-values.
    y : [float]
        Y-values.
    --------------------------------------------------------------------------------------------------------------------
    """

    slope = stats.linregress(x, y)[0]

    return slope


def polyfit(da: xr.DataArray, deg=1, dim=cfg.dim_time):

    """
    --------------------------------------------------------------------------------------------------------------------
    Least squares polynomial fit.
    Fit a polynomial ``p(x) = p[deg] * x ** deg + ... + p[0]`` of degree `deg`
    Returns a vector of coefficients `p` that minimises the squared error.

    Parameters
    ----------
    da : xr.DataArray
        The array to fit
    deg : int, optional
        Degree of the fitting polynomial, Default is 1.
    dim : str
        The dimension along which the data will be fitted. Default is `time`.

    Returns
    -------
    output : xr.DataArray
        Polynomial coefficients with a new dimension to sort the polynomial
        coefficients by degree
    --------------------------------------------------------------------------------------------------------------------
    """

    # Remove NaNs.
    y = da.dropna(dim=dim, how="any")
    coord = y.coords[dim]

    # Compute the x value.
    x = get_index(coord)

    # Fit the parameters (lazy computation).
    coefs = dask.array.apply_along_axis(
            np.polyfit, da.get_axis_num(dim), x, y, deg=deg, shape=(deg+1, ), dtype=float)

    coords = dict(da.coords.items())
    coords.pop(dim)
    coords["degree"] = range(deg, -1, -1)

    dims = list(da.dims)
    dims.remove(dim)
    dims.insert(0, "degree")

    out = xr.DataArray(data=coefs, coords=coords, dims=dims)
    return out


def polyval(coefs: xr.DataArray, coord: xr.Coordinate):

    """
    --------------------------------------------------------------------------------------------------------------------
    Evaluate polynomial function.

    Parameters
    ----------
    coefs : xr.DataArray
        Polynomial coefficients as returned by polyfit.
    coord : xr.Coordinate
        Coordinate (e.g. time) used as the independent variable to compute polynomial.
    --------------------------------------------------------------------------------------------------------------------
    """

    x = get_index(coord)
    y = xr.apply_ufunc(np.polyval, coefs, x, input_core_dims=[["degree"], []], dask="allowed")

    return y


def detrend(obj, dim=cfg.dim_time, deg=1, kind=cfg.kind_add):

    """
    --------------------------------------------------------------------------------------------------------------------
    Detrend a series with a polynomial.
    The detrended object should have the same mean as the original.
    obs : ?
        ?
    --------------------------------------------------------------------------------------------------------------------
    """

    # Fit polynomial coefficients using Ordinary Least Squares.
    coefs = polyfit(obj, dim=dim, deg=deg)[0]

    # Set the 0th order coefficient to 0 to preserve.
    # Compute polynomial.
    trend = polyval(coefs, obj[dim])

    # Remove trend from original series while preserving means.
    # TODO.MAB: Get the residuals directly from polyfit.
    detrended = None
    if kind == cfg.kind_add:
        detrended = obj - trend - trend.mean() + obj.mean()
    elif kind == cfg.kind_mult:
        detrended = obj / trend / trend.mean() * obj.mean()

    return detrended, trend, coefs


def get_index(coord: xr.Coordinate):

    """
    --------------------------------------------------------------------------------------------------------------------
    Return x coordinate for polynomial fit.

    Parameters
    ----------
    coord : xr.Coordinate
        Coordinate (e.g. time) used as the independent variable to compute polynomial.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Scale from nanoseconds to days.
    f = 1e9 * 86400

    if pd.api.types.is_datetime64_dtype(coord.data):
        x = pd.to_numeric(coord) / f
    elif "calendar" in coord.encoding:
        dt = xr.coding.cftime_offsets.get_date_type(coord.encoding["calendar"])
        offset = dt(1970, 1, 1)
        x = xr.Variable(data=xr.core.duck_array_ops.datetime_to_numeric(coord, offset) / f, dims=(cfg.dim_time,))
    else:
        x = coord

    return x
