# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Basic univariate quantile mapping post-processing algorithms.
#
# Author:
# 1. marc-andre.bourgault@ggr.ulaval.ca (original)
# (C) 2020 Ouranos, Canada
# ----------------------------------------------------------------------------------------------------------------------

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def train(x, y, nq, group='time.dayofyear', kind="+", time_win=0, detrend_order=0):

    """
    --------------------------------------------------------------------------------------------------------------------
    Compute quantile bias-adjustment factors.

    Parameters
    ----------
    x : xr.DataArray
        Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
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
    if '.' in group:
        dim, prop = group.split('.')
    else:
        dim = group

    # Detrend.
    cx = None
    cy = None
    if detrend_order is not None:
        x, _, cx = detrend(x, dim=dim, deg=detrend_order)
        y, _, cy = detrend(y, dim=dim, deg=detrend_order)

    # Define nodes. Here 'n' equally spaced points within [0, 1].
    # E.g. for nq=4 :  0---x------x------x------x---1
    dq = 1 / nq / 2
    q = np.linspace(dq, 1 - dq, nq)
    q = sorted(np.append([0.0001, 0.9999], q))

    # Group values by time, then compute quantiles. The resulting array will have new time and quantile dimensions.
    xq = None
    yq = None
    if '.' in group:
        if prop == "dayofyear":
            if time_win == 0:
                xq = x.groupby(group).quantile(q)
                yq = y.groupby(group).quantile(q)
            else:
                xq = x.rolling(time=time_win, center=True).construct(window_dim="values").groupby(group).\
                    quantile(q, dim=["values", dim])
                yq = y.rolling(time=time_win, center=True).construct(window_dim="values").groupby(group).\
                    quantile(q, dim=["values", dim])
        if prop == "month":
            xq = x.groupby(group).quantile(q)
            yq = y.groupby(group).quantile(q)
    else:
        xq = x.quantile(q, dim=dim)
        yq = y.quantile(q, dim=dim)

    # Compute correction factor.
    if kind == "+":
        out = yq - xq
    elif kind == "*":
        out = yq / xq
    else:
        raise ValueError("kind must be + or *.")

    # Save input parameters as attributes of output DataArray.
    out.attrs["kind"] = kind
    out.attrs["group"] = group

    if detrend_order is not None:
        out.attrs["detrending_poly_coeffs_x"] = cx
        out.attrs["detrending_poly_coeffs_y"] = cy

    return out


def predict(x, qmf, interp=False, detrend_order=4):

    """
    --------------------------------------------------------------------------------------------------------------------
    Apply quantile mapping delta to an array.

    Parameters
    ----------
    x : xr.DataArray
        Data to predict on.
    qmf : xr.DataArray
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

    if '.' in qmf.group:
        dim, prop = qmf.group.split('.')
    else:
        dim, prop = qmf.group, None

    if prop == "season" and interp:
        raise NotImplementedError

    # Detrend.
    trend = None
    coeffs = None
    if detrend_order is not None:
        x, trend, coeffs = detrend(x, dim=dim, deg=detrend_order)

    coord = x.coords[dim]

    # Add cyclical values to the scaling factors for interpolation.
    if interp:
        qmf = add_cyclic(qmf, prop)
        qmf = add_q_bounds(qmf)

    # Compute the percentile time series of the input array.
    q = x.groupby(qmf.group).apply(xr.DataArray.rank, pct=True, dim=dim)
    iq = xr.DataArray(q, dims=q.dims, coords=q.coords, name="quantile index")

    # Create DataArrays for indexing
    # TODO: Adjust for different calendars if necessary.
    if interp:
        ind = q.indexes[dim]
        if prop == "month":
            y = ind.month - 0.5 + ind.day / ind.daysinmonth
        elif prop == "dayofyear":
            y = ind.dayofyear
        else:
            raise NotImplementedError

    else:
        y = getattr(q.indexes[dim], prop)

    it = xr.DataArray(y, dims=dim, coords={dim: coord}, name=dim + " group index")

    # Extract the correct quantile for each time step.
    if interp:  # Interpolate both the time group and the quantile.
        factor = qmf.interp({prop: it, "quantile": iq})
    else:  # Find quantile for nearest time group and quantile.
        factor = qmf.sel({prop: it, "quantile": iq}, method="nearest")

    # Apply correction factors.
    out = x.copy()

    if qmf.kind == "+":
        out += factor
    elif qmf.kind == "*":
        out *= factor

    out.attrs["bias_corrected"] = True
    if detrend_order is not None:
        out.attrs["detrending_poly_coeffs"] = coeffs

    # Add trend back.
    if detrend_order is not None:
        out += trend

    # Remove time grouping and quantile coordinates.
    return out.drop(["quantile", prop])


def add_cyclic(qmf, att):

    """
    --------------------------------------------------------------------------------------------------------------------
    Reindex the scaling factors to include the last time grouping at the beginning and the first at the end.
    This is done to allow interpolation near the end-points.
    TODO.MAB: Use pad.
    --------------------------------------------------------------------------------------------------------------------
    """

    gc = qmf.coords[att]
    i = np.concatenate(([-1], range(len(gc)), [0]))
    qmf = qmf.reindex({att: gc[i]})
    qmf.coords[att] = range(len(qmf))

    return qmf


def add_q_bounds(qmf):

    """
    --------------------------------------------------------------------------------------------------------------------
    Reindex the scaling factors to set the quantile at 0 and 1 to the first and last quantile respectively.
    This is a naive approach that won't work well for extremes.
    TODO.MAB: Use pad?
    --------------------------------------------------------------------------------------------------------------------
    """

    att = "quantile"
    q = qmf.coords[att]
    i = np.concatenate(([0], range(len(q)), [-1]))
    qmf = qmf.reindex({att: q[i]})
    qmf.coords[att] = np.concatenate(([0], q, [1]))

    return qmf


def _calc_slope(x, y):

    """
    --------------------------------------------------------------------------------------------------------------------
    Wrapper that returns the slope from a linear regression fit of x and y.
    --------------------------------------------------------------------------------------------------------------------
    """

    slope = stats.linregress(x, y)[0]

    return slope


def polyfit(da, deg=1, dim="time"):

    """
    --------------------------------------------------------------------------------------------------------------------
    Least squares polynomial fit.
    Fit a polynomial ``p(x) = p[deg] * x ** deg + ... + p[0]`` of degree `deg`
    Returns a vector of coefficients `p` that minimises the squared error.

    Parameters
    ----------
    da : xarray.DataArray
        The array to fit
    deg : int, optional
        Degree of the fitting polynomial, Default is 1.
    dim : str
        The dimension along which the data will be fitted. Default is `time`.

    Returns
    -------
    output : xarray.DataArray
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
    coords['degree'] = range(deg, -1, -1)

    dims = list(da.dims)
    dims.remove(dim)
    dims.insert(0, "degree")

    out = xr.DataArray(data=coefs, coords=coords, dims=dims)
    return out


def polyval(coefs, coord):

    """
    --------------------------------------------------------------------------------------------------------------------
    Evaluate polynomial function.

    Parameters
    ----------
    coord : xr.Coordinate
        Coordinate (e.g. time) used as the independent variable to compute polynomial.
    coefs : xr.DataArray
        Polynomial coefficients as returned by polyfit.
    --------------------------------------------------------------------------------------------------------------------
    """

    x = get_index(coord)
    y = xr.apply_ufunc(np.polyval, coefs, x, input_core_dims=[['degree'], []], dask='allowed')

    return y


def detrend(obj, dim="time", deg=1, kind="+"):

    """
    --------------------------------------------------------------------------------------------------------------------
    Detrend a series with a polynomial.
    The detrended object should have the same mean as the original.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Fit polynomial coefficients using Ordinary Least Squares.
    coefs = polyfit(obj, dim=dim, deg=deg)

    # Set the 0th order coefficient to 0 to preserve.
    # Compute polynomial.
    trend = polyval(coefs, obj[dim])

    # Remove trend from original series while preserving means.
    # TODO.MAB: Get the residuals directly from polyfit
    detrended = None
    if kind == "+":
        detrended = obj - trend - trend.mean() + obj.mean()
    elif kind == "*":
        detrended = obj / trend / trend.mean() * obj.mean()

    return detrended, trend, coefs


def get_index(coord):


    """
    --------------------------------------------------------------------------------------------------------------------
    Return x coordinate for polynomial fit.
    --------------------------------------------------------------------------------------------------------------------
    """

    # Scale from nanoseconds to days.
    f = 1e9 * 86400

    if pd.api.types.is_datetime64_dtype(coord.data):
        x = pd.to_numeric(coord) / f
    elif 'calendar' in coord.encoding:
        dt = xr.coding.cftime_offsets.get_date_type(coord.encoding['calendar'])
        offset = dt(1970, 1, 1)
        x = xr.Variable(data=xr.core.duck_array_ops.datetime_to_numeric(coord, offset) / f, dims=("time",))
    else:
        x = coord

    return x
