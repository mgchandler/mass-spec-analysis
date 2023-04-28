# -*- coding: utf-8 -*-
"""
Functions used in the analysis of mass spectrometry data.
Authors: Matt Chandler (m.chandler@bristol.ac.uk), Michael Henehan (michael.henehan@bristol.ac.uk)
"""
import datetime as dt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings

#%% Input-output functions

def str_to_timestamp(date, time_string):
    """ 
    Converts times imported from *.exp files to pandas Timestamps. `date`
    should be a string of format "DD/M/YYYY" and `time_string` should be a 
    pandas Series of format "HH:MM:SS:sss" where sss is the fractional
    component of s. Returns a pandas Series of these timestamps.
    """
    # Parse strings into datetime objects.
    timestamps = [dt.datetime.strptime("{} {}".format(date, time_string.iloc[i]), '%d/%m/%Y %H:%M:%S:%f') for i in range(len(time_string))]
    # Output as pandas Series object.
    return pd.Series(timestamps, time_string.index, name=time_string.name)

def read_exp(filename, **kwargs):
    """ 
    Wrapper for pandas read_csv(). Does this to convert times into datetime
    objects, as pandas does not recognise the format provided. Would prefer
    numpy arrays but DataFrames support multi-typed arrays out of the box where
    numpy uses weird structured dtypes. `filename` is a string and `kwargs` are
    passed directly through to pd.read_csv(). Returns a dictionary of header
    info and a pandas DataFrame containing the data.
    
    N.B. Note that footers are not automatically found: this should be
    specified in `kwargs` as `skipfooter=13`, for example.
    """
    # Extract header information.
    info = extract_header(filename)
    # If header not externally provided, use the number we've just found.
    if "header" not in kwargs:
        kwargs["header"] = info["header length"]
    # Read in the data. Pass kwargs straight through - note that types cannot
    # be determined automatically if footer is present.
    data = pd.read_csv(filename, **kwargs)
    # Convert times from *.exp to datetimes so we can actually use them.
    data["Time"] = str_to_timestamp(info["Analysis date"], data["Time"])
    return info, data

def extract_header(filename, sep="\t"):
    """
    Reads in data using pythonic open(), extracting header info. Ignores
    footer info and returns header as a dict and data as a pandas DataFrame.
    `filename` should be a string, `sep` is the delimiter. Returns a dictionary
    containing header info.
    """
    header_info = {}
    with open(filename, 'r') as file:
        for num, line in enumerate(file):
            line = line.split(sep)
            # Are we in the data?
            if line[0].isnumeric():
                # Get size of header to skip when reading in data. Keep column
                # headings.
                header_info["header length"] = num-1
                break
            # Must be in the header.
            else:
                line_info = line[0].split(': ')
                # If line contains ': ' then it is part of the header.
                if len(line_info) > 1:
                    header_info[line_info[0]] = line_info[1]
    return header_info

def rearrange_means(means, std, supplementary=None):
    """ 
    Stack the means and SE next to one another in an output Series. Append
    any supplementary data to the end.
    
    Parameters
    ----------
    means : pd.Series
        Set of means with length (N,).
    std : pd.Series
        Set of standard errors with the same length and index names as `means`.
    supplementary : pd.Series, optional
        Any additional information. Indices should be unique. The default is
        None.
    
    Returns
    -------
    output : pd.Series
        Rearranged input data to stack related `means` and `std` into the same
        series.
    """
    # Sort them to make sure they are in the same order.
    means.sort_index(inplace=True)
    std.sort_index(inplace=True)
    mean_index = means.index
    std_index = std.index
    # Check that they contain info about the same data.
    # Series have both been sorted so they should be in the same order.
    if (mean_index != std_index).any():
        raise ValueError("rearrange_means: Indices of `means` and `std` should be the same.")
    mean_index += " Mean"
    std_index += " SE"
    means.index = mean_index
    std.index = std_index
    # Concatenate them and rearrange the order.
    output_data = pd.concat([means, std])
    new_order = []
    for i in range(len(means)):
        new_order += [i, i+len(means)]
    output_data = output_data[output_data.index[new_order]]
    # Add on any extra data.
    if supplementary is not None:
        output_data = pd.concat([output_data, supplementary])
    return output_data



#%% Analysis functions

def diff(time, values, num=5, return_residual=False):
    """ 
    Determines the differential of `values` with respect to `time`. To smooth
    out noise, this is done as a fitting over a subset of `num` points.
    Makes use of the example in the documentation of np.linalg.lstsq to do it.
    Note from documentation of lstsq, residuals are sum of squared residuals,
    or variance of residuals. Typically want to use std = √variance.

    Parameters
    ----------
    time : pd.Series
        Time series of shape (N,) containing either numeric values or 
        dt.datetime objects, the times at which `values` were captured.
    values : pd.DataFrame
        Values of shape (N, m), measurements obtained for which the time
        differential will be computed. Zeroth dimension should be N large (same
        size as `time`) and may contain multiple data sets in the first axis.
    num : int, optional
        The number of consecutive points which will be used to regress the
        slope at each time. The function iterates through each column of the
        data taking `num` points, finding the slope of each set and returns
        this value. The default is 5.
    return_residual : bool, optional
        Whether to also return the residual values, might be used for
        evaluating confidence in regression at each point. The default is
        False.
    
    Returns
    -------
    time : pd.Series
        Reduced time series at which the calculated slope is found.
        Mathematically this should be the mean of the `num` values used at
        each point, however it seems more useful to return the closest time in
        the input data to this average.
    dVdt : np.ndarray
        Slopes found from regressing `values` of length equal to the reduced
        `time` Series and m columns.
    res : np.ndarray, optional
        Residuals output directly from each call to np.linalg.lstsq(), the sum
        of squares of the residuals from the fit. This will not be returned if
        `return_residual` is False.
    """
    if len(time) != len(values):
        raise ValueError("diff(): `time` and `values` should have the same size.")
    values = np.asarray(values).reshape(values.shape[0], -1)
    # Initialise output arrays
    dVdt = np.zeros((len(time)-num, values.shape[1]))
    res = np.zeros((len(time)-num, values.shape[1]))
    for j in range(values.shape[1]):
        for i in range(time.shape[0]-num):
            # Construct input data. We only care about slope, so collect x as time
            # after the first sample.
            x = (np.asarray(time, dtype=float)[i:i+num] - np.asarray(time, dtype=float)[0]) / 10**9
            y = np.asarray(values[i:i+num, j])
            A = np.vstack([x, np.ones(len(x))]).T
            # Do regression for this point and save it.
            (m, c), r, _, _ = np.linalg.lstsq(A, y, rcond=None)
            dVdt[i, j] = m
            res[i, j] = r[0]
    # dV/dt corresponds to the middle point of each x - it's more valuable to
    # have this line up with the input `time` array, so output relative to that
    # rather than storing the actual centre.
    time = time.iloc[int((num+1)/2):len(time)-int((num)/2)]
    if return_residual:
        return time, dVdt, res
    else:
        return time, dVdt

def stable_cycles(time, values, start_offset=10, end_offset=5, slope=.001):
    """ 
    Returns the start and end index within `values` for which the sample is 
    stable within the machine. This is done based on the slope of `values` at
    each point: it is assumed that the largest slope will occur at the start
    and at the end, and the sample will be stable at every point in between.

    Parameters
    ----------
    time : pd.Series
        Time series of shape (N,) containing the times at which `values` were
        captured.
    values : pd.Series or pd.DataFrame
        Values of shape (N, m) containing the measurements taken at times
        corresponding to `time`. Note that only the first column will be used.
    num : int, optional
        The number of consecutive points which will be used to regress the
        slope at each time, passed through to diff().
    slope : TYPE, optional
        The prominence used to locate peaks, passed through to find_peaks().
        Peaks found in dVdt will be larger than this number. Note that only
        the first and last peak are used: this only needs to be large enough to
        ignore blips in the 11B gas signal. The default is .001, will need
        tailoring based on blips.
    
    Returns
    -------
    ramp_up : int
        The cycle number at which the first column in `values` is considered
        stable from.
    ramp_down : int
        The cycle number at which the first column in `values` is considered
        stable until.
    """
    if len(values.shape) != 1:
        if values.shape[1] != 1:
            warnings.warn("stable_cycles(): `values` contains multiple columns. Only the first column will be used to compute the stable cycles.")
    # Compute the slope of `values` wrt `time`
    t, dVdt = diff(time, values)
    
    # Find the peaks of this plot. It is expected that there will be a lot of
    # peaks when the sample is stable, but when it is ramping up and down there
    # will not be much at all. In this case, we only care about the first and
    # last peak.
    peaks = find_peaks(np.abs(dVdt[:, 0]), slope)[0]
    if len(peaks) < 2:
        warnings.warn("No peaks found in this data set!")
        return 0, 0
    else:
        # Get the cycle numbers of the first and last peak.
        ramp_up   = t.index.to_list()[peaks[0]]
        ramp_down = t.index.to_list()[peaks[-1]]
        # These points are where slope is largest - move start and end inwards to
        # reduce the amount of ramping left in the data.
        return ramp_up + start_offset, ramp_down - end_offset

def blank_signal(time, values, num=5, factor=3):
    """ 
    Finds the mean of the blank gas signal. Values are first filtered to remove
    outliers from 3std from the mean to remove blips in the output signal. The
    mean is then calculated over each column of `values` for which the slope is
    zero (i.e. determine where in each column of `values` the confidence
    interval of the slope contains zero, and find the mean of all points after
    then).
    
    Parameters
    ----------
    time : pd.Series
        Time series of shape (N,) containing the times at which `values` were
        captured.
    values : pd.DataFrame
        Values of shape (N, m) containing the measurements taken at times
        corresponding to `time`.
    num : int, optional
        The number of consecutive points which will be used to regress the
        slope at each time, passed through to diff().
    factor : float, optional
        The multiple of std which defines the window of good data. The default
        is 3.
    
    Returns
    -------
    means : pd.Series
        A table of means calulated from each column in `values`.
    """
    # First pass to remove blips and large portion of ramping down.
    mean, std = values.mean(0), values.std(0)
    blip_window = (values > mean-factor*std).all(1) & (values < mean+factor*std).all(1)
    time = time[blip_window]
    values = values[blip_window]
    # Compute the slope of `values` wrt `time`.
    t, dVdt, res = diff(time, values, num, return_residual=True)
    # Slope -> zero with time. Work out when slope is no longer statistically
    # significant -ve: will occur when (dVdt +- 3*std) straddles zero.
    slope_window = ((dVdt + factor*np.sqrt(res) > 0) & (dVdt - factor*np.sqrt(res) < 0)).all(axis=1)
    # Compute the mean gas signal after the slope is zero.
    first_blank_cycle = t[slope_window].index.to_list()[0]
    # Return means, SE and timestamp.
    avg_time = pd.Series([time.loc[first_blank_cycle], time.iloc[-1]]).mean()
    means = rearrange_means(values.loc[first_blank_cycle:].mean(0), values.loc[first_blank_cycle:].std(0), pd.Series(avg_time, index=["Time"]))
    
    return means, first_blank_cycle

def remove_outliers(data, outlier_columns, factor=3):
    """
    Iteratively remove outliers from the data set. Outliers are defined as 
    being outside of a multiple of `factor` from the mean.

    Parameters
    ----------
    data : pd.DataFrame
        The data for which outliers will be removed. Should be entirely numeric
        data for which means and std can be found.
    outlier_columns : list
        The columns which are used to determine outliers.
    factor : float, optional
        The multiple of std which defines the window of good data. The default
        is 3.

    Returns
    -------
    data : pd.DataFrame
        Reduced `data` after outliers have been removed. Note that this is a
        view of input `data` rather than a copy - do not try to change values.
    """
    # Initialise outliers array. Do this to make sure we have at least one pass.
    in_range = np.array([False])
    # Repeat process for as long as any data lies outside of 3std from the mean.
    while ~in_range.any():
        # Find mean and std for corrected and uncorrected data.
        mean = data.mean(0)
        std  = data.std(0)
        # Find outliers.
        in_range = ((data.loc[:, outlier_columns] < mean.loc[outlier_columns] - factor*std.loc[outlier_columns]) |  # Find data outside of the range
                    (data.loc[:, outlier_columns] > mean.loc[outlier_columns] + factor*std.loc[outlier_columns])).any(axis=1) # All columns should be in the range for the cycle to be valid.
        in_range = ~in_range
        # Keep everything that isn't an outlier.
        data = data[in_range.to_list()]
    
    return data