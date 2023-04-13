# -*- coding: utf-8 -*-
"""
Main body to perform analysis on mass spectrometry data. Two passes are
performed on the data: once to obtain all gas (background) signals, and again
to perform the analysis which subtracts this background.

Optionally can run from the command line as:
    python main.py <directory> output_name=<output_name> plot=<plot_diagnostic> stable_col=<stable_col>
with arguments:
    directory : path
        Location (relative or absolute) to the location in which data is stored
    output_name : str
        Name of the csv containing summary data. Default: output.csv
    plot : bool
        True/False switch for plotting summary data. Default: False
    stable_col : str
        Column heading used to determine which data is considered stable. Default: 11B

Authors: Matt Chandler (m.chandler@bristol.ac.uk), Michael Henehan (michael.henehan@bristol.ac.uk)
"""
from massspecanalysis import *

import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import interp1d
import sys

if __name__ == "__main__":
    #%% Get input parameters. Optionally can run from command line.
    # If run from the command line.
    # N.B. would be nice to move this over to argparse module. This is fine atm.
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    else:
        dirname = os.path.join(".", "data")
    # Default kwargs
    kwargs = {
        "output_name":"output.csv",
        "plot":False,
        "stable_col":"11B",
    }
    # Get the rest.
    if len(sys.argv) > 2:
        for kwarg in sys.argv[2:]:
            key, val = kwarg.split('=')
            kwargs[key] = val
    output_name = str(kwargs["output_name"])
    if output_name[-4:] != ".csv": # Make sure it's a csv.
        output_name += ".csv"
    plot_diagnostic = bool(kwargs["plot"])
    stable_col = str(kwargs["stable_col"])
        
    files = os.listdir(dirname)
    # Initialise output location.
    if "output" not in files:
        os.mkdir(os.path.join(dirname, "output"))
    
    #%% To regress the blank signal when corrected stable data later on, we need
    # two passes through the data. Collect the blank signals.
    gas_signals = pd.DataFrame([])
    for filename in files:
        # Check filetype, make sure that we get output data only.
        if filename[-4:] == ".exp":
            # Read in the data from this file.
            info, data = read_exp(os.path.join(dirname, filename), skipfooter=13, index_col=0, sep='\t', engine='python', usecols=range(8))
            mmnt_cols = data.columns[data.columns != "Time"]
            # Work out when the data is stable. Use 11B signal to do it.
            start_cycle, end_cycle = stable_cycles(data["Time"], data[stable_col])
            # Else define a custom point to start measuring the gas signal.
            # Use case tends to be ~500 cycles long, get the last ~50.
            end_cycle = 450
            # Work out what the blank gas signal is for all data cols.
            gas_signal, first_blank_cycle = blank_signal(data.loc[end_cycle:, "Time"], data.loc[end_cycle:, mmnt_cols], factor=3)
            # Store it for regression later.
            gas_signals = pd.concat([gas_signals, pd.DataFrame(gas_signal, columns=[filename[:-4]]).transpose()])                



    #%% Second pass: process the stable data.
    processed_data = pd.DataFrame([])
    for filename in files:
        # Check filetype
        if filename[-4:] == ".exp":
            # Read in the data.
            info, data = read_exp(os.path.join(dirname, filename), skipfooter=13, index_col=0, sep='\t', engine='python', usecols=range(8))
            mmnt_cols = data.columns[data.columns != "Time"]
            # Work out when sample is stable using 11B. Returns indices within which the sample is considered stable.
            start_cycle, end_cycle = stable_cycles(data["Time"], data[stable_col])
            stable_data = data.loc[start_cycle:end_cycle, data.columns != "Time"]
            
            # Regress the blank signals in time.
            stable_corrected = stable_data.copy()
            mmnt_time = [timestamp.timestamp() for timestamp in data.loc[start_cycle:end_cycle, "Time"].to_list()]
            gas_time  = [timestamp.timestamp() for timestamp in gas_signals.loc[:, "Time"].to_list()]
            for column in mmnt_cols:
                gas_vals = gas_signals.loc[:, column+" Mean"].to_list()
                # interp1d creates a function whose input is the query points. Change "kind" for different type of interpolation,
                # options at https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
                # Note that kind="next" subtracts the constant value coming up (this is the current behaviour)
                # And kind="linear" does a linear interpolation between previous and next points.
                func = interp1d(gas_time, gas_vals, kind="next", bounds_error=False, fill_value=(gas_vals[0], gas_vals[-1]))
                stable_corrected.loc[:, column] -= func(mmnt_time)
            
            # Combine stable_data and stable_corrected to remove any outliers.
            # Note that we only need one copy of Time column, so remove the one
            # in stable_corrected.
            corrected_cols = stable_corrected.columns + " Corrected"
            stable_corrected.columns = corrected_cols
            stable_data = pd.concat([stable_data, stable_corrected[mmnt_cols+" Corrected"]], axis=1)
            
            # Remove outliers. Hard coded, could be moved to be an input parameter.
            outlier_cols = ["11B/10B (1)", "11B/10B (1) Corrected"]
            stable_data = remove_outliers(stable_data, outlier_cols)
            
            #%% Do the analysis.
            mean   = stable_data.mean(axis=0)
            std    = stable_data.std(axis=0)
            cycles = len(stable_data)
            
            # Do the plotting
            if plot_diagnostic:
                # 11B vs cycle number: Showing stability.
                title = "{} Stable Data".format(filename[:-4])
                fig = plt.figure(figsize=(10, 5), dpi=100)
                ax = fig.add_subplot(111)
                ax.scatter(data.index, data[stable_col], 4, label="All data")
                ax.scatter(stable_data.index, stable_data[stable_col], 4, label="Stable data")
                ax.scatter(data.loc[first_blank_cycle:].index, data.loc[first_blank_cycle:, stable_col], 4, label="Gas signal")
                ax.set_xlabel("Cycle Number")
                ax.set_ylabel(f"{stable_col} Voltage")
                ax.set_title(title)
                ax.legend(loc=0)
                ax.grid()
                fig.savefig(os.path.join(dirname, "output", title+".png"))
                plt.close()
                
                # 11/10 B vs cycle number: Showing correction.
                title = "{} Correction".format(filename[:-4])
                fig = plt.figure(figsize=(10, 5), dpi=100)
                ax = fig.add_subplot(111)
                ax.scatter(stable_data.index, stable_data["11B/10B (1)"], 4, label="Uncorrected")
                ax.scatter(stable_data.index, stable_data["11B/10B (1) Corrected"], 4, label="Corrected")
                ax.set_xlabel("Cycle Number")
                ax.set_ylabel("11/10 B Voltage")
                ax.set_title(title)
                ax.legend(loc=0)
                ax.grid()
                fig.savefig(os.path.join(dirname, "output", title+".png"))
                plt.close()
                
            output_data = rearrange_means(mean, std, pd.Series(cycles, index=['Cycles']))
            
            #%% Store the data.
            processed_data = pd.concat([processed_data, pd.DataFrame(output_data, columns=[filename[:-4]]).transpose()])
    processed_data.to_csv(os.path.join(dirname, "output", output_name))