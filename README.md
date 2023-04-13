# mass-spec-analysis
Analyses output of mass spectrometry study of boron isotopes to produce summary data.

The program finds the background (gas) signal made available in the last ~50 cycles, to remove from subsequent analysis.

A second pass which computes the summary data is performed. Only 'stable cycles' are considered in analysis, for which the sample is considered stable within the machine. Background is then removed by interpolating gas signals, and output data is saved.
