# mass-spec-analysis
Analyses output of mass spectrometry study of boron isotopes to produce summary data.

The program finds the background (gas) signal made available in the last ~50 cycles, to remove from subsequent analysis.

A second pass which computes the summary data is performed. Only 'stable cycles' are considered in analysis, for which the sample is considered stable within the machine. Background is then removed by interpolating gas signals, and output data is saved.

Developed as part of the [Ask-JGI](https://www.bristol.ac.uk/golding/ask-jgi/) support service run by the [Jean Golding Institute](https://www.bristol.ac.uk/golding/)

## Files:
- `massspecanalysis.py` : module containing functions used for analysis.
- `main.py` : script designed to be executed from the command line or within any Python IDE. Spyder was used to develop it. Call this from the command line with command `python main.py <directory> output_name=<output_name> plot=<plot_diagnostic> stable_col=<stable_col>`. All arguments here are optional and therefore do not need to be included when calling the function.
  - `directory` : path, relative or absolute, to the location of the data to be summarised. Default: `.\data`
  - `output_name` : csv filename which will be saved. Default: `output.csv`
  - `plot_diagnostic` : `True`/`False` switch to be used to save plots. Default: `False`
  - `stable_col` : column heading which will be used to determine which data is stable. Default: `11B`
- `analysis.ipynb` : Functionally equivalent to `main.py`, written for use in Jupyter notebooks.
