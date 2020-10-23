# scen_workflow_afr

## Purpose

This code produces climate scenarios using CORDEX NetCDF files and observations. It also performs the following tasks:
- automated bias adjustment and statistical downscaling when producing climate scenarios;
- download of reanalysis data;
- aggregation of hourly data to a daily frequency for reanalysis data;
- calculation of climate indices;
- calculation of statistics related to climate scenarios and indices (min., max., mean or sum, quantiles);
- generation of time series and maps.

The current version of the script (v1.0.4) performs these tasks at a station.

The technical documentation can be found [here](https://teams.microsoft.com/_#/my/file-personal?context=scen_workflow_afr&rootfolder=%252Fpersonal%252Fyanrou1_ouranos_ca%252FDocuments%252Fscen_workflow_afr).

## Releases
### v1.0.0

This is the initial stable release.
At the moment, there is one 'sample' climate indicator.

### v1.0.1

The following feature was added:
- calculation of statistics for climate scenarios to include user-defined quantiles.

### v1.0.2

The following features were added:
- generation of time series for climate variables (was done earlier for climate indices);
- determination of minimum and maximum values in plots of time series (scenarios and indices).

### v1.0.3

The following features were added:
- storage of script parameters in a .ini file (instead of launch.py);
- integration of parameters d_exec, d_in1 and d_in2 in .ini file.

### v1.0.4

The following features were added:
- a technical documentation describing the Python code;
- conversion of precipitation values (from kg m-2 s-1 to mm) in workflow and postprocess figures (time series).
- correction of an anomaly during the calculation of statistics of precipitation. 

## Upcoming features

The following features are currently being implemented:
- compatibility with gridded data (not only at a station);
- generation of maps for climate variables (interpolation).

## Contributing
This is a private development that is being used in production by climate services specialists. If you're interested in participating to the development, want to suggest features or report bugs, please leave us a message on the [issue tracker](https://github.com/Ouranosinc/scen_workflow_afr/issues).