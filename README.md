# scen_workflow_afr

## Purpose

This code produces climate scenarios using CORDEX NetCDF files and observations. It also performs the following tasks:
- automated bias adjustment and statistical downscaling when producing climate scenarios;
- download of reanalysis data;
- aggregation of hourly data to a daily frequency for reanalysis data;
- calculation of climate indices;
- calculation of statistics related to climate scenarios and indices (min., max., mean or sum, quantiles);
- generation of time series and maps.

The current version of the script (v1.0.2, August 204h, 2020) performs these tasks at a station.

## Releases
### 1.0.0

This is the initial stable release.
At the moment, there is one 'sample' climate indicator.

### 1.0.1

The following feature was added:
- calculation of statistics for climate scenarios to include user-defined quantiles.

### 1.0.2

The following feature was added:
- generation of time series and maps for climate indicators.

## Upcoming versions

The following features will be added to upcoming versions:
- Storage of script parameters in a .ini file.
- Compatibility with grid data (not only at a station).

## Contributing
This is a private development that is being used in production by climate services specialists. If you're interested in participating to the development, want to suggest features, new indices or report bugs, please leave us a message on the `issue tracker <https://github.com/Ouranosinc/scen_workflow_afr/issues>`_.
