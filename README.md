# scen_workflow_afr

## Purpose

This code produces climate scenarios using CORDEX NetCDF files and observations. It also performs the following tasks:
- automated bias adjustment and statistical downscaling when producing climate scenarios;
- download of reanalysis data;
- aggregation of hourly data to a daily frequency for reanalysis data;
- calculation of climate indices;
- calculation of statistics related to climate scenarios and indices;
- generation of time series and maps.

The current version of the script (v1.0.2, August 204h, 2020) performs these tasks at a station. Soon, it will also perform these tasks on a grid.

## Releases
### 1.0.0

This is the initial stable release.
At the moment, there is one 'sample' climate indicator.

### 1.0.1

The following functionality was added:
- calculation of statistics for climate scenarios to include user-defined quantiles.

### 1.0.2

The following functionality was added:
- generation of time series and maps for climate indicators.
