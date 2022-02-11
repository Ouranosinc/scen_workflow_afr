# scen_workflow_afr

## Purpose

This code produces climate scenarios using CORDEX NetCDF files and observations. It also performs the following tasks:
- download of reanalysis data;
- aggregation of hourly data to a daily frequency for reanalysis data;
- bias adjustment and statistical downscaling when producing climate scenarios (with error calculation) based on
  at-a-station observations or reanalysed data (ERA5 and ERA5-Land datasets are supported);
- calculation of climate indices;
- calculation of statistics related to climate scenarios and indices (min., max., mean or sum, centiles);
- generation of time series;
- generation of maps;
- multiprocesssing (per simulation).

## Notes

The latest version of the code requires the following project:
- https://github.com/yrouranos/dashboard

- This project, which is essentially a plot generator, must be downloaded within the scen_workflow_afr directory/project. 

Technical documentation can be found [here](https://ouranos-my.sharepoint.com/:f:/g/personal/yanrou1_ouranos_ca/EknV5GO46cxChVpilQwKzMQBrB3wu4e6aS3bUfoUlZ3gwg?e=R1Ju2C).

## Releases

### v1.0.0

This is the initial stable release.
At the moment, there is one 'sample' climate indicator.

### v1.0.1

Implemented features:
- calculation of statistics for climate scenarios to include user-defined centiles.

### v1.0.2

Implemented features:
- generation of time series for climate variables (was done earlier for climate indices);
- determination of minimum and maximum values in plots of time series (scenarios and indices).

### v1.0.3

Implemented features:
- storage of script parameters in a .ini file (instead of launch.py);
- integration of parameters d_exec, d_in1 and d_in2 in .ini file.

### v1.0.4

Implemented features:
- a technical documentation describing the Python code;
- conversion of precipitation values (from kg m-2 s-1 to mm) in workflow and postprocess figures (time series).
- correction of an anomaly during the calculation of statistics of precipitation. 

### v1.1.0

Implemented features:
- compatibility with gridded data (not only at a station);
- generation of maps for climate variables (interpolation);
- implementation of parallel processing for computationally expensive steps;
- clipping of scenario and index results if region boundaries are specified (in a GEOJSON file);
- calculation of statistics based on region boundaries (GEOJSON file) if reanalysis data is provided;
- conversion of NetCDF files to CSV files (only if the analysis is based on observations).

### v1.1.1

Implemented features:
- conversion of heat maps and time series to CSV;
- heat maps of wind variables now use a blue-white-red color bar with white corresponding to zero.

Other change made:
- relocated heat maps under directory stn/fig/scen/maps. 

### v1.1.2

Implemented features:
- possibility to enable/disable plot generation independently in scenarios and indices;
- improved description during logging;
- a few options differentiated between scenarios and indices tasks;
- created 'scen' and 'idx' directories under 'stat';
- added an aggregation option that will allow examining wind gusts;
- created several climate indices: etr, tx90p, heatwavemaxlen, heatwavetotlen, hotspellfreq, hotspellmaxlen,
  tgg, tng, tnx, txdaysabove, txx, tropicalnights, wsdi, rx1day, rx5day, cdd, cwd, drydays, prcptot, r10mm, r20mm,
  rainstart, rainend, raindur, rnnmm, sdii, wetdays, wgdaysabove, dc;
- added an option to export in georeferenced maps (GeoTIFF files).

### v1.2.0

Implemented features:
- created several climate indices: txg, tngmonthsbelow, tndaysbelow, drydurtot ('1d' and 'tot' calculation modes);
- added the possibility to specify a day-of-year interval for indices 'txdaysabove', 'tndaysbelow', 'wgdaysabove' and 'prcptot' 
- applying a mask on generated indices to remove values if located outside ERA5Land domain;
- added the possibility to add a line corresponding to the 90th percentile in time series (for the index 'prcptot');
- enabled parallel processing during the calculation of climate indices;
- enabled dependency between consecutive rain seasons; 
- added a second calculation method for 'rainend' index ('event'); the existing method was named 'depletion');
- an index is no longer calculated if it belongs to an exception list;
- now interpolating values to replace nan values in climate indices (especially for rain season);
- now calculating delta maps.

Bugs fixed:
- colour scale in maps (based on map content rather than statistics);
- legend in time series.

### v1.3.0

Implemented features:
- added the option 'opt_map_centiles' to generate maps for specific percentiles, in addition of the mean;
- added the option 'opt_bias_centiles' to specify centiles that are specific to bias adjustment plots;
- renamed the option 'stat_centiles' to 'opt_stat_centiles';
- added the option 'opt_map_delta' to enable/disable the generation of delta heat maps;
- made the 4th argument of index 'drydurtot' (minimum daily precipitation amount to consider) optional when the first
  argument is 'tot' (total mode);
- changed the color scale of dryness-related indices to tones of oranges.

### v1.3.1

Implemented features:
- added the possibility to group map colors by variable and period; for all statistics and scenarios. The previous
  grouping approach (by variable and statistic; for all scenarios and periods) is also working. The output is generated
  for both of these approaches.

### v1.3.2

Implemented features:
- ensured compatibility with at-a-station analysis (only for scenarios);
- fixed a bug where the precipitation values were n_days (ex: 10950 for 30 years) too small in statistic tables;
- improved daily and month plot generation function (applied to all periods and more robust).

### v1.3.3

Implemented features:
- adding missing lines in bias.csv file (containing bias adjustment error) automatically when a new simulation,
  station, or variable is added or if bias adjustment is performed with new nq, up_qmf and time_win parameters;
  the error is now calculated in scalar mode to avoid a competition between processes when updating bias.csv;
- changed the location of bias.csv so that there is one file per region.

Bugs fixed:
- fixed a typo in the name of CORDEX variables 'evspsbl' and 'evspsblpot';
- ensured compatibility of script with CORDEX variable 'evspsbl';
- fixed a bug that was happening when calculating statistics for scenarios (values must not be averaged over years);
- prevent duplicates in bias.csv.

### v1.3.4

Implemented features:
- created a rain season group of indices comprising rainstart, rainend, raindur and rainqty (now calculated at once);
- increased the calculation speed of rain season indices;
- now exporting bias adjustment and post-process plots if 'csv' in 'opt_diagnostic_format' (only mean values);
- standardized variable names (within the code).

Bugs fixed:
- reordering dimensions after calculating climate indices (required for drought code, or 'dc');
- the function utils.coord_names was returning a set (rather than an array), and the fact that the order was not
  always the same from one run to another had consequences on the subsequent analyses,
- added options 'opt_stat_clip' and 'opt_map_clip' to clip according to a polygon (default is False) when calculating 
  statistics or generating a map.

### v1.3.5

Implemented features:
- renamed the following indices: 'rainseason'->'rain_season', rainstart'->'rain_season_start',
  'rainend'->'rain_season_end', 'raindur'->'rain_season_length', 'rainqty'->'rain_season_prcptot', 'dc'->'drought_code',
  'hotspellfreq'->'hot_spell_frequency','hotspellmaxlen'->'hot_spell_max_length','heatwavemaxlen'->'heat_wave_max_len',
  'heatwavetotlen'->'heat_wave_total_len', 'wetdays'->'wet_days','drydays'->'dry_days',
  'txdaysabove' -> 'tx_days_above', 'wgdaysabove'->'wg_days_above','wxdaysabove'->'wx_days_above',
  'tropicalnights'->'tropical_nights', tngmonthsbelow'->'tng_months_below', 'drydurtot'->'dry_spell_total_length'; 
  modified ini files accordingly;
- added information about the climate indices parameters in config.py;
- now using a dayofyear string (ex: '04-14' for April 14th) instead of dayofyear (ex: 104 for April 14th) as the input
  to climate index functions (also applies to .ini files);
- adding function 'file_utils.clean_NetCDF' to discard potentially incomplete NetCDF files from the current exec
  directory (issue #9);
- adjusting file separator automatically according to d_exec parameter to ensure compatibility with Windows (issue #5);
- created a unit testing module that can be enabled using 'opt_test';
- increased the performance of 'rain_season*' and 'dry_spell_total_length' indices.

### v1.4.0 (under development)

Implemented features:
- relocated plotting functions within a distinct Github project (see note in the introduction);
- added compatibility with ENACTS reanalysis ensemble;
- restructured code to make it easier to maintain (object-oriented), which takes the form of a partial inheritance from
  the dashboard;
- regridding and interpolation is now performed with xesmf, which considerably reduces the duration of the analysis;
- parallel processing per variable during the generation of statistics and visual elements;
- cluster analysis in n dimensions; a plot is produced if two variables are provided by the used;
- fragmentation of former configuration file into a context (parameters of the analysis that depend on user
  input) and a se of constants (constant.py) (parameters that are constant throughout the analysis);
- grouping of figures under the 'fig' directory with improved directory access from the context;
- data shown on plots always saved as .csv files; an attempt is made to load these files to regenerate plots;
- improved aesthetics of visual elements (using matplotlib, hvplot, altair and plotly, depending on the context);
- now saving NetCDF files with the same coordinate names ('longitude' and 'latitude') and eliminating unnecssary
  variables to reduce the volume of files generated during the analysis;
- removed the option to optimize the selection of quantile mapping parameters;
- the lower and upper boundaries of RCP grops in time-series is now defined using centiles (instead of min-max).

## Contributing

This is a private development that is being used in production by climate services specialists. If you're interested in
being involved in the development, want to suggest features or report bugs, please leave us a message on the
[issue tracker](https://github.com/Ouranosinc/scen_workflow_afr/issues).