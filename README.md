# gemini_density_tec

Calculate straight-ray slant total electron content between every rocket and receiver at each available GEMINI output time. Outputs one CSV row per time/rocket/receiver, plus a JSON record of settings and frame names.

Uses [pygemini at commit 6168ba3](https://github.com/gemini3d/pygemini/tree/6168ba3cebdfd43690f1eecf30404cf482ab860b) for grid reading, geographic-to-magnetic and dipole transforms, and species conventions. SciPy interpolates electron density on the native grid. The HDF5 reader selects only the electron species, following pygemini's axis ordering, instead of loading all seven species.

## Selected GNEISS run

Source: [gneiss_20260210_35400_09_SD_UM_xa_gneiss](https://rcweb.dartmouth.edu/LynchK/Gemini3D/gneiss_20260210_35400_09_SD_UM_xa_gneiss/).
The timestamps in its files are incorrect. Per the supplied correction, the 12 science outputs represent **2026-02-10 10:24:00 through 10:24:55 UTC**, at five-second intervals. `frame_times.csv` explicitly maps each original filename to its corrected UTC. Results preserve `source_filename`; both model and trajectory times use the corrected time. Filtering with `--start`/`--end` also uses corrected time.

Inputs:

* Rockets 397 and 398: `36397_GPS_Time_Export_01.csv` and `36398_GPS_Time_Export_00.csv` in `/Users/anniepflaum/lab317/data/raw/rocket/trajectories/GNEISS`.
* All 11 receivers from `/Users/anniepflaum/lab317/data/raw/receiver-data/asi-mapping/receivers.csv`.

The preparation command parses day-of-year GPS UTC for 2026 and converts altitude from km to m. It preserves receiver acronyms and altitudes. The combined spacecraft ephemeris is not used. Both rocket tracks cover the full corrected window: 22 pairs per frame, 264 rows total.

## Setup and run

```sh
/Users/anniepflaum/lab317/.venv/bin/python -m pip install -e .
# Prepare normalized CSVs, corrected times, and input provenance:
python -m gemini_density_tec.gneiss
# Download/resume the grid and all science output files:
python -m gemini_density_tec.gneiss --download
```

The download is tens of GB; `.part` files are resumed and renamed only after successful transfer. Preparation has been run locally; simulation downloads have not.

```sh
gemini-density-tec data/gneiss_20260210_35400_09_SD_UM_xa_gneiss \
  --trajectories data/gneiss_20260210_35400_09_SD_UM_xa_gneiss/trajectories.csv \
  --receivers data/gneiss_20260210_35400_09_SD_UM_xa_gneiss/receivers.csv \
  --frame-times data/gneiss_20260210_35400_09_SD_UM_xa_gneiss/frame_times.csv \
  --step-m 500 --output results/gneiss_slant_tec.csv
```

Without installing this package, prefix module commands with `PYTHONPATH=src` and use `/Users/anniepflaum/lab317/.venv/bin/python`; the calculator module is `gemini_density_tec.cli`.

Required model files are `inputs/simsize.h5`, `inputs/simgrid.h5`, and timestamped output `.h5` files. Only coordinate vectors and small metric slices are read from the large grid. Each density frame is loaded once for all pairs. Data/results are ignored by Git.

## Inputs and time alignment

Trajectory CSV columns:

```csv
time_utc,rocket_id,lat_deg,lon_deg,alt_m
2026-02-10T09:49:00Z,rocket_A,66.5,-147.5,200000
```

Receiver CSV columns:

```csv
receiver_id,lat_deg,lon_deg,alt_m
receiver_A,66.5,-147.5,0
```

Files in `examples/` are **synthetic schema examples**, not measured GNEISS positions. Altitude is meters; angles are degrees. UTC timestamps must carry `Z` or an offset. Positions interpolate linearly in Earth-centered Cartesian coordinates at each frame time; no extrapolation occurs. Gaps longer than `--max-gap-seconds` (default 10) yield `missing_trajectory`, except at an exact trajectory sample.

For other runs, filenames supply model UTC unless `--frame-times` is provided. `--trajectory-offset-seconds N` remains available for an additional explicit offset; leave it at zero for this GNEISS run. Density is not interpolated in time.

## Geometry and interpretation

* TEC = integral of electron density (m^-3) over physical path length (m), divided by 10^16 to give TECU.
* Geometry uses GEMINI's spherical Earth radius of 6,370 km. Latitude and altitude are interpreted as spherical geographic coordinates. GPS WGS84/geodetic values are therefore an approximation if supplied directly; this implementation does not provide an ellipsoidal conversion. Confirm the trajectory datum before scientific use.
* Rays are straight in Earth-centered Cartesian space. Refraction is not modeled; Earth-intersecting rays are flagged `earth_blocked`.
* Three-dimensional dipole grids and fixed-origin magnetic UEN Cartesian grids are supported. Cartesian origins are recovered from stored theta/phi and native axes, avoiding recentering on each ray. Other grid geometries require an adapter.
* Density uses trilinear interpolation, then trapezoidal integration at spacing no greater than `--step-m`. Repeat at half the spacing to check convergence on real data.
* `stec_tecu` is populated only when the entire sampled path is in the model. `modeled_stec_tecu` integrates only segments with two valid endpoints. It never bridges missing intervals. `coverage_fraction` is the fraction of physical path length represented by those segments. Boundary segments are conservatively omitted, so their error depends on step size.
* The linked configuration starts near 80 km: ground receivers will normally produce **partial** coverage. Missing lower-ionosphere density is not silently set to zero, and modeled TEC must not be presented as a complete observed slant TEC.

## Validation and status

```sh
PYTHONPATH=src /Users/anniepflaum/lab317/.venv/bin/python -m unittest discover -s tests -v
```

Tests cover analytic constant/linear density integrals, partial coverage, Earth blockage, selective electron-species reading and HDF5 transposition, Cartesian origin recovery, and CSV processing across rockets with different time coverage. The pinned pygemini source is compatible with the shared NumPy 2 environment through this adapter (its stock interpolation helper uses removed `np.NaN`).

The actual multi-GB GNEISS grid and frames have not been downloaded or scientifically validated. Trajectory/receiver inputs and corrected times are prepared; production calculations await the simulation files. This is a local Git repository; no GitHub remote has been created.

## Line-of-sight density notebook

Open `line_of_sight.ipynb` and select `/Users/anniepflaum/lab317/.venv/bin/python` as the kernel, then Run All. Defaults to VEE → 397 at corrected 10:24:15 UTC. The final cell provides a time slider and receiver/rocket selectors. Density color limits stay fixed across time. Notebook dependencies are available through `pip install -e '.[notebook]'`.

The left panel shows electron density in the radial plane through the two endpoints, with the ray overlaid against altitude and ground arc distance. The right panel shows density and accumulated modeled TEC along that same ray. Gray marks unmodeled intervals. The default plot and sampled profile export to `results/vee_397_los_102415.png` and `results/vee_397_los_102415_samples.csv`. This uses the same spherical geometry and native-grid interpolation as the calculator, including its coordinate-convention limitations.

Timestamp correction: exclude `20260210_35340.000000.h5` (blank-run copy). `20260210_35345.000000.h5` is 10:24:00 UTC; `20260210_35400.000000.h5` is 10:24:55 UTC. The manifest selects the included frames and is automatically used by the calculator when present in the model directory. Recompute TEC after changing alignment because rocket positions also change.

## Plot TEC results with common axes

From the repository directory, run:

```sh
PYTHONPATH=src /Users/anniepflaum/lab317/.venv/bin/python -m gemini_density_tec.plot_tec
```

This reads `results/gneiss_slant_tec.csv` and writes `results/gneiss_slant_tec.png`: one panel per receiver, with consistent rocket colors. Every panel uses the same UTC limits and the same TECU limits, computed across the full CSV (zero to the maximum plus 5%). NaNs remain gaps, not zeros.

To fix the scale yourself and also export each pair separately:

```sh
PYTHONPATH=src /Users/anniepflaum/lab317/.venv/bin/python -m gemini_density_tec.plot_tec \
  results/gneiss_slant_tec.csv --ylim 0 14 --individual-dir results/tec_pairs
```

All individual plots share the overview's limits. Use `--output path.png` to change the overview destination. These plots show modeled TEC, not the unmodeled contribution outside the grid.
