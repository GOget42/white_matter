#!/usr/bin/env python3
"""
extract_historical_snowdepth.py

Extract snow depth data from historical GeoTIFFs with filenames like:
  snd_LImon_MODEL_historical_r1i1p1f1_grYYYYMM.tif

Saves a combined NetCDF with time, lat, lon, and metadata coords.

Requirements:
    pip install rasterio xarray netCDF4 tqdm
"""

import os
import re
import glob
import datetime
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from tqdm import tqdm

# Regex for parsing file names
FNAME_RE = re.compile(
    r'^snd_[^_]+_'                           # ignore 'snd_LImon_'
    r'(?P<model>[^_]+)_'                     # model name
    r'(?P<experiment>historical)_'           # fixed experiment
    r'(?P<realization>r\d+i\d+p\d+f\d+)_'
    r'gr(?P<yyyymm>\d{6})\.tif$'
)

def parse_filename(fname):
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename not in expected format: {fname}")
    parts = m.groupdict()
    yyyymm = parts.pop("yyyymm")
    parts["time"] = datetime.datetime(int(yyyymm[:4]), int(yyyymm[4:]), 1)
    return parts

def load_and_clip(tif_path, bbox):
    meta = parse_filename(os.path.basename(tif_path))
    with rasterio.open(tif_path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        arr2d = src.read(1, window=window)
        tr = src.window_transform(window)
        nrows, ncols = arr2d.shape
        xs = [(tr * (col, 0))[0] for col in range(ncols)]
        ys = [(tr * (0, row))[1] for row in range(nrows)]

    da = xr.DataArray(
        arr2d[None, :, :],  # expand to 3D (time, lat, lon)
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [meta["time"]],
            "latitude": ys,
            "longitude": xs,
            "model": meta["model"],
            "experiment": meta["experiment"],
            "realization": meta["realization"]
        },
        name="snow_depth"
    )
    return da

def extract_and_combine(input_dir, output_file, bbox):
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")

    arrays = []
    for tif in tqdm(tif_files, desc="Processing historical GeoTIFFs", unit="file"):
        da = load_and_clip(tif, bbox)
        arrays.append(da)

    ds = xr.concat(arrays, dim="time")
    ds.to_netcdf(output_file)
    print(f"\n✅ Saved historical NetCDF to {output_file}")

if __name__ == "__main__":
    # Input directory with historical GeoTIFFs
    input_dir = "IPSL-CM6A-LR_historial"

    # Output file name
    output_nc = "historical_snowdepth_IPSL_CM6A_LR.nc"

    # Laax region bounding box (lon_min, lat_min, lon_max, lat_max)
    bbox = (9.1506, 46.8015, 9.2876, 46.8827)

    extract_and_combine(input_dir, output_nc, bbox)
