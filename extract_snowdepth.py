#!/usr/bin/env python3
"""
extract_snowdepth.py

Extract snow depth data for a specified bounding box from GeoTIFFs
and save as a combined NetCDF, with a progress indicator.

Now also parses model, scenario and realization from filenames
and stores them as coordinates.

Requirements:
    pip install rasterio xarray netCDF4 tqdm
"""

import os
import glob
import re
import datetime

import xarray as xr
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm

# regex to grab model, scenario, realization, date
FNAME_RE = re.compile(
    r'^snd_[^_]+_'                  # snd_LImon_
    r'(?P<model>[^_]+)_'            # EC-Earth3-Veg-LR
    r'(?P<scenario>ssp\d+)_'
    r'(?P<realization>r\d+i\d+p\d+f\d+)_'
    r'gr(?P<yyyymm>\d{6})\.tif$'    # gr201501.tif
)

def parse_filename(fname):
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename not in expected format: {fname}")
    parts = m.groupdict()
    # parse time
    yyyymm = parts.pop("yyyymm")
    parts["time"] = datetime.datetime(int(yyyymm[:4]), int(yyyymm[4:]), 1)
    return parts  # dict with keys model, scenario, realization, time

def load_and_clip(tif_path, bbox):
    """Load one GeoTIFF, clip to bbox, return 3D DataArray (time,lat,lon)."""
    meta = parse_filename(os.path.basename(tif_path))
    with rasterio.open(tif_path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        arr2d = src.read(1, window=window)
        tr = src.window_transform(window)

        # build lat/lon coords
        nrows, ncols = arr2d.shape
        xs = [(tr * (col, 0))[0] for col in range(ncols)]
        ys = [(tr * (0, row))[1] for row in range(nrows)]

    # expand to 3D: time × lat × lon
    data3d = arr2d[None, :, :]

    da = xr.DataArray(
        data3d,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [meta["time"]],
            "latitude": ys,
            "longitude": xs,
            "model": meta["model"],
            "scenario": meta["scenario"],
            "realization": meta["realization"],
        },
        name="snow_depth"
    )
    return da

def extract_and_combine(input_dir, output_file, bbox):
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")

    arrays = []
    for tif in tqdm(tif_files, desc="Processing GeoTIFFs", unit="file"):
        da = load_and_clip(tif, bbox)
        arrays.append(da)

    # concatenate along time
    ds = xr.concat(arrays, dim="time")

    # ensure model/scenario/realization are stored per-time
    # they’re already coords on time dimension

    ds.to_netcdf(output_file)
    print(f"\n✅ Saved combined NetCDF to {output_file}")

if __name__ == "__main__":
    input_dir = "IPSL-CM6A-LR_future"
    output_nc = "snow_depth_prediction.nc"
    # Bounding box for Laax (lon_min, lat_min, lon_max, lat_max)
    bbox = (9.1506, 46.8015, 9.2876, 46.8827)

    extract_and_combine(input_dir, output_nc, bbox)
