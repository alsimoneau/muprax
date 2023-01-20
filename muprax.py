#!/usr/bin/env python3

import astropy.convolution
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.warp
import yaml


def transform(s_crs=4326, t_crs=4326):
    return pyproj.Transformer.from_crs(s_crs, t_crs, always_xy=True).transform


def estimate_utm_epsg(lon, lat):
    return pyproj.database.query_utm_crs_info(
        datum_name="WGS84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            np.min(lon), np.min(lat), np.max(lon), np.max(lat)
        ),
    )[0].code


def warp(src, epsg, resampling):
    crs = rasterio.CRS.from_epsg(epsg)
    T, w, h = rasterio.warp.calculate_default_transform(
        src.crs, crs, src.width, src.height, *src.bounds
    )
    data = src.read(1)
    data[data <= -9999] = np.nan
    dest = np.zeros((h, w))
    return rasterio.warp.reproject(
        data,
        dest,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=T,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )


def arr_equal(a, b):
    return len(a) == len(b) and np.all(a == b)


def main(points, raster, outfile, params):
    db = pd.read_csv(points)
    if not arr_equal(db.columns, ["ID", "Latitude", "Longitude"]):
        raise ValueError("Wrong point list format")

    lat = pd.to_numeric(db["Latitude"], errors="coerce")
    lon = pd.to_numeric(db["Longitude"], errors="coerce")
    mask = ~(np.isnan(lon) | np.isnan(lat))
    lat = lat[mask]
    lon = lon[mask]

    rst = rasterio.open(raster)

    epsg = rst.crs.to_epsg()
    x, y = transform(4326, epsg)(lon, lat)

    bounds = (
        (x > rst.bounds.left)
        & (x < rst.bounds.right)
        & (y > rst.bounds.bottom)
        & (y < rst.bounds.top)
    )
    lat = lat[bounds]
    lon = lon[bounds]

    epsg = estimate_utm_epsg(lon.mean(), lat.mean())
    x, y = transform(4326, epsg)(lon, lat)

    arr, T_warp = warp(rst, epsg, resampling=rasterio.enums.Resampling.average)

    n = int(np.ceil(params["dist"] / abs(T_warp.a)))
    Y, X = np.mgrid[-n : n + 1, -n : n + 1]

    r = np.sqrt(X * X + Y * Y)
    r[n, n] = params["min_dist"]
    kernel = np.power(r, -params["exp"])
    kernel[r * abs(T_warp.a) > params["dist"]] = 0

    arr_conv = astropy.convolution.convolve(arr, kernel, fill_value=np.nan)
    trans_arr = arr_conv[rasterio.transform.rowcol(T_warp, x, y)]

    out = pd.DataFrame()
    out["ID"] = db["ID"][mask][bounds]
    out["Latitude"] = lat
    out["Longitude"] = lon
    out["Interp"] = trans_arr
    out.to_csv(outfile, index=False)


if __name__ == "__main__":
    input_file = "muprax.in"
    with open(input_file.strip().strip("'").strip('"')) as f:
        p = yaml.safe_load(f)

    main(
        points=p["points_file"],
        raster=p["raster_file"],
        outfile=p["out_file"],
        params=p["params"],
    )
