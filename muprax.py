#!/usr/bin/env python3

import click
import joblib
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.warp
import scipy.ndimage


def parallelize(func):
    def wrapper(iterable, *args):
        return joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(func)(i, *args) for i in iterable
        )

    return wrapper


def transform(s_crs=4326, t_crs=4326):
    return pyproj.Transformer.from_crs(s_crs, t_crs, always_xy=True).transform


def estimate_utm_epsg(lon, lat):
    return pyproj.database.query_utm_crs_info(
        datum_name="WGS84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            min(lon), min(lat), max(lon), max(lat)
        ),
    )[0].code


def warp(src, crs, resampling):
    T, w, h = rasterio.warp.calculate_default_transform(
        src.crs, crs, src.width, src.height, *src.bounds
    )
    dest = np.zeros((h, w))
    return rasterio.warp.reproject(
        src.read(1),
        dest,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=T,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )


def main(points, raster, func, params):
    db = pd.read_csv(points)
    if db.columns != ["ID", "Latitude", "Longitude"]:
        raise ValueError("Wrong point list format")

    lat = db["Latitude"]
    lon = db["Longitude"]

    rst = rio.open(raster)

    epsg = estimate_utm_epsg(lon.mean(), lat.mean())
    y, x = transform(4326, epsg)(lat, lon)

    if func == "nn":
        arr, T_warp = warp(
            rst, epsg, resampling=rasterio.enums.Resampling.nearest
        )

    elif func == "idw":
        arr, T_warp = warp(
            rst, epsg, resampling=rasterio.enums.Resampling.average
        )

        yc = params["dist"] // abs(T_warp.e)
        xc = params["dist"] // abs(T_warp.a)
        shape = (2 * yc + 1, 2 * xc + 1)
        Y, X = np.indices(shape, sparse=True)
        kernel = np.power(X * X + Y * Y, params["exp"] / 2)

        arr = scipy.ndimage.convolve(arr, kernel, modemode="constant")

    db["Interp"] = arr[rasterio.transform.rowcol(T_warp, y, x)]


if __name__ == "__main__":
    main()
