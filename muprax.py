#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.warp
import scipy.ndimage
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


def arr_equal(a, b):
    return len(a) == len(b) and np.all(a == b)


def main(points, raster, outfile, func, params):
    db = pd.read_csv(points)
    if not arr_equal(db.columns, ["ID", "Latitude", "Longitude"]):
        raise ValueError("Wrong point list format")

    lat = db["Latitude"]
    lon = db["Longitude"]

    rst = rasterio.open(raster)

    epsg = estimate_utm_epsg(lon.mean(), lat.mean())
    x, y = transform(4326, epsg)(lon, lat)

    if func == "nn":
        arr, T_warp = warp(
            rst, epsg, resampling=rasterio.enums.Resampling.nearest
        )

    elif func == "idw":
        arr, T_warp = warp(
            rst, epsg, resampling=rasterio.enums.Resampling.average
        )

        yc = int(params["dist"] // abs(T_warp.e))
        xc = int(params["dist"] // abs(T_warp.a))
        shape = (2 * yc + 1, 2 * xc + 1)
        Y, X = np.indices(shape, sparse=True)
        X -= xc
        Y -= yc
        kernel = np.power(X * X + Y * Y, -params["exp"] / 2)
        kernel[yc, xc] = 0

        arr = scipy.ndimage.convolve(arr, kernel, mode="constant")

    db["Interp"] = arr[rasterio.transform.rowcol(T_warp, x, y)]
    db.to_csv(outfile, index=False)


if __name__ == "__main__":
    with open("muprax.in") as f:
        p = yaml.safe_load(f)

    main(
        points=p["points_file"],
        raster=p["raster_file"],
        outfile=p["out_file"],
        func=p["mode"],
        params=p["params"],
    )
