import os
import numpy as np
from irsa_inferenceserver.dataset.dataset_info import Bigdata_info, getimgrect
from rasterio.crs import CRS
from osgeo import gdal, osr
import geopandas as gpd

import pandas as pd
from utile import *
import time
import tqdm


def overlayimg(img_dir, imlist, savepath=None):
    gdf = gpd.GeoDataFrame(geometry=[])
    basecrs = None
    outarea = {}
    for imname in imlist:
        print(imname)
        imgpolys, imgcrs = getimgrect(os.path.join(img_dir, imname), nodata=0)
        print('clu finish')
        
        if basecrs is None: basecrs = imgcrs
        for imgpoly in imgpolys:
            if remove_inter: 
                for existing_polygon in gdf.geometry:
                    imgpoly = imgpoly.difference(existing_polygon)
            if imgpoly.geom_type=='MultiPolygon':
                print('=== MultiPolygon', )
                outarea[imname] = list(imgpoly.geoms)
            else:
                outarea[imname] = [imgpoly]

            if not imgpoly.is_empty:
                gdf = pd.concat([gdf, gpd.GeoDataFrame(geometry=[imgpoly])], ignore_index=True)

    if savepath is not None:
        gdf.crs = basecrs
        gdf.to_file(os.path.join(save_path, f'range_part.shp'), driver='ESRI Shapefile', encoding='utf-8', engine='pyogrio')
    return outarea
    


if __name__ == '__main__':
    import sys
    cfg = load_config(sys.argv[1])

    workdir = cfg['workdir']
    pyscale = cfg['pyscale']# 5层金字塔
    droplast = cfg['buildbox-01']['droplast'] # 丢弃最后一层
    remove_inter = cfg['buildbox-01']['remove_inter']  # 去除重叠区域数据
    img_dir = cfg['img_dir']
    extension = cfg['extension']
    save_path = os.path.join(workdir, cfg['buildbox-01']['save_path'])

    os.makedirs(save_path, exist_ok=True)
    imlist = list_files_with_extension(img_dir, extension)
    # imlist = imlist[3:4]
    # print(imlist)
    limitareas = overlayimg(img_dir, imlist, os.path.join(workdir, cfg['buildbox-01']['save_path']) if cfg['buildbox-01']['save_rangepart'] else None)

    for nowpy in pyscale:
        print('pyarmia stage ', nowpy)
        meanfeature = []
        ids = []
        polygons = []
        feanames = []
        stx = []
        sty = []
        basecrs = None
        for imname in tqdm.tqdm(imlist):
            arealist = limitareas[imname]
            for al in arealist:
                dataset = Bigdata_info(os.path.join(img_dir, imname), 
                                        cut_size = cfg['cut_size']*(2 ** nowpy), 
                                        edge_padding = 0, 
                                        overlap = cfg['overlap'], 
                                        droplast=droplast,
                                        area_limit=al)
                sds = dataset.imbase_info
                if basecrs is None: basecrs = CRS.from_wkt(dataset.imbase_info['proj'])  
                # outname = os.path.splitext(imname)[0]
                for cutinfo in dataset.cut_list:
                    ids.append(len(ids))
                    polygons.append(cutinfo['cut_box'])
                    stx.append(cutinfo['base_startpt'][0])
                    sty.append(cutinfo['base_startpt'][1])
                    feanames.append(imname)
        # end 
        print('pyarmia stage ', nowpy, 'cutsize ', len(polygons))
        gdf = gpd.GeoDataFrame({'id':ids, 'feaname':feanames, 'stx':stx, 'sty':sty, 'geometry':polygons})
        gdf.crs = basecrs
        gdf.to_file(os.path.join(save_path, f'range_py{nowpy}.shp'), driver='ESRI Shapefile', encoding='utf-8', engine='pyogrio')

