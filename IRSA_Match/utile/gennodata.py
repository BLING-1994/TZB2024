import os
from osgeo import gdal
from utile import *
import numpy as np
import cv2
import rasterio.features
import geopandas as gpd
from shapely.affinity import affine_transform
from rasterio.crs import CRS
from shapely.geometry import shape
from irsa_inferenceserver.dataset.dataset_info import Bigdata_info, getimgrect


cfg = load_config('config/hebei.yaml')
img_dir = cfg['img_dir']
extension = cfg['extension']
imlist = list_files_with_extension(img_dir, extension)
savep = 'cs'
os.makedirs(savep, exist_ok=True)

nodata = 0
imlist = imlist[1:2]
for imname in imlist:
    pysacle = 5
    dataset = gdal.Open(os.path.join(img_dir, imname))
    felist, basecrs = getimgrect(os.path.join(img_dir, imname), 0)
    print(felist)
    tgdf = gpd.GeoDataFrame(geometry=felist)

    tgdf.crs = basecrs

    print(imname + '.shp')
    tgdf.to_file(os.path.join(savep, imname + '.shp'), driver='ESRI Shapefile', encoding='utf-8')



