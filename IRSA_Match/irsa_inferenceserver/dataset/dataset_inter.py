import os
from .dataset_local import Bigdata_local
import json
from shapely.geometry import Polygon, MultiPolygon, shape
from multiprocessing import Pool
from rasterio.transform import from_origin
import numpy as np
import geopandas as gpd
from osgeo import gdal, osr
from rasterio.crs import CRS
from shapely.affinity import affine_transform
from rasterio.features import geometry_mask
from affine import Affine
import cv2
from ..geotools import *
import rasterio as rio


def geojson2features(geojson):
    if isinstance(geojson, str):
        data = json.loads(geojson)
    else:
        data = geojson
    if isinstance(data, list):
        geometry = []
        for da in data:
            geometry.append(shape(da))
    if 'coordinates' in data:
        geometry = shape(data)
    else:
    # 提取几何对象
        geometry = shape(data['features'][0]['geometry'])
    
    tgdf = gpd.GeoDataFrame(geometry=[geometry])
    tgdf.crs = "EPSG:4326"
    return tgdf




def context2polygon(area_context, to_proj=None):
    if area_context is None: return
    if isinstance(area_context, dict):
        area_context = json.dumps(area_context)
    # shp geojson文件
    if os.path.exists(area_context):
        dataform = gpd.read_file(area_context)
    else:
        dataform = geojson2features(area_context)
    
    if dataform is None:
        return None
    
    if to_proj is not None:
        dataform = dataform.to_crs(CRS.from_wkt(to_proj))
    if len(dataform['geometry'])==0:
        return None
    return dataform


# 计算仿射变换矩阵
def calculate_affine_params(geo_transform):
    x_origin = geo_transform[0]
    x_pixel_size = geo_transform[1]
    x_rotation = geo_transform[2]
    y_origin = geo_transform[3]
    y_rotation = geo_transform[4]
    y_pixel_size = geo_transform[5]
    
    affine_matrix = [1 / x_pixel_size, -x_rotation / x_pixel_size, -y_rotation / y_pixel_size, 1 / y_pixel_size, -x_origin / x_pixel_size, -y_origin / y_pixel_size]
    return affine_matrix


def getlimitrange(x0, x1, w, minx, maxx, model=0):
    xr = abs(x1 - x0)
    if maxx - minx<w: w = maxx - minx
    if xr<w:
        centerx = (x1 + x0)/2.0
        x0 = centerx - w/2.0
        x1 = centerx + w/2.0
        if x0<minx: 
            x1 = x1 + abs(x0-minx)
            x0 = minx
        
        if x1>maxx:
            x0=x0- abs(x1-maxx)
            x1 = maxx
    return x0, x1
        



class Bigdata_inter(Bigdata_local):
    def __init__(self,
                img_pathes='',
                area_limit = None,
                **kwargs) -> None:
        
        self.img_pathes = img_pathes
        print(self.img_pathes)
        super().__init__(
            img_pathes=img_pathes,
            area_limit=area_limit,
            **kwargs)
        self.edge_padding = 0
        print(len(self.cut_list))

    def setrunarea(self, area_limit):

        # 获取图像本身区域
        self.imageinter_polygon = affine_transform(self.limit_area, calculate_affine_params(self.imbase_info['geotrans']))
        
        dataform = context2polygon(area_limit, self.imbase_info['proj'])
        assert len(dataform['geometry'])>0
        print(len(dataform['geometry']))
        # 转换坐标系
        if dataform is None:
            print(f"Error:  get geometry error, context is {area_limit}")
            return
        self.limit_area = dataform
        
        # todo 过滤小的


    def gencut_list(self):
        if self.limit_area is None: return
        width_overlap = self.overlap
        height_overlap = self.overlap
        cut_width = self.cut_size
        cut_height = self.cut_size
        limit_area_loacl = self.limit_area.copy(deep=True)
        limit_area_loacl['geometry'] = limit_area_loacl['geometry'].apply(lambda geom: affine_transform(geom, calculate_affine_params(self.imbase_info['geotrans'])))
        imbound = [round(x) for x in self.imageinter_polygon.bounds]
        print('img', imbound)
        # todo 根据不同模式切换  目前是根据cut_width 和 cut_height 以及和数据区域距离进行判别
        for row in limit_area_loacl.itertuples(index=True):
            if self.imageinter_polygon.intersects(row.geometry) is False: continue
            rgeom = row.geometry.intersection(self.imageinter_polygon)

            x0, y0, x1, y1 = rgeom.bounds
            if rgeom.area<500: continue
 
            # 
            x0, x1 = getlimitrange(x0, x1, cut_width, imbound[0], imbound[2], model=0)
            y0, y1 = getlimitrange(y0, y1, cut_height, imbound[1], imbound[3], model=0)
            
            # cw = round(abs(x1 - x0))
            # ch = round(abs(y1 - y0))
            

            # 切割图像
            # img = self.cutbaseimg(x0, y0)
            # print(img.shape)
            # 转换坐标
            cutgeom = affine_transform(rgeom, [1, 0, 0, 1, -x0, -y0])
            # shapes = [(cutgeom, 1)]
            # mask = geometry_mask(shapes, transform=Affine.identity(), invert=True, out_shape=(ch, cw)).astype('uint8')
            
            # cv2.imwrite('out.png', mask*255)
            # cv2.imwrite('img.png', np.transpose(img, (1,2,0)))
            cut_box = imgrect2geobox(self.imbase_info['geotrans'], osr.SpatialReference(wkt=self.imbase_info['proj']), x0, y0, cut_width, cut_height)
            self.cut_list.append({'cut_box': cut_box,
                                'limit_area': cut_box.wkt, 
                                'pad_box':imgrect2geobox(self.imbase_info['geotrans'], 
                                                        osr.SpatialReference(wkt=self.imbase_info['proj']),
                                                        x0 - self.edge_padding, 
                                                        y0 - self.edge_padding, 
                                                        cut_width + self.edge_padding*2, 
                                                        cut_height + self.edge_padding*2), 
                                'pad_transform': gettransformer(self.imbase_info['geotrans'], 
                                                        osr.SpatialReference(wkt=self.imbase_info['proj']),
                                                        x0 - self.edge_padding, 
                                                        y0 - self.edge_padding, 
                                                        cut_width + self.edge_padding*2, 
                                                        cut_height + self.edge_padding*2),
                                'transform': gettransformer(self.imbase_info['geotrans'], 
                                                        osr.SpatialReference(wkt=self.imbase_info['proj']),
                                                        x0, 
                                                        y0, 
                                                        cut_width, 
                                                        cut_height),
                                'mask': cutgeom, 
                                'proj': self.imbase_info['proj'],
                                'base_startpt':(x0, y0)})
        # print("Cut size", len(self.cut_list))
        print("Cut size", len(self.cut_list))
        
    def __getitem__(self, i):
        poly_info = self.cut_list[i]

        with rio.Env(AWS_HTTPS='NO', GDAL_DISABLE_READDIR_ON_OPEN='YES', AWS_VIRTUAL_HOSTING=False, AWS_S3_ENDPOINT=self.endpoint):
            out_data = self.cutimgfrompoly(poly_info)
        
        sample = dict(
            poly=poly_info,
            image=out_data,
            mask=poly_info['mask'],
            base_index= self.im_baseindex)
        if self.preprocess_function is not None:
            data = self.preprocess_function(sample, self.server_config)
            sample['processdata'] = data
        return sample



if __name__ == "__main__":
    import time
    dataset = Bigdata_inter(img_pathes = "",
                
                )
    
    for data in dataset:
        img = data['image']
        mask = data['mask']
        print(img.shape, mask)
        break