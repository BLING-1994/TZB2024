# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:05:46 2020

@author: DYP
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:24:41 2020

@author: DYP
"""
import os
import numpy as np
from osgeo import gdal, osr
import osgeo.ogr as ogr
import multiprocessing
import cv2
from multiprocessing import Pool
from shapely import geometry as geo
from shapely.geometry import Polygon, box
from rasterio.mask import mask
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio import Affine
import json
from shapely.geometry import shape, Polygon, MultiPolygon, MultiLineString
import time
from multiprocessing import Process, Queue
from shapely.wkt import dumps, loads
from shapely import wkt
from ..geotools import *
import traceback
from torch.utils.data import Dataset
import torch
from rasterio.crs import CRS
import rasterio.features
from shapely.affinity import affine_transform
from shapely.validation import make_valid

def getimgrect(img_pathe, nodata=None):
    im_proj, im_geotrans, im_width, im_height = read_infoall(img_pathe)
    if nodata is None:
        img_poly = imgrect2geobox(im_geotrans, osr.SpatialReference(wkt=im_proj), 0, 0, im_width, im_height)
        return [img_poly], CRS.from_wkt(im_proj) 
    
    pysacle = 4
    dataset = gdal.Open(img_pathe)
    im_proj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    band1 = dataset.GetRasterBand(1).GetOverview(pysacle-1)
    overview_data = band1.ReadAsArray()
    out = np.where(overview_data==nodata, 0, 255).astype('uint8')
    a,b,c,d,e,f = im_geotrans
    b = b*(2**pysacle)
    f = f*(2**pysacle)
    felist = []
    maskShape = rasterio.features.shapes(out)
    for feature, value in maskShape:
        if value!=255: continue
        fe = shape(feature)
        fe = make_valid(fe)
        fe = fe.buffer(0).simplify(1)
        fe = affine_transform(fe, [b, c, e, f, a, d])
        felist.append(fe)
    return felist, CRS.from_wkt(im_proj) 
    

            


# 多输入对齐模式,尺寸，分辨率，空间范围。
# 不存在空间范围参考
# 1、不指定基准数据，按数据尺寸最小进行度量
# 2、指定基准数据，按基准数据切片

# 必须存在空间范围参考（6参数投影）
# 设定基准按照基准切，其他数据区域如果不存在交集则为空，因此注意避免数据交集出现问题
# 1、数据尺寸完全一致，不存在空间参考，直接按照第一张切
# 2、数据尺寸不同，分辨率相同，如果不存在基准数据，按照范围取交集，构建新基准数据
# 3、数据分辨率不同，如果指定基准数据，按照空间索引切片，默认不缩放数据
# 4、数据分辨率不同，未指定基准数据，以高分辨率数据为默认基准数据

class Bigdata_info(Dataset):
    def __init__(
            self,
            img_pathes: str or list or dict,
            cut_size=1280,
            edge_padding=128,
            overlap=0.1,
            ignore_value=0,
            area_limit = None,
            droplast=False,
            
    ):
        if isinstance(img_pathes, list) and len(img_pathes)==1:
            img_pathes = img_pathes[0]

        self.droplast = droplast

        self.limit_area = None
        self.img_pathes = img_pathes
        self.get_datainfo(self.img_pathes)
        self.setrunarea(area_limit)
        self.ignore_value = ignore_value
        self.cut_list = []
        self.cut_size = self.check_cutsize(cut_size)
        
        self.overlap = overlap
        
        self.edge_padding = edge_padding
        self.gencut_list()
        # self.cut_list = self.cut_list[:10]

    def updateinterarea(self, imtrans, improj, imw, imh):
        ss = osr.SpatialReference(wkt=improj)
        img_poly = imgrect2geobox(imtrans, osr.SpatialReference(wkt=improj), 0, 0, imw, imh)

        if self.limit_area is None: 
            self.limit_area = img_poly
            return
        if self.limit_area.intersects(img_poly):
            print("intersection ", self.limit_area.area, img_poly.area)
            self.limit_area = self.limit_area.intersection(img_poly)
        

    def cluinterarea(self):
        polys = []
        if isinstance(self.img_infos, dict):
            for key, imginfo in self.img_infos.items():
                img_poly = imgrect2geobox(imginfo['geotrans'], osr.SpatialReference(wkt=imginfo['proj']), 0, 0, imginfo['im_width'], imginfo['im_height'])
                if key != self.im_baseindex:
                    coordTransform = osr.CoordinateTransformation(osr.SpatialReference(wkt=imginfo['proj']), osr.SpatialReference(wkt=self.img_infos[self.im_baseindex]['proj']))
                    src_osr = ogr.CreateGeometryFromWkt(img_poly.wkt)
                    src_osr.Transform(coordTransform)
                    img_poly = loads(src_osr.ExportToWkt())
                polys.append(img_poly)       
                
        elif isinstance(self.img_infos, list):
            for index in range(len(self.img_infos)):
                imginfo = self.img_infos[index]
                img_poly = imgrect2geobox(imginfo['geotrans'], osr.SpatialReference(wkt=imginfo['proj']), 0, 0, imginfo['im_width'], imginfo['im_height'])
                if index != self.im_baseindex:
                    coordTransform = osr.CoordinateTransformation(osr.SpatialReference(wkt=imginfo['proj']), osr.SpatialReference(wkt=self.img_infos[self.im_baseindex]['proj']))
                    src_osr = ogr.CreateGeometryFromWkt(img_poly.wkt)
                    print("------------- coord ")
                    print(coordTransform)
                    print(imginfo['proj'])
                    print(self.img_infos[self.im_baseindex]['proj'])
                    src_osr.Transform(coordTransform)
                    img_poly = loads(src_osr.ExportToWkt())         
                polys.append(img_poly)
        
        for poly in polys:
            print('poly  ', poly)
            if self.limit_area is None: 
                self.limit_area = poly
                continue
            if self.limit_area.intersects(poly):
                print("intersection ", self.limit_area.area, poly.area)
                self.limit_area = self.limit_area.intersection(poly)
            else:
                self.limit_area = None
                print("Error: no inter area")
                break
        # print('self.limit_area', self.limit_area)
        

    def setrunarea(self, context):
        if context is None: return
        # print(context)
        
        geometry = context2polygon(context, self.imbase_info['proj'])
        # 转换坐标系
        if geometry is None:
            print(f"Error:  get geometry error, context is {context}")
            return


        if self.limit_area is None: 
            self.limit_area = geometry
            return
        if self.limit_area.intersects(geometry):
            self.limit_area = self.limit_area.intersection(geometry)
        else:
            self.limit_area = None
            print("input limit area error overlap with images")
    

    def get_datainfo(self, img_pathes):
        if isinstance(img_pathes, dict):
            self.img_infos = dict()
            min_resolution = 100000
            min_index = None
            for key, img_path in img_pathes.items():
                im_proj, im_geotrans, im_width, im_height = read_info_thread(img_path)
                # self.updateinterarea(im_geotrans, im_width, im_height)
                self.img_infos[key] = {'proj':im_proj, 'geotrans':im_geotrans, 'im_width':im_width, 'im_height':im_height, 'im_path':img_path}
                resu = abs(im_geotrans[1])
                if resu>0 and resu<min_resolution:
                    min_resolution = resu
                    min_index = key
            if self.im_baseindex is not None:
                self.imbase_info = self.img_infos[self.im_baseindex]
            else:  # 选取分辨率最高的
                self.im_baseindex = min_index
                self.imbase_info = self.img_infos[min_index]
            base_osr = osr.SpatialReference(wkt=self.imbase_info['proj'])
            for key, value in self.img_infos.items():
                if key == self.im_baseindex: continue
                self.img_infos[key]['coordTransform'] = osr.CoordinateTransformation(base_osr, osr.SpatialReference(wkt=self.img_infos[key]['proj']))

        elif isinstance(img_pathes, list):
            self.img_infos = []
            min_resolution = 100000
            min_index = None
            for index, img_path in enumerate(img_pathes):
                im_proj, im_geotrans, im_width, im_height = read_info_thread(img_path)
                print('im_geotrans', im_geotrans)
                # self.updateinterarea(im_geotrans, im_width, im_height)
                self.img_infos.append({'proj':im_proj, 'geotrans':im_geotrans, 'im_width':im_width, 'im_height':im_height, 'im_path':img_path})
                resu = abs(im_geotrans[1])
                if resu>0 and resu<min_resolution:
                    min_resolution = resu
                    min_index = index
            
            if self.im_baseindex is not None:
                self.imbase_info = self.img_infos[self.im_baseindex]
            else:  # 选取分辨率最高的
                self.im_baseindex = min_index
                self.imbase_info = self.img_infos[min_index]
            base_osr = osr.SpatialReference(wkt=self.imbase_info['proj'])
            for index in range(len(self.img_infos)):
                if index == self.im_baseindex: continue
                self.img_infos[index]['coordTransform'] = osr.CoordinateTransformation(base_osr, osr.SpatialReference(wkt=self.img_infos[index]['proj']))

        elif isinstance(img_pathes, str):
            im_proj, im_geotrans, im_width, im_height = read_info_thread(img_pathes)
            self.imbase_info = {'proj':im_proj, 'geotrans':im_geotrans, 'im_width':im_width, 'im_height':im_height, 'im_path':img_pathes}
            self.updateinterarea(im_geotrans, im_proj, im_width, im_height)
            self.img_infos = None  
            self.im_baseindex = None  
        self.cluinterarea()


    def check_cutsize(self, cut_size):
        im_h = (self.imbase_info['im_height'] // 128) * 128
        im_w = (self.imbase_info['im_width'] // 128) * 128

        if cut_size > im_h: cut_size = im_h
        if cut_size > im_w: cut_size = im_w
        return cut_size



    def gencut_list(self):
        if self.limit_area is None: return
        width_overlap = self.overlap
        height_overlap = self.overlap
        cut_width = self.cut_size
        cut_height = self.cut_size
        start_x, start_y, range_w, range_h = polygon2box(self.imbase_info['geotrans'], osr.SpatialReference(wkt=self.imbase_info['proj']), self.limit_area)
        # print(start_x, start_y, range_w, range_h)
        width_list = []
        if cut_width>range_w: 
            width_list = [start_x]
        else:
            move_xstep = int(cut_width * (1 - width_overlap))
            for index in range(start_x, start_x + range_w - cut_width, move_xstep):
                width_list.append(index)
            if len(width_list)>0:
                if start_x + range_w - cut_width != width_list[-1]:
                    width_list.append(start_x + range_w - cut_width)
            else:
                width_list.append(start_x + range_w - cut_width)
            if self.droplast and len(width_list)>1:
                if abs(width_list[-1] - width_list[-2])/float(cut_width)<0.2:
                    width_list = width_list[:-1]
        
        
        height_list = []
        if cut_height>range_h: 
            height_list = [start_y]
        else:
            move_ystep = int(cut_height * (1 - height_overlap))
            for index in range(start_y, start_y + range_h - cut_height, move_ystep):
                height_list.append(index)
            if len(height_list)>0:
                if start_y + range_h - cut_height != height_list[-1]:
                    height_list.append(start_y + range_h - cut_height)
            else:
                height_list.append(start_y + range_h - cut_height)
            if self.droplast and len(height_list)>1:
                if abs(height_list[-1] - height_list[-2])/float(cut_height)<0.2:
                    height_list = height_list[:-1]
                


        for im_x in width_list:
            for im_y in height_list:
                cut_box = imgrect2geobox(self.imbase_info['geotrans'], osr.SpatialReference(wkt=self.imbase_info['proj']), im_x, im_y, cut_width, cut_height)
                if self.limit_area.intersects(cut_box)==False: continue
                self.cut_list.append({'cut_box': cut_box, 'base_startpt':(im_x, im_y)})

        # print("Cut size", len(self.cut_list))
    
   



if __name__ == "__main__":
    import tqdm
    # root_path = '/vsis3/irsastudio/DOM_1_part.tif'
    # img_path = ['/irsa/data_tocog/cp/ces/before_errorref/by.tif', '/irsa/data_tocog/cp/ces/after/by.tif']
    # 写文件，写成tiff
    strrange = {'type': 'Polygon', 'coordinates': [[[121.48385513358122, 25.08529154165197], [121.51649147708495, 25.089830283905513], [121.51302226359802, 25.06655284644968], [121.4833411734861, 25.066203656464083], [121.48385513358122, 25.08529154165197]]]}


    img_path = [['/irsa/测试数据COG/台湾省淡水河/20180328.tif']]

    test_dataset = Bigdata_local(img_path,
                                cut_size=768,
                                edge_padding=128,
                                overlap=0.1,
                                ignore_value=0,
                                max_batch=1,
                                area_limit=None)
    
    # rs = MultiprocessDataLoader(test_dataset, 1)
    
    for data in tqdm.tqdm(test_dataset):
        # print(data['image'].shape, data['poly']['transform'], test_dataset.imbase_info['geotrans'])

        poly_info = data['poly']

        limit_area = wkt.loads(poly_info['limit_area'])
        print(limit_area)
        # limit_gdf = gpd.GeoDataFrame(geometry=[limit_area])
        # limit_gdf.crs = CRS.from_wkt(im_proj)
        # limit_gdf = limit_gdf.to_crs(epsg=3857)
        # for key, gdf in vec_dataes.items():
        #     intersected_data = gpd.overlay(gdf, limit_gdf, how='intersection')
        #     vec_dataes[key] = intersected_data
        # save_name = 'cut_' + str(startpt[0]) + '_' + str(startpt[1]) + '.png'
        # cv2.imwrite('/irsa/IRSA_Inferenceserver/Inference_v007/temp/in0_' + save_name, data0)

        # cv2.imwrite('/irsa/IRSA_Inferenceserver/Inference_v007/temp/in1_' + save_name, data1)


        # write_img('/irsa/IRSA_Inferenceserver/Inference_v007/temp' + save_name, data['image'].transpose(2,0,1), data['poly']['transform'], data['poly']['proj'])

        # st_x, st_y = data['poly']['base_startpt']
        # image = data['image'][:,:,::-1]
        
        # if np.sum(image[:,:,0]>0)==0:
        #     continue
        # savename = str(st_x) + '_' + str(st_y) + '.png'
        # image = cv2.resize(image, (512, 512))
        # cv2.imwrite('/irsa/data_tocog/hz/hz_cut/' +  savename, image)







