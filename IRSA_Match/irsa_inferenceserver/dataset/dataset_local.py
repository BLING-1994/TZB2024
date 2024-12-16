# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:05:46 2020

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

def data_loader_worker(dataset, index_queue, data_queue):
    while True:

        index = index_queue.get()  # 从索引队列获取索引
        if index is None:
            break  # 如果接收到None，结束进程
        # 加载和处理数据
        sample = dataset[index]
        # 可以在这里添加数据预处理的代码
        data_queue.put(sample)  # 将数据放入数据队列


class MultiprocessDataLoader:
    def __init__(self, dataset, num_workers, timeout=300):
        self.dataset = dataset
        self.num_workers = num_workers if len(self.dataset) > num_workers else len(self.dataset)
        self.index_queue = multiprocessing.Manager().Queue()
        self.data_queue = multiprocessing.Manager().Queue(maxsize=self.num_workers*2)
        self.workers = []
        self.timeout=timeout
        for _ in range(num_workers):
            worker = Process(target=data_loader_worker, args=(dataset, self.index_queue, self.data_queue))
            worker.daemon = True
            worker.start()
            
            self.workers.append(worker)


    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        # 将索引发送到索引队列
        for i in range(len(self.dataset)):
            self.index_queue.put(i)
        for _ in self.workers:
            self.index_queue.put(None)
        return self

    def __next__(self):
        if self.data_queue.qsize()==0:
            for t in range(self.timeout*10):
                worksss = []
                worksillrun = False
                for w in self.workers:
                    worksss.append(w.is_alive())
                    if w.is_alive(): worksillrun = True
                if self.data_queue.qsize()>0: break
                if worksillrun==False: break
                time.sleep(0.1)
            if worksillrun==False and self.data_queue.qsize()==0:
                raise StopIteration
            print(worksillrun, self.data_queue.qsize())
        try:
            data = self.data_queue.get(timeout=self.timeout)
            return data
        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
            raise StopIteration

    def close(self):
        # 发送None以停止工作进程
        print('close ', self.index_queue.qsize())
        self.index_queue.empty()
        for _ in self.workers:
            self.index_queue.put(None)
        for worker in self.workers:
            worker.join()






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

class Bigdata_local(Dataset):
    def __init__(
            self,
            img_pathes: str or list or dict,
            im_baseindex = None, # 指定基准数据，不指定默认为高分辨率数据为基准数据
            resolution_norm: bool = True, # 分辨率是否强制统一
            cut_size=1280,
            edge_padding=128,
            overlap=0.1,
            ignore_value=0,
            max_batch=8,
            endpoint = '10.13.1.18:29000',
            AWS_SECRET_ACCESS_KEY = 'miniotuxiangshi202301102140',
            AWS_ACCESS_KEY_ID = 'tuxiangshiminio',
            area_limit = None,
            server_config=None,
            preprocess_function=None,
            droplast=False,
            
    ):
        if isinstance(img_pathes, list) and len(img_pathes)==1:
            img_pathes = img_pathes[0]

        self.droplast = droplast
        self.preprocess_function = preprocess_function
        self.server_config = server_config
        self.limit_area = None
        self.img_pathes = img_pathes
        self.max_batch = max_batch
        self.im_baseindex = im_baseindex
        if im_baseindex is not None:
            if isinstance(img_pathes, list): assert isinstance(im_baseindex, int)
            elif isinstance(img_pathes, dict):
                assert isinstance(im_baseindex, type(list(img_pathes.keys())[0]))
                assert im_baseindex in img_pathes
        self.endpoint = endpoint
        self.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY
        self.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID
        self.resolution_norm = resolution_norm
        self.inits3()
        
        print("========1111", self.img_pathes)
        self.get_datainfo(self.img_pathes)
        # print("========11112222", self.img_infos)
        self.setrunarea(area_limit)
        self.ignore_value = ignore_value
        self.cut_list = []
        self.cut_size = self.check_cutsize(cut_size)
        
        self.overlap = overlap
        
        self.edge_padding = edge_padding
        self.gencut_list()
        # self.cut_list = self.cut_list[:10]
        
        
        # self.dataset_read = None
    def inits3(self):
        gdal.SetConfigOption('AWS_HTTPS', 'NO')
        gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
        gdal.SetConfigOption('CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE', 'YES')
        if self.endpoint is not None:
            gdal.SetConfigOption('AWS_S3_ENDPOINT', self.endpoint)
            gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', self.AWS_SECRET_ACCESS_KEY)
            gdal.SetConfigOption('AWS_ACCESS_KEY_ID', self.AWS_ACCESS_KEY_ID)

            os.environ["AWS_ACCESS_KEY_ID"] = self.AWS_ACCESS_KEY_ID
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.AWS_SECRET_ACCESS_KEY
    

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
        print('self.limit_area', self.limit_area)
        

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

    def getbatchsize(self):
        if len(self.cut_list) < self.max_batch:
            return len(self.cut_list)
        return self.max_batch

    def gencut_list(self):
        if self.limit_area is None: return
        width_overlap = self.overlap
        height_overlap = self.overlap
        cut_width = self.cut_size
        cut_height = self.cut_size
        start_x, start_y, range_w, range_h = polygon2box(self.imbase_info['geotrans'], osr.SpatialReference(wkt=self.imbase_info['proj']), self.limit_area)
        print(start_x, start_y, range_w, range_h)
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
                endw = start_x + range_w - cut_width
                if self.droplast:
                    if ((width_list[-1] - endw)/float(cut_width))>0.2:
                        width_list.append(start_x + range_w - cut_width)
                else:
                    width_list.append(start_x + range_w - cut_width)
        
        
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
                endw = start_y + range_h - cut_height
                if self.droplast:
                    if ((height_list[-1] - endw)/float(cut_height))>0.2:
                        width_list.append(start_y + range_h - cut_height)
                else:
                    height_list.append(start_y + range_h - cut_height)
        
        # width_list = [6912]
        # height_list = [0]

        for im_x in width_list:
            for im_y in height_list:
                cut_box = imgrect2geobox(self.imbase_info['geotrans'], osr.SpatialReference(wkt=self.imbase_info['proj']), im_x, im_y, cut_width, cut_height)
                if self.limit_area.intersects(cut_box)==False: continue

                self.cut_list.append({'cut_box': cut_box,
                                      'limit_area': self.limit_area.intersection(cut_box).wkt, 
                                      'pad_box':imgrect2geobox(self.imbase_info['geotrans'], 
                                                               osr.SpatialReference(wkt=self.imbase_info['proj']),
                                                               im_x - self.edge_padding, 
                                                               im_y - self.edge_padding, 
                                                               cut_width + self.edge_padding*2, 
                                                               cut_height + self.edge_padding*2), 
                                        'pad_transform': gettransformer(self.imbase_info['geotrans'], 
                                                               osr.SpatialReference(wkt=self.imbase_info['proj']),
                                                               im_x - self.edge_padding, 
                                                               im_y - self.edge_padding, 
                                                               cut_width + self.edge_padding*2, 
                                                               cut_height + self.edge_padding*2),
                                       'transform': gettransformer(self.imbase_info['geotrans'], 
                                                               osr.SpatialReference(wkt=self.imbase_info['proj']),
                                                               im_x, 
                                                               im_y, 
                                                               cut_width, 
                                                               cut_height),
                                       'proj': self.imbase_info['proj'],
                                       'base_startpt':(im_x, im_y)})
        print("Cut size", len(self.cut_list))
    
    def __len__(self):
        return len(self.cut_list)
    
    def cutimgfrompoly(self, poly_info):
        base_img = self.cutbaseimg(*poly_info['base_startpt']).transpose(1,2,0)

        pad_box = poly_info['pad_box']
        if isinstance(self.img_infos, dict):
            outdata = dict()
            for key in self.img_infos.keys():
                if key==self.im_baseindex: 
                    outdata[key] = base_img
                    continue 
                if 'dataset' not in self.img_infos[key]:
                    self.img_infos[key]['dataset'] = rio.open(self.img_infos[key]['im_path'])
                src_osr = ogr.CreateGeometryFromWkt(pad_box.wkt)
                src_osr.Transform(self.img_infos[key]['coordTransform'])
                coor_poly = loads(src_osr.ExportToWkt())
                try:
                    out_img, out_transform = rio.mask.mask(self.img_infos[key]['dataset'], [coor_poly], crop=True, nodata=self.ignore_value, filled =True)
                except:
                    self.img_infos[key]['dataset'] = rio.open(self.img_infos[key]['im_path'])
                    out_img, out_transform = rio.mask.mask(self.img_infos[key]['dataset'], [coor_poly], crop=True, nodata=self.ignore_value, filled =True)
                out_img = paddingimg(coor_poly, out_img, out_transform.to_gdal(), osr.SpatialReference(wkt=self.img_infos[key]['proj'])).transpose(1,2,0)
                if self.resolution_norm and out_img.shape!=base_img.shape: out_img = cv2.resize(out_img, base_img.shape[:2], interpolation = cv2.INTER_LINEAR)
                outdata[key] = out_img
            return outdata
        elif isinstance(self.img_infos, list):
            outdata = []
            for index in range(len(self.img_infos)):
                if index==self.im_baseindex: 
                    outdata.append(base_img)
                    continue
                if 'dataset' not in self.img_infos[index]:
                    self.img_infos[index]['dataset'] = rio.open(self.img_infos[index]['im_path'])
                src_osr = ogr.CreateGeometryFromWkt(pad_box.wkt)
                src_osr.Transform(self.img_infos[index]['coordTransform'])
                coor_poly = loads(src_osr.ExportToWkt())
                try:
                    out_img, out_transform = rio.mask.mask(self.img_infos[index]['dataset'], [coor_poly], crop=True, nodata=self.ignore_value, filled =True)
                except:
                    self.img_infos[index]['dataset'] = rio.open(self.img_infos[index]['im_path'])
                    out_img, out_transform = rio.mask.mask(self.img_infos[index]['dataset'], [coor_poly], crop=True, nodata=self.ignore_value, filled =True)
               
                out_img = paddingimg(coor_poly, out_img, out_transform.to_gdal(), osr.SpatialReference(wkt=self.img_infos[index]['proj']))
                out_img = out_img.transpose(1,2,0)
                if self.resolution_norm and out_img.shape!=base_img.shape: out_img = cv2.resize(out_img, base_img.shape[:2], interpolation = cv2.INTER_LINEAR)
                outdata.append(out_img)
            return outdata
        elif self.img_infos is None:
            outdata = base_img
            return outdata
    
    def cutbaseimg(self, im_x, im_y):
        im_width = self.imbase_info['im_width']
        im_height = self.imbase_info['im_height']
        padding = [0, 0, 0, 0]
        cut_rect = [im_x - self.edge_padding,
                    im_y - self.edge_padding,
                    im_x + self.cut_size + self.edge_padding,
                    im_y + self.cut_size + self.edge_padding]
        if cut_rect[0] < 0:
            padding[0] = - cut_rect[0]
            cut_rect[0] = 0
        if cut_rect[1] < 0:
            padding[1] = - cut_rect[1]
            cut_rect[1] = 0
        if cut_rect[2] >= im_width:
            padding[2] = cut_rect[2] - im_width
            cut_rect[2] = im_width
        if cut_rect[3] >= im_height:
            padding[3] = cut_rect[3] - im_height
            cut_rect[3] = im_height

        cut_rect = [int(i) for i in cut_rect]
        padding = [int(i) for i in padding]
        if 'dataset' not in self.imbase_info:
            self.imbase_info['dataset'] = rio.open(self.imbase_info['im_path'])
        try:
            out_img = self.imbase_info['dataset'].read(window=((cut_rect[1], cut_rect[3]), (cut_rect[0], cut_rect[2])))
        except:
            self.imbase_info['dataset'] = rio.open(self.imbase_info['im_path'])
            out_img = self.imbase_info['dataset'].read(window=((cut_rect[1], cut_rect[3]), (cut_rect[0], cut_rect[2])))
        out_img = np.pad(out_img, ((0,0),(padding[1], padding[3]),(padding[0], padding[2])), constant_values=0)
        return out_img


    def __getitem__(self, i):

        poly_info = self.cut_list[i]

        with rio.Env(AWS_HTTPS='NO', GDAL_DISABLE_READDIR_ON_OPEN='YES', AWS_VIRTUAL_HOSTING=False, AWS_S3_ENDPOINT=self.endpoint):
            out_data = self.cutimgfrompoly(poly_info)
        x, y = poly_info['base_startpt']

        image = out_data.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(np.ascontiguousarray(image / 255.0)).float()
        # image = torch.from_numpy(np.ascontiguousarray(image)).int()
        sample = {'tensor':image, 'box':poly_info['cut_box']}
        return sample



def custom_collate_fn(batch):
    """
    Custom collate function to prevent conversion of 'biasxy' field.
    """
    # 默认 collate 的字段
    default_collated = torch.utils.data.dataloader.default_collate([
        {k: v for k, v in item.items() if k != "box"} for item in batch
    ])
    
    # 保留 'biasxy' 原样
    biasxy = [item["box"] for item in batch]
    
    # 合并结果
    default_collated["box"] = biasxy
    return default_collated


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







