import os
import sys
import json
import logging
import uuid
import threading
import numpy as np
import subprocess as sp
import psutil
import time
from osgeo import gdal
import osgeo.ogr as ogr
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely import wkt
from shapely.wkt import loads
import zipfile
import json
from osgeo import ogr, osr
import chardet
from rasterio.features import rasterize
import rasterio
from multiprocessing import Pool
from shapely import geometry as geo
from rasterio.crs import CRS
import zipfile

def search_files(dir_path):
    result = []
    file_list = os.listdir(dir_path)  # 获取当前文件夹下的所有文件
    for file_name in file_list:
        complete_file_name = os.path.join(dir_path, file_name)  # 获取包含路径的文件名
        if os.path.isdir(complete_file_name):  # 如果是文件夹
            result.extend(search_files(complete_file_name)) # 文件夹递归
        if os.path.isfile(complete_file_name):  # 文件名判断是否为文件
            result.append(complete_file_name)   # 添加文件路径到结果列表里
            print(complete_file_name)           # 输出找到的文件的路径
    return result






def zipdir(path, ziph):
   org_path = path
   save_dirname = os.path.basename(org_path)
   with zipfile.ZipFile(ziph, 'w', zipfile.ZIP_DEFLATED) as zipObj:
        for root, dirs, files in os.walk(path):
            for file in files:
                save_dir = root.replace(org_path, save_dirname)
                zipObj.write(os.path.join(root, file), os.path.join(save_dir, file))


def imgrect2geobox(trans, spatial_ref, imx, imy, imw, imh):
    # spatial_ref = osr.SpatialReference(wkt=self.img_infos[self.im_baseindex]['proj'])
    start_pt = pt2geo(trans, [imx, imy + imh]) 
    end_pt = pt2geo(trans, [imx + imw, imy]) 

    poly = geo.Polygon.from_bounds(xmin=start_pt[0],ymin=start_pt[1],xmax=end_pt[0],ymax=end_pt[1])
    return poly


def imgrect2geobox1(trans, box):
    # spatial_ref = osr.SpatialReference(wkt=self.img_infos[self.im_baseindex]['proj'])
    x0, y0, x1, y1 = box
    start_pt = pt2geo(trans, [x0, y1]) 
    end_pt = pt2geo(trans, [x1, y0]) 

    poly = geo.Polygon.from_bounds(xmin=start_pt[0],ymin=start_pt[1],xmax=end_pt[0],ymax=end_pt[1])
    return poly


def gettransformer(trans, spatial_ref, imx, imy, imw, imh):
    start_pt = pt2geo(trans, [imx, imy])
    transout = list(trans)

    transout[0] = start_pt[0]
    transout[3] = start_pt[1]
    return transout

def polygon2box(trans, polygon):
    
    rect = polygon.bounds
    x0, y1 = geo2pt(trans, rect[0], rect[1])
    x1, y0 = geo2pt(trans, rect[2], rect[3])
    return [x0, y0, x1-x0, y1-y0]

def paddingimg(pad_box, im, im_trans, spatial_ref):

    start_x, start_y, range_w, range_h  = polygon2box(im_trans, spatial_ref, pad_box)
    end_x = start_x + range_w
    end_y = start_y + range_h

    top_pad = -start_y if start_y<0 else 0
    bottom_pad =  end_y-im.shape[1] if end_y-im.shape[1]>0 else 0

    left_pad = -start_x if start_x<0 else 0
    right_pad = end_x-im.shape[2] if end_x-im.shape[2]>0 else 0
    out_data = np.pad(im, ((0,0),(top_pad, bottom_pad),(left_pad, right_pad)), constant_values=0)
    return out_data


def convertfilepath2utf8(path):
    detected_encoding = chardet.detect(path.encode())['encoding']
    decoded_path = path.encode().decode(detected_encoding)
    return decoded_path


def read_info(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    del dataset
    return im_proj, im_geotrans

def read_infoall(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    # print('im_proj', im_proj)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize
    del dataset
    if im_proj is None:
        im_proj = pyproj.CRS.from_epsg(4326)
    return im_proj, im_geotrans, im_width, im_height

def read_infoall_rasterio(filename):
    with rasterio.open(filename) as src:
        transform = src.transform
        width, height = src.width, src.height
        crs = src.crs
    return crs, transform, width, height


def read_info_thread(filename):
    pool = Pool(processes=1)
    res = pool.apply_async(read_infoall, (filename,))
    pool.close()
    pool.join()
    return  res.get()



def pt2geo(trans, pt, bias_x=0, bias_y=0):
    pt0 = pt[0] + bias_x
    pt1 = pt[1] + bias_y
    px = trans[0] + pt0 * trans[1] + pt1 * trans[2]
    py = trans[3] + pt0 * trans[4] + pt1 * trans[5]
    return [px, py]


def geo2pt(geotrans, XGeo, YGeo):

    offsets = gdal.ApplyGeoTransform(gdal.InvGeoTransform(geotrans), XGeo, YGeo)
    return map(int, offsets)

def convertshpref(inshp, outshp, refimgpath):

    im_proj, im_geotrans, im_width, im_height = read_infoall(refimgpath)

    # 打开原始矢量文件
    source_ds = ogr.Open(inshp)
    source_layer = source_ds.GetLayer()

    # 创建目标坐标系（这里以 WGS 84 作为示例）
    target_srs = osr.SpatialReference(wkt=im_proj) # im_proj# osr.SpatialReference()

    # 创建新的数据源以保存转换后的数据
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    driver = ogr.GetDriverByName('ESRI Shapefile')
    target_ds = driver.CreateDataSource(outshp)
    target_layer = target_ds.CreateLayer(source_layer.GetName(), target_srs, ogr.wkbMultiPolygon)

    fieldNames = []
    fieldTypes = dict()
    # 将字段从原始图层复制到目标图层
    source_layer_defn = source_layer.GetLayerDefn()
    for i in range(source_layer_defn.GetFieldCount()):
        field_defn = source_layer_defn.GetFieldDefn(i)
        target_layer.CreateField(field_defn)
        fieldName = field_defn.GetName()
        field_type = field_defn.GetType()
        fieldNames.append(fieldName)
        fieldTypes[fieldName] = field_type

    # 创建坐标转换对象
    coordTransform = osr.CoordinateTransformation(source_layer.GetSpatialRef(), target_srs)

    # 遍历原始图层中的要素，并进行坐标转换
    for feature in source_layer:
        transformed_feature = ogr.Feature(target_layer.GetLayerDefn())
        # transformed_feature.SetFrom(feature)

        for fieldName in fieldNames:
            field_value = feature.GetField(fieldName)
            if fieldTypes[fieldName] == ogr.OFTString:
                detected_encoding = chardet.detect(field_value.encode())['encoding']
                utf8_field_value = field_value.encode(detected_encoding).decode('utf-8')
                # print(utf8_field_value)
                transformed_feature.SetField(fieldName, utf8_field_value)
            else:
                transformed_feature.SetField(fieldName, field_value)


        geom = feature.GetGeometryRef()
        geom.Transform(coordTransform)
        transformed_feature.SetGeometry(geom)

        target_layer.CreateFeature(transformed_feature)
        transformed_feature = None

    # 清理
    source_ds = None
    target_ds = None
    cpg_file_path = outshp.replace('.shp', '.cpg')
    with open(cpg_file_path, 'w') as cpg_file:
        cpg_file.write('UTF-8')


def orgstr2utf8str(data):
    if data == ogr.OFTString:
        detected_encoding = chardet.detect(data.encode())['encoding']
        data = data.encode(detected_encoding).decode('utf-8')
    return data




def wxzfshp2features(shp_path):
    if os.path.isdir(shp_path):
        pathes = os.listdir(shp_path)
        for filename in pathes:
            if filename.split('.')[-1]=='shp':
                shp_path = os.path.join(shp_path, filename)
                break
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(shp_path, 0)
    layer = shp.GetLayer()

    fieldNames = []
    fieldTypes = dict()
    # 获取字段信息
    layerDefinition = layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        field_defn = layerDefinition.GetFieldDefn(i)
        fieldName = field_defn.GetName()
        field_type = field_defn.GetType()
        fieldNames.append(fieldName)
        fieldTypes[fieldName] = field_type

    polys = []
    for feature in layer:
        fe = dict()

        qsx_path = orgstr2utf8str(feature.GetField("QSX"))
        hsx_path = orgstr2utf8str(feature.GetField("HSX"))
        Output = orgstr2utf8str(feature.GetField("Output"))

        im_proj, im_geotrans, im_width, im_height = read_infoall(hsx_path)
        coordTransform = osr.CoordinateTransformation(layer.GetSpatialRef(), osr.SpatialReference(wkt=im_proj))      
        # print(geom.ExportToWkb())
        # 将几何数据转换为Shapely对象
        geom = feature.GetGeometryRef()
        geom.Transform(coordTransform)
        # print(geom.ExportToWkt())
        # print("------------")
        shapely_geom = Polygon(wkt.loads(geom.ExportToWkt()))
        geoj_str = shapely_geom.__geo_interface__
        fe['geojson'] = json.dumps(geoj_str)
        fe['QSX'] = qsx_path
        fe['HSX'] = hsx_path
        fe['Output'] = Output
        # print(fe)
        polys.append(fe)
    return polys



# 写文件，写成tiff
def write_img(filename, im_data, trans, proj):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(trans)
    dataset.SetProjection(proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def getfeatures(shp_path, selectfieldname='code'):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(shp_path, 0)
    layer = shp.GetLayer()

    # 获取字段信息
    layerDefinition = layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        fieldName = layerDefinition.GetFieldDefn(i).GetName()
        # print(fieldName)

    polys = dict() 
    # 读取第一个要素的几何数据
    dfdf = True
    for feature in layer:
        # feature = layer.GetNextFeature()
        attribute_value = feature.GetField(selectfieldname)

        # print(geom.ExportToWkb())
        # 将几何数据转换为Shapely对象
        geom = feature.GetGeometryRef()
        shapely_geom = Polygon(loads(geom.ExportToWkt()))
        if attribute_value not in polys:
             polys[attribute_value] = []
             # print(attribute_value)
        polys[attribute_value].append(shapely_geom)
    return polys


def feature2reasternp(features, geotrans, width, height):
    ss = np.zeros((height, width))
    if isinstance(features, dict):
        infeature = []
        for key,value in features.items():
            infeature.extend([(poly, int(key)) for poly in value])

    rasterize(
            infeature,
            out_shape=(height, width),
            transform=geotrans,
            fill=0,
            default_value=255,
            dtype='uint8',
            out=ss
        )
    return ss


def shp2rasternp(inputshp, refimg):
    convertshpref(inputshp, inputshp.replace('.shp', '_ref.shp'), refimg)
    fe = getfeatures(inputshp.replace('.shp', '_ref.shp'))
    im_crs, im_geotrans, im_width, im_height = read_infoall_rasterio(refimg)
    return feature2reasternp(fe, im_geotrans, im_width, im_height)


def shpdir2features(shp_path):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    shp_pathes = []
    if os.path.isdir(shp_path):
        pathes = os.listdir(shp_path)
        for filename in pathes:
            if filename.split('.')[-1]=='shp':
                sppath = os.path.join(shp_path, filename)
                shp_pathes.append(sppath)
    else:
        shp_pathes.append(shp_path)
    
    outpoly = []
    for shpp in shp_pathes:
        outpoly.extend(wxzfshp2features(shpp))
    return outpoly


def shp2features(shp_path, selectfieldname=None):
    if os.path.isdir(shp_path):
        pathes = os.listdir(shp_path)
        for filename in pathes:
            if filename.split('.')[-1]=='shp':
                shp_path = os.path.join(shp_path, filename)
                break
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(shp_path, 0)
    layer = shp.GetLayer()

    fieldNames = []
    fieldTypes = dict()
    # 获取字段信息
    layerDefinition = layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        field_defn = layerDefinition.GetFieldDefn(i)
        fieldName = field_defn.GetName()
        field_type = field_defn.GetType()
        fieldNames.append(fieldName)
        fieldTypes[fieldName] = field_type

    if selectfieldname is None:
        polys = []
    else:
        polys = dict() 
    
    
    for feature in layer:
        fe = dict()
        if selectfieldname is not None:
            field_value = feature.GetField(selectfieldname)
            fe[selectfieldname] = field_value
            if fieldTypes[selectfieldname] == ogr.OFTString:
                detected_encoding = chardet.detect(field_value.encode())['encoding']
                utf8_field_value = field_value.encode(detected_encoding).decode('utf-8')
                fe[selectfieldname] = utf8_field_value
            
        else:
            for fieldName in fieldNames:
                field_value = feature.GetField(fieldName)
                fe[fieldName] = field_value
                
                if fieldTypes[fieldName] == ogr.OFTString:
                    detected_encoding = chardet.detect(field_value.encode())['encoding']
                    detected_encoding='utf-8'
                    utf8_field_value = field_value.encode(detected_encoding).decode('utf-8')
                    fe[fieldName] = utf8_field_value
        
        # print(geom.ExportToWkb())
        # 将几何数据转换为Shapely对象
        geom = feature.GetGeometryRef()
        # print(geom.ExportToWkt())
        # print("------------")
        wktstr = geom.ExportToWkt()
        if 'MULTIPOLYGON' in wktstr:
            shapely_geom = MultiPolygon(wkt.loads(wktstr))
        elif 'POLYGON' in wktstr:
            shapely_geom = Polygon(wkt.loads(wktstr))  
        # geoj_str = shapely_geom.__geo_interface__
        # fe['geojson'] = json.dumps(shapely_geom)
        # print(fe)
        polys.append(shapely_geom)
    return polys


def isinter_geojson_imgp(geojson, impath):
    im_proj, im_geotrans, im_width, im_height = read_infoall(impath)
    img_extent = imgrect2geobox(im_geotrans, osr.SpatialReference(wkt=im_proj), 0,0,im_width,im_height)
    return geojson2features(geojson).intersects(img_extent)


def geojson2features(geojson):
    if isinstance(geojson, str):
        if geojson=='':
            return None
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


# def context2polygon(area_context):
#     if area_context is None: return
#     if isinstance(area_context, dict):
#         area_context = json.dumps(area_context)
#     # shp geojson文件
#     if os.path.exists(area_context):
#         hz = area_context.split('.')[-1].lower() 
#         if hz=='geojson':
#             with open(area_context, 'r') as f:   
#                 geometry = geojson2features(json.load(f))
#         elif hz=='shp':
#             geometry = shp2features(area_context)
#     else:
#         geometry = geojson2features(area_context)
#     return geometry

import geopandas as gpd


def context2polygon(area_context, to_proj=None):
    if area_context is None: return
    if isinstance(area_context, Polygon):
        return area_context
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
    geometry = dataform['geometry'][0]
    return geometry


if __name__ == "__main__":
    # context2polygon('/irsa/IRSA_Inferenceserver/Inference_v010/log_remoteserver/ttt.geojson')
    strdata = {'type': 'Polygon', 'coordinates': [[[121.48428748062132, 25.09050815822097], [121.51681085224138, 25.092661086469278], [121.50671960675277, 25.0777061859537], [121.48859393738911, 25.07619313331675], [121.48428748062132, 25.09050815822097]]]}
    res = context2polygon(strdata)
    print(res)