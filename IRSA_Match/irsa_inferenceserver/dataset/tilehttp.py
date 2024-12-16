import os
import math
import geopandas as gpd
import rasterio.crs
from shapely.geometry import Polygon
# from utile_3857 import *
from osgeo import gdal, osr
import cv2
import requests as req
import tqdm
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, shape
from multiprocessing import Pool
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.transform import from_bounds
import rasterio
from rasterio.crs import CRS
from shapely.affinity import affine_transform

def gdal_to_rasterio_params(gdal_params, width, height):
    # 获取GDAL的6个仿射变换参数
    a, b, c, d, e, f = gdal_params
    
    # 计算左上角像素的坐标
    xoff = a
    yoff = b
    
    # 计算x和y方向上的像素分辨率
    xscale = c
    yscale = e
    
    # 计算旋转和错切参数
    skew_x = -b / height * xscale
    skew_y = c / width * yscale
    
    return (xscale, skew_x, xoff, skew_y, yscale, yoff)


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

def context2polygon(area_context):
    if area_context is None: return
    if isinstance(area_context, dict):
        area_context = json.dumps(area_context)
    # shp geojson文件
    if os.path.exists(area_context):
        dataform = gpd.read_file(area_context)
  
    else:
        dataform = geojson2features(area_context)

    dataform = dataform.to_crs("EPSG:4326")

    return dataform


def convertCoordinatesToTile(lon, lat, zoom):
    xtile = int(math.floor((lon + 180) / 360 * (1 << zoom)))
    ytile = int(math.floor((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * (1 << zoom)))

    if xtile < 0:
        xtile = 0
    if xtile >= (1 << zoom):
        xtile = (1 << zoom) - 1
    if ytile < 0:
        ytile = 0
    if ytile >= (1 << zoom):
        ytile = (1 << zoom) - 1
    return xtile, ytile



def downloadImg(indata, url):
    tile = indata.get('tile')
    savepath = indata.get('savepath')
    zoom = indata.get('zoom')
    jlkeydlg = '2b31a6d0dbc74ed6704191d227c75d0b'
    jlkeydyp = '031d396812c5e24eabc56ebb2e8296c8'
    tdtkey='0cd15985b5426316c00a7c22f8aacd67'
    
    sp = os.path.join(savepath, f'{tile}.jpg')
    if os.path.exists(sp):
        return True
    x,y,z = [int(x) for x in tile.split('_')]
    # baseUrl = f'http://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={key}'
    baseUrl = f'https://api.jl1mall.com/getMap/{z}/{x}/{ 2 ** z - y -1}?mk=2d9bf902749f1630bc25fc720ba7c29f&tk={jlkeydlg}'

    if url is not None:
        baseUrl = url.format(x=x, y=y, z=z)
        print("@@", baseUrl)

    headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.47",
    'Cookie': "HWWAFSESID=44a6d8b1b8eb7d6335; HWWAFSESTIME=1660715969030; TDTSESID=rBACBGL8g8RStC+/gJUJAg==",
    }
    
    try:
        res = req.get(baseUrl, headers=headers)
    except:
        print("Error ")
        return False

    if res.status_code == 200:
        f = open(sp, 'wb')
        f.write(res.content)
        f.close()
        return True
    else:
        print("Error  ", res.status_code)
        return False


class Httpdata_3857:
    def __init__(self, rangegdf, zoom, save_path, model='JL', threadcount=2, url=None):
        self.model = model
        assert self.model in ['JL', 'TDT']
        if url is not None:
            self.model = "Tile"
        self.zoom = zoom
        self.save_path = save_path
        self.range = range
        self.threadcount = threadcount
        self.rangegdf = rangegdf
        self.url = url
        self.culdata()
    
    def process(self):
        tempsave_path = os.path.join(self.save_path, 'Temp', self.model, str(self.zoom))
        os.makedirs(tempsave_path, exist_ok=True)
       
        # gdf = context2polygon(self.range)
        self.rangegdf = self.rangegdf.to_crs("EPSG:4326")
        bounds = self.rangegdf.total_bounds

        # todo 多边形支持
            # 获取矩形相交切片
        alllist, iminfo = self.bount2tiles_3857(bounds, self.rangegdf)

        # tile, savepath, zoom,
        processlist = [{'tile': tileinfo['id'], 'savepath': tempsave_path, 'zoom': self.zoom} for tileinfo in tqdm.tqdm(alllist[self.zoom])]
        
        for data in processlist:
            downloadImg(data, self.url)
        
        # # # 下载数据  并进行缓存
        # with Pool(self.threadcount) as p:
        #     p.map(downloadImg, processlist)

        picoutdata = np.zeros((iminfo['height'], iminfo['width'], 3), 'uint8')
        # dataset.SetProjection(save_proj)
        for tileinfo in tqdm.tqdm(alllist[self.zoom]):
            nameid = tileinfo['id']
            if os.path.exists(os.path.join(tempsave_path, f'{nameid}.jpg')) == False: continue
            
            img = cv2.imread(os.path.join(tempsave_path, f'{nameid}.jpg'))
            if img is None: continue
            img = img[:,:,::-1]
            tileindex = tileinfo['index']
            picoutdata[tileindex[1]*256:tileindex[1]*256+256, tileindex[0]*256:tileindex[0]*256+256] = img
        padbound = iminfo['padbox']
        picoutdata = picoutdata[padbound[1]:padbound[3], padbound[0]:padbound[2]]
        return picoutdata

        # # 聚合数据  保存在内存 或者保存为文件
        # driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        # dataset = driver.Create(os.path.join(self.save_path, self.model + '_' + iminfo['name'] + '.tif'), iminfo['width'], iminfo['height'], 3, gdal.GDT_Byte,  options=['COMPRESS=LZW', "BIGTIFF=YES"])
        # dataset.SetProjection(iminfo['proj'])
        # dataset.SetGeoTransform(iminfo['trans'])

        # # dataset.SetProjection(save_proj)
        # for tileinfo in tqdm.tqdm(alllist[self.zoom]):
        #     nameid = tileinfo['id']
        #     if os.path.exists(os.path.join(tempsave_path, f'{nameid}.jpg')) == False: continue
        #     img = cv2.imread(os.path.join(tempsave_path, f'{nameid}.jpg'))[:,:,::-1]
        #     tileindex = tileinfo['index']
        #     for i in range(3):
        #         dataset.GetRasterBand(i+1).WriteArray(img[:,:,i], tileindex[0]*256, tileindex[1]*256)
    def down2file(self, savepath):
        picoutdata, iminfo = self.process()

        # 聚合数据  保存在内存 或者保存为文件
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(savepath, iminfo['width'], iminfo['height'], 3, gdal.GDT_Byte,  options=['COMPRESS=LZW', "BIGTIFF=YES"])
        dataset.SetProjection(iminfo['proj'])
        dataset.SetGeoTransform(iminfo['trans'])
        for i in range(3):
            dataset.GetRasterBand(i+1).WriteArray(picoutdata[:,:,i])

    def culdata(self):
        self.start = 20037508.342789244
        # 每个瓦片的边长（以米为单位)
        alllen = 40075016.68557849
        scale = 2 ** self.zoom
        self.tilelen = alllen/scale
        self.pixdistance = self.tilelen/256

    def tile_to_bounds(self, x, y):
        return x*self.tilelen - self.start, self.start - (y+1)*self.tilelen, (x+1)*self.tilelen - self.start, self.start - y*self.tilelen

    def create_tile_polygon(self, x, y):
        xmin, ymin, xmax, ymax = self.tile_to_bounds(x, y)
        # 创建 Polygon
        polygon = Polygon([
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin)
        ])
        return polygon

    def alltile_to_bounds(self, xmin, xmax, ymin, ymax):
        w_len = self.tilelen*(xmax + 1 - xmin)
        y_len = self.tilelen*(ymax + 1 - ymin)
        return xmin*self.tilelen - self.start, self.start - ymin*self.tilelen, w_len, y_len

    def bount2tiles_3857(self, bounds, gdf):
        dataform = gdf.to_crs("EPSG:3857")
        assert len(dataform['geometry'])>0
        bigpolygon = dataform['geometry'][0]

        xtile_min, ytile_min = convertCoordinatesToTile(bounds[0], bounds[3], self.zoom)
        xtile_max, ytile_max = convertCoordinatesToTile(bounds[2], bounds[1], self.zoom)  
        alllist = {self.zoom:[]} 
        width = (xtile_max+1 - xtile_min)*256
        height = (ytile_max+1 - ytile_min)*256
        save_name = f'z_{self.zoom}_x_{xtile_min}_{xtile_max}_y_{ytile_min}_{ytile_max}'
        xmin, ymax, x_len, y_len = self.alltile_to_bounds(xtile_min, xtile_max, ytile_min, ytile_max)
        trans = (xmin, x_len/width, 0, ymax, 0, -y_len/height)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)  # 例如，使用 EPSG:3857 投影
        for index_x, x in enumerate(tqdm.tqdm(range(xtile_min, xtile_max+1))):
            for index_y, y in enumerate(range(ytile_min, ytile_max+1)):
                tile_polygon = self.create_tile_polygon(x, y)
                if bigpolygon.intersects(tile_polygon):
                    alllist[self.zoom].append({'id':f'{x}_{y}_{self.zoom}', 'index':(index_x, index_y), 'geometry':tile_polygon})
        a, b, c, d, e, f = trans
        bigpolygon = affine_transform(bigpolygon, [1, 0, 0, 1, -a, -d])
        bigpolygon = affine_transform(bigpolygon, [1/b, 0, 0, 1.0/f, 0, 0])
        bbound = [int(np.round(x)) for x in bigpolygon.bounds]
        iminfo = {'width':width, 'height':height, 'trans':trans, 'padbox':bbound,  'name':save_name}
        return alllist, iminfo


if __name__ == "__main__":

    Httpdata_3857('/irsa/road/HLJ_DATA/HLJ/range/佳木斯市.shp', 
                17, 
                '/irsa/road/HLJ_DATA/HLJ', 
                model='JL',
                threadcount=20)










