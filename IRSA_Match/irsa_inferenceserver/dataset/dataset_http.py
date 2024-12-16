import os
from .dataset_local import Bigdata_local
import json
from shapely.geometry import Polygon, MultiPolygon, shape
from multiprocessing import Pool
from rasterio.transform import from_origin
import numpy as np
import geopandas as gpd
from osgeo import gdal, osr
from .tilehttp import Httpdata_3857

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

    dataform = dataform.to_crs("EPSG:3857")

    return dataform




class Bigdata_http(Bigdata_local):
    def __init__(self,
                img_pathes='',
                area_limit = None,
                **kwargs) -> None:
        
        self.zoom = 18
        self.img_pathes = img_pathes
        print(self.img_pathes)
        super().__init__(
            img_pathes=img_pathes,
            area_limit=area_limit,
            **kwargs)
        print(self.cut_list[0])
        print(len(self.cut_list))

    def setrunarea(self, area_limit):
        dataform = context2polygon(area_limit)
        assert len(dataform['geometry'])>0
        self.limit_area = dataform['geometry'][0]
        self.limit_bound = dataform.total_bounds
        alllen = 40075016.68557849
        scale = 2 ** self.zoom
        tilelen = alllen/scale
        pixdistance = tilelen/256
        im_width = np.round((self.limit_bound[2] - self.limit_bound[0])/pixdistance)
        im_height = np.round((self.limit_bound[3] - self.limit_bound[1])/pixdistance)
        im_geotrans  = from_origin(self.limit_bound[0], self.limit_bound[3], pixdistance, pixdistance).to_gdal()
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        self.imbase_info = {'proj':srs.ExportToWkt(), 'geotrans':im_geotrans, 'im_width':im_width, 'im_height':im_height, 'im_path':self.img_pathes}


    def get_datainfo(self, img_pathes):
        pass

    def __getitem__(self, i):
        poly_info = self.cut_list[i]
        rangegdf = gpd.GeoDataFrame(geometry=[poly_info['pad_box']])
        rangegdf.crs = "EPSG:3857"

        da = Httpdata_3857(rangegdf, self.zoom, 'Temp', url=self.img_pathes)
        out_data = da.process()

        sample = dict(
            poly=poly_info,
            image=out_data,
            base_index= None)
        if self.preprocess_function is not None:
            data = self.preprocess_function(sample, self.server_config)
            sample['processdata'] = data
        return sample



if __name__ == "__main__":
    import time
    dataset = Bigdata_http(img_pathes = "",
                area_limit={'type': 'Polygon', 'coordinates': [[[121.4978984528239, 25.078924061699894], [121.50370189854341, 25.07942189342775], [121.5050102551826, 25.07018285215692], [121.48909271758839, 25.0696719147317], [121.48942115046582, 25.08107704766533], [121.49709986022016, 25.082456496576455], [121.4978984528239, 25.078924061699894]]]}, 
                )
    
    for data in dataset:
        print(data)
        break