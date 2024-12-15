from torch.utils.data import Dataset
from utile import *
import geopandas as gpd
import os
from osgeo import gdal
import numpy as np
import cv2
import torch
import tqdm
from loftr_dyp.loftr import LoFTR
from loftr_dyp.misc import lower_config
from loftr_dyp.config import get_cfg_defaults
from Dinov2.dino_extract import DINOExtract
import os
import numpy as np
from utile import *
from torch.utils.data import Dataset
from osgeo import gdal
from irsa_inferenceserver.geotools import imgrect2geobox1, read_info
import geopandas as gpd
from rasterio.crs import CRS
import shutil

import multiprocessing as mp  
import time



class Get_local(Dataset):
    def __init__(self, img_dir, shp_dir, nowscale):
        self.img_dir = img_dir
        self.nowscale = nowscale
        self.reuescount = 0
        self.gdf = gpd.read_file(shp_dir)
        self.filterexsit()
        self.dataset = None
        self.nowimg = None
    
    def filterexsit(self):
        self.runlist = []
        for index, row in tqdm.tqdm(self.gdf.iterrows()):
            self.runlist.append(index)
        print('run len ', len(self.runlist))
            
    def __len__(self):
        return len(self.runlist)

    def initdata(self, imname):
        if self.nowimg == imname: return
        del self.dataset
        self.dataset = gdal.Open(os.path.join(self.img_dir, imname))
        self.band_count = self.dataset.RasterCount
        self.nowimg = imname
        self.band_layers = []
        if self.nowscale>0:
            for bandindex in range(self.band_count):
                self.band_layers.append(self.dataset.GetRasterBand(bandindex+1).GetOverview(self.nowscale-1))
        else:
            for bandindex in range(self.band_count):
                self.band_layers.append(self.dataset.GetRasterBand(bandindex+1))



    def __getitem__(self, i):
        index = self.runlist[i]
        row = self.gdf.iloc[index]
        id = row['id']
        imname = row['feaname']
        stx = int(row['stx']/(2 ** self.nowscale))
        sty = int(row['sty']/(2 ** self.nowscale))
        self.initdata(imname)
        out = []

        for ly in self.band_layers:
            out.append(ly.ReadAsArray(stx, sty, 512, 512)[None])
        image = np.concatenate(out, 0)# .transpose(1,2,0)
        image = torch.from_numpy(np.ascontiguousarray(image / 255.0)).float()
        sample = {'image':image, 'id': id, 'imname': imname, 'stx': row['stx'], 'sty':row['sty']}
        return sample
   
            
def init_model(gimpath, device='cuda'):
    gimmodel = LoFTR(lower_config(get_cfg_defaults())['loftr'])
    state_dict = torch.load(gimpath, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    gimmodel.load_state_dict(state_dict)
    # if use_half:
    gimmodel.half()
    gimmodel.to(device)
    gimmodel.eval()
    return gimmodel


if __name__ == '__main__':
    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')
    
    workdir = cfg['workdir']

    img_dir = cfg['img_dir']
    device = cfg['device']

    nowpy = 3

    lsc = 2**nowpy

    testpath = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/TEST/testdata/SC_3'
    
    shp_dir = os.path.join(workdir, cfg['buildbox-01']['save_path'], f'range_py{nowpy}.shp')

    model = init_model(cfg['genfeature-02']['gimpath'], device)

    dataset = Get_local(img_dir, shp_dir, nowpy)
    test_load = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4)

    isd = 0
    imlist = os.listdir(testpath)
    imlist = ['0_SAT_198.png']
    for searchname in imlist:
        image1 = read_image(os.path.join(testpath, searchname))
        image1 = preprocess(image1)
        _, height, width = image1.shape
        destination_corners = np.array([
            [0, 0],            # Top-left
            [width-1, 0],    # Top-right
            [width - 1, height - 1],  # Bottom-right
            [0, height - 1]    # Bottom-left
        ], dtype=np.float32)
        orgpts = destination_corners.reshape(-1, 1, 2)

        image1 = image1.to(device)[None]
        
        with torch.no_grad():
            print(image1.shape)
            feat_c1, feat_f1, c1sp, f1sp = model.getfeature(image1.half())
            for cutinfo in tqdm.tqdm(test_load):
                image_ts = cutinfo['image'].to(device)    
                biasx =  cutinfo['stx'].numpy()[0]
                biasy =  cutinfo['sty'].numpy()[0]
                bigname = cutinfo['imname'][0]
                feat_c0, feat_f0, c0sp, f0sp = model.getfeature(image_ts.half())

                data = dict(color0=None, color1=image1, image0=None, image1=image1)
                data.update({
                    'bs': image_ts.size(0),
                    'hw0_i': (image_ts.shape[2], image_ts.shape[3]), 
                    'hw1_i': (image1.shape[2], image1.shape[3]), 
                })
                data.update({
                    'hw0_c': c0sp, 'hw1_c': c1sp,
                    'hw0_f': f0sp, 'hw1_f': f1sp
                })
                model.coarse_match(feat_c0, feat_c1, data)
                model.fine_match(feat_c0, feat_f0, feat_c1, feat_f1, data)
                            
                kpts0 = data['mkpts0_f']
                kpts1 = data['mkpts1_f']
                b_ids = data['m_bids']
                mconf = data['mconf'].cpu().detach().numpy()
                # robust fitting
                pts1_filtered = kpts0.cpu().detach().numpy()
                pts2_filtered = kpts1.cpu().detach().numpy()

                iou, tranbox, score, matchcount, H = getioubox(pts1_filtered, pts2_filtered, orgpts, mconf)

                if iou is None: continue

                if iou>0.9:
                    print(searchname, cutinfo['imname'], 'boxid', cutinfo['id'], iou)
                    tranbox = np.array([x*lsc for x in tranbox])
                    print(tranbox)
                    # # tranbox = np.round(tranbox)
                    box = [tranbox[0]+biasx, tranbox[1]+biasy, tranbox[2]+biasx, tranbox[3]+biasy]

                    dataset = gdal.Open(os.path.join(img_dir, bigname))
                    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
                    im_proj = dataset.GetProjection()
                    conbox = imgrect2geobox1(im_geotrans, box)
                    gdf = gpd.GeoDataFrame({'geometry':[conbox]})
                    gdf.crs = CRS.from_wkt(im_proj)  
                    
                    gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson'], f'{searchname}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
                    break
                elif iou>0.4:
                    tranbox = np.array([x*lsc for x in tranbox])
                    # tranbox = np.round(tranbox)
                    box = [tranbox[0]+biasx, tranbox[1]+biasy, tranbox[2]+biasx, tranbox[3]+biasy]
                    dataset = gdal.Open(os.path.join(img_dir, bigname))
                    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
                    im_proj = dataset.GetProjection()
                    conbox = imgrect2geobox1(im_geotrans, box)
                    gdf = gpd.GeoDataFrame({'geometry':[conbox]})
                    gdf.crs = CRS.from_wkt(im_proj)  
                    os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],searchname), exist_ok=True)
                    gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson_temp'], searchname, f'{int(iou*100)}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
                    print("Notice -> fine match", searchname, searchname, bigname, iou)
                





