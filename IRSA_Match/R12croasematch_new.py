# -*- coding: utf-8 -*-
# @Author  : xuelun
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from loftr_dyp.loftr import LoFTR
from loftr_dyp.misc import lower_config
from loftr_dyp.config import get_cfg_defaults
import tqdm
from utile import *
import time
import geopandas as gpd
from torch.utils.data import Dataset
import logging
from irsa_inferenceserver.geotools import imgrect2geobox1, read_info
from osgeo import gdal
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


# def roatepoint(gpts, k):
#     if k==1:

def getrote(H):
    # 提取旋转矩阵部分
    R = H[:2, :2]
    print(R)

    # 计算旋转角度 (弧度)
    theta = np.arctan2(R[1, 0], R[0, 0])

    # 转换为角度
    angle = np.degrees(theta)
    return angle


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



class Get_Feature(Dataset):
    def __init__(self, featuredir, idlist, pyscale):
        self.featuredir = featuredir
        self.idlist = idlist
        self.pyscale = pyscale

   
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, i):
        id = self.idlist[i]
        name = f"C_{self.pyscale}_{id}.IP"
        #feat_c011 = torch.load(os.path.join(featuredir, name))

        return {'feature': torch.load(os.path.join(self.featuredir, name)), 'id': id}


def singelpicfind(model, search_name, pyscale, cfg, rangegdf, search_idlist, finematchfun=None):
    workdir = cfg['workdir']
    # pyscale = cfg['pyscale']# 5层金字塔
    img_dir = cfg['img_dir']
    device = cfg['device']
    cut_size = cfg['cut_size']
    fine_pysacle = cfg['fine_pysacle']
    lsc = 2**pyscale

    searchnamelist = search_name.split('-')
    if len(searchnamelist)>1: 
        orgname = searchnamelist[1]
    else:
        orgname = search_name
    
    finerangepd = gpd.read_file(os.path.join(cfg['workdir'],cfg['buildbox-01']['save_path'], f'range_py{fine_pysacle}.shp'))


    if os.path.exists(os.path.join(cfg['workdir'], cfg['output_geojson'], f'{orgname}.geojson')): 
        print(' 已检索 continue  orgname:', orgname, ' search_name:', search_name)
        return
    os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname), exist_ok=True)

    img_path1 = os.path.join(workdir, cfg['processtestimg-10']['save_path'], f'SC_{pyscale}/{search_name}')
    featuredir = os.path.join(workdir, cfg['genfeature-02']['save_path'], f'SC_{pyscale}/Match')


    image1 = read_image(img_path1)
    image1 = preprocess(image1)

    _, height, width = image1.shape
    destination_corners = np.array([
        [0, 0],            # Top-left
        [width-1, 0],    # Top-right
        [width - 1, height - 1],  # Bottom-right
        [0, height - 1]    # Bottom-left
    ], dtype=np.float32)
    orgpts_norm = destination_corners.reshape(-1, 1, 2)

    
    destination_corners = np.array([
        [0, 0],            # Top-left
        [height-1, 0],    # Top-right
        [height - 1, width - 1],  # Bottom-right
        [0, width - 1]    # Bottom-left
    ], dtype=np.float32)
    orgpts_change = destination_corners.reshape(-1, 1, 2)

    image1 = image1.to(device)[None]
    b_ids, mconf, kpts0, kpts1 = None, None, None, None
    outdata = {}
    
    find = False

    fedataset = Get_Feature(featuredir, search_idlist, pyscale)
    test_load = torch.utils.data.DataLoader(fedataset, batch_size=1, pin_memory=True, num_workers=4)
    print(f'==================={search_name}', image1.shape)

    rotlist = [0, 1, 2, 3]
    with torch.no_grad():
        rfealist = dict()
        image1 = image1.half()
        for rt in rotlist:
            
            if rt==0:
                cvimg = image1.clone()
                rfealist[rt] = {'orgpts':orgpts_norm}
            elif rt==3:
                rfealist[rt] = {'orgpts':orgpts_change}
                cvimg = torch.rot90(image1.clone(), k=1, dims=(2, 3))
            elif rt==2:
                rfealist[rt] = {'orgpts':orgpts_norm}
                cvimg = torch.rot90(image1.clone(), k=2, dims=(2, 3))  # 确定
            elif rt==1:
                rfealist[rt] = {'orgpts':orgpts_change}
                cvimg = torch.rot90(image1.clone(), k=3, dims=(2, 3))
            feat_c1, feat_f1, c1sp, f1sp = model.getfeature(cvimg.half())
            rfealist[rt]['feat_c1'] = feat_c1
            rfealist[rt]['c1sp'] = c1sp
            # 旋转点
            
        # for id in tqdm.tqdm(search_idlist):
        for fedata in tqdm.tqdm(test_load):
            id = fedata['id'][0].numpy()
            row = rangegdf.loc[rangegdf['id'] == id].iloc[0]
            name = f"C_{pyscale}_{row['id']}.IP"
            # feat_c011 = torch.load(os.path.join(featuredir, name))[None]
            feat_c011 = fedata['feature'].to(device)
            feat_c0, feat_f0, c0sp, f0sp = model.getfeature(feat_c0 = feat_c011)
            for rt, fda in rfealist.items():
                data = dict(color0=None, color1=image1, image0=None, image1=image1)
                
                data.update({
                    'bs': data['image1'].size(0),
                    'hw0_i': (cut_size, cut_size), 
                    'hw1_i': (data['image1'].shape[2], data['image1'].shape[3]), 
                })

                data.update({
                    'hw0_c': c0sp, 'hw1_c': fda['c1sp'],
                    'hw0_f': None, 'hw1_f': None
                })
                
                model.coarse_match(feat_c0, fda['feat_c1'], data)                      
                kpts0 = data['mkpts0_c']
                kpts1 = data['mkpts1_c']
                b_ids = data['m_bids']
                mconf = data['mconf'].cpu().detach().numpy()

                # robust fitting
                pts1_filtered = kpts0.cpu().detach().numpy()
                pts2_filtered = kpts1.cpu().detach().numpy()


                iou, tranbox, score, matchcount, H = getioubox(pts1_filtered, pts2_filtered, fda['orgpts'], mconf)

                if iou is None: continue
                if iou>0.9:
                    angle = getrote(H)
                    if abs(angle)>45: continue 

                    print('------', getrote(H), rt)
                    if name not in outdata: outdata[name] = []
                    biasx, biasy = row['stx'], row['sty']
                    tranbox = np.array([x*lsc for x in tranbox])
                    tranbox = np.round(tranbox)
                    transbox = np.array([tranbox[0]+biasx, tranbox[1]+biasy, tranbox[2]+biasx, tranbox[3]+biasy])
                    print(row['feaname'], name)
                    dataset = gdal.Open(os.path.join(cfg['img_dir'], row['feaname']))
                    band_count = dataset.RasterCount
                    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
                    im_proj = dataset.GetProjection()  # 地图投影信息
                    conbox = imgrect2geobox1(im_geotrans, transbox)  # 给的是当前找寻影像part范围，并不是真实范围
                    gdf = gpd.GeoDataFrame({'geometry':[conbox]})
                    gdf.crs = CRS.from_wkt(im_proj)  
                    os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson']), exist_ok=True)
                    gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson'], f'C_{orgname}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')

                    if finematchfun(cfg, model, search_name, conbox, finerangepd, fine_pysacle, device, rt, True):
                        find=True
                        
                    break
            if find is True: 
                print('找寻成功')
                break
    if find is False:
        print('失败', image1.shape)

        




if __name__ == '__main__':
    import sys

    cfg = load_config('config/hebei.yaml')

    # device = cfg['device']
    # pyscale = 3
    
    
    # model = init_model(cfg['genfeature-02']['gimpath'], device)

    # search_name = '37_SAT_044.png'

    # singelpicfind(model, search_name, pyscale, cfg)

    

    

                
        
                


   
