import os
import numpy as np
from utile import *
from torch.utils.data import Dataset
from osgeo import gdal
from loftr_dyp.loftr import LoFTR
from loftr_dyp.misc import lower_config
from loftr_dyp.config import get_cfg_defaults
from irsa_inferenceserver.geotools import imgrect2geobox1, read_info
import geopandas as gpd
from rasterio.crs import CRS
import shutil

import logging
logger = logging.getLogger(__name__)

def getimgfrombox(datapath, fine_pysacle, finebox):
    finebox = finebox/(2**fine_pysacle)
    # 切出8的整数倍
    dfactor = 32
    stx, sty, stw, sth = finebox[0], finebox[1], finebox[2]-finebox[0], finebox[3]-finebox[1]
    stw = stw// dfactor * dfactor
    sth = sth// dfactor * dfactor

    dataset = gdal.Open(datapath)
    band_count = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息

    out = []
    if fine_pysacle>0:
        for bandindex in range(band_count):
            out.append(dataset.GetRasterBand(bandindex+1).GetOverview(fine_pysacle-1).ReadAsArray(stx, sty, stw, sth)[None])
    else:
        for bandindex in range(band_count):
            out.append(dataset.GetRasterBand(bandindex+1).ReadAsArray(stx, sty, stw, sth)[None])
    image = np.concatenate(out, 0)# .transpose(1,2,0)
    return image, im_geotrans,im_proj  # RGB


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


def finematch(cfg, model, searchname, finebox, bigname, fine_pysacle, device, outshp=True):
    lsc = 2**fine_pysacle
    biasx = finebox[0]*lsc
    biasy = finebox[1]*lsc
    
    # search_info
    finepath = os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'], f'Fine_sc{fine_pysacle}')
    finenpy = np.load(os.path.join(finepath, searchname + '.npy'))[1]
    finenpy[2] = finenpy[2]-1
    finenpy[3] = finenpy[3]-1
    
    finepts = np.array([
            [finenpy[0], finenpy[1]],            # Top-left
            [finenpy[2], finenpy[1]],    # Top-right
            [finenpy[2], finenpy[3]],  # Bottom-right
            [finenpy[0], finenpy[3]]    # Bottom-left
        ], dtype=np.float32)
    finepts = finepts.reshape(-1, 1, 2)

    # search_feature
    image1 = read_image(os.path.join(finepath, searchname))
    image1 =  torch.from_numpy(image1.transpose((2, 0, 1)) / 255.0).float()
    image1 = image1.to(device)[None]
    with torch.no_grad():
        feat_c1, feat_f1, c1sp, f1sp = model.getfeature(image1.half())


    img_path = os.path.join(cfg['img_dir'], bigname)
    image0, im_geotrans, im_proj = getimgfrombox(img_path, fine_pysacle, finebox)
    image0 =  torch.from_numpy(image0 / 255.0).float()
    image0 = image0.to(device)[None]


    with torch.no_grad():
        feat_c0, feat_f0, c0sp, f0sp = model.getfeature(image0.half())
        data = dict(color0=None, color1=image1, image0=None, image1=image1)
        data.update({
            'bs': image0.size(0),
            'hw0_i': (image0.shape[2], image0.shape[3]), 
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

        iou, tranbox, score, matchcount, H = getioubox(pts1_filtered, pts2_filtered, finepts, mconf)
        if iou is None: return False

        searchnamelist = searchname.split('-')
        if len(searchnamelist)>1: orgname = searchnamelist[1]
        else: orgname = searchname
        
        if iou>0.9:
            tranbox = np.array([x*lsc for x in tranbox])
            # tranbox = np.round(tranbox)
            box = [tranbox[0]+biasx, tranbox[1]+biasy, tranbox[2]+biasx, tranbox[3]+biasy]
            conbox = imgrect2geobox1(im_geotrans, box)
            gdf = gpd.GeoDataFrame({'geometry':[conbox]})
            gdf.crs = CRS.from_wkt(im_proj)  
            
            gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson'], f'{orgname}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
            if os.path.exists(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname)):
                shutil.rmtree(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname))
            return True
        elif iou>0.4:
            tranbox = np.array([x*lsc for x in tranbox])
            # tranbox = np.round(tranbox)
            box = [tranbox[0]+biasx, tranbox[1]+biasy, tranbox[2]+biasx, tranbox[3]+biasy]
            conbox = imgrect2geobox1(im_geotrans, box)
            gdf = gpd.GeoDataFrame({'geometry':[conbox]})
            gdf.crs = CRS.from_wkt(im_proj)  
            os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname), exist_ok=True)
            gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson_temp'], orgname, f'{int(iou*100)}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
            print("Notice -> fine match", searchname, orgname, finebox, bigname, iou)
        return False
    

if __name__ == '__main__':

    # todo 需要考虑旋转和超出边界的情况

    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')
    # workdir = cfg['workdir']
    # fine_pysacle = cfg['fine_pysacle']
    # img_dir = cfg['img_dir']
    device = cfg['device']
    model = init_model(cfg['genfeature-02']['gimpath'], device)
    # input 
    searchname = '37_SAT_044.png'
    finebox = np.array([87744.0, 39020.0, 88504.0, 39780.0])
    bigname = 'warp_保定定州.TIF'
    # print("=============", iou, tranbox, score, matchcount, box)  

