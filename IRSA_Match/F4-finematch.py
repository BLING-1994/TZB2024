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



def finematchsigle(datainfo, model, device):
    image1 = datainfo['image1']
    image0 = datainfo['image0']
    partpts = datainfo['scpartpts']
    transf = datainfo['transform']

    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]
    with torch.no_grad():
        feat_c1, feat_f1, c1sp, f1sp = model.getfeature(image1.half())
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

        iou, tranbox, score, matchcount, H = getioubox(pts1_filtered, pts2_filtered, partpts, mconf)
        print(H)
        angle = getrote(H)
        if abs(angle)>45: return None, None 
        if iou is None: return None, None
        return iou, tranbox



def readprocessdata(path, maxlen=1536, Getinfo=False, rot=None):
    bigimg = cv2.imread(path, -1)
    if rot==1:
        bigimg = cv2.rotate(bigimg, cv2.ROTATE_90_CLOCKWISE)
    if rot==2:
        bigimg = cv2.rotate(bigimg, cv2.ROTATE_180)
    if rot==3:
        bigimg = cv2.rotate(bigimg, cv2.ROTATE_90_COUNTERCLOCKWISE)

    biglen = max(bigimg.shape)
    bigscale = maxlen/biglen
    dfactor = 8
    height, width = bigimg.shape[:2]
    new_width = int(np.round(width * bigscale))
    new_height = int(np.round(height * bigscale))
    new_size = (new_width, new_height)
    bigimg = cv2.resize(bigimg, new_size, interpolation=cv2.INTER_CUBIC)
    height, width = bigimg.shape[:2]
    orgw = width
    orgh = height
    # cut 
    height = height// dfactor * dfactor
    width = width // dfactor * dfactor

    bigimg = bigimg[:height, :width]
    bigimg = np.ascontiguousarray(bigimg[:,:,::-1].transpose(2,1,0))
    # bigimg = np.ascontiguousarray(bigimg)
    if Getinfo:
        dataset = gdal.Open(path)  # 打开文件
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        # im_width = dataset.RasterXSize  # 栅格矩阵的列数
        # im_height = dataset.RasterYSize
        im_proj = dataset.GetProjection()  # 地图投影信息
        return bigimg, bigscale,  im_geotrans, im_proj
    scrangebox = [[0, 0],            # Top-left
                    [orgw, 0],    # Top-right
                    [orgw, orgh-1],  # Bottom-right
                    [0, orgh-1]]
    
    return bigimg, np.array(scrangebox, dtype=np.float32).reshape(-1, 1, 2)


    

if __name__ == '__main__':

    # todo 需要考虑旋转和超出边界的情况

    import sys
    cfg = load_config('config/hebei.yaml')
    device = cfg['device']

    maxlen = 1536
    rotlist = [0, 1, 2, 3]
    model = init_model(cfg['genfeature-02']['gimpath'], device)

    save_path = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testmatch/fine_geojson'
    serachdir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testmatch/searchdata'
    basedir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testmatch/basemap'
    os.makedirs(save_path, exist_ok=True)
    for name in os.listdir(basedir):
        if os.path.exists(os.path.join(save_path, name+'.geojson')): 
            print('exits fine pATH', name)
            continue
        bigcutpath = os.path.join(basedir, name)
        searchpath = os.path.join(serachdir, name.replace('.tif', '.png'))
        bigimg, bigscale,  im_geotrans, im_proj= readprocessdata(bigcutpath, maxlen = maxlen, Getinfo=True)
        for rot in rotlist:
            searimg, searpts = readprocessdata(searchpath, maxlen = maxlen, Getinfo=False, rot=rot)
            simplr = {'image1': (torch.from_numpy(searimg) / 255.0).float(),
                        'image0': (torch.from_numpy(bigimg) / 255.0).float(),
                        'transform': im_geotrans,
                        'proj':im_proj,
                        'scpartpts': searpts}

            iou, tranbox = finematchsigle(simplr, model, device)
            if iou is None: continue 
            print(tranbox)
            stransbox = [x/bigscale for x in tranbox]
            print(stransbox)
            print('clu iou  ', iou)
            # stransbox = [0, 0, 3832, 4962-1]
            conbox = imgrect2geobox1(im_geotrans, stransbox)
            gdf = gpd.GeoDataFrame({'geometry':[conbox]})
            gdf.crs = CRS.from_wkt(im_proj)  
            gdf.to_file(os.path.join(save_path, name+'.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
            break
    

    
    
  
 

