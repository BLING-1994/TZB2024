import os
import geopandas as gpd
from torch.utils.data import Dataset
from osgeo import gdal
import numpy as np
import cv2
from rasterio.transform import from_bounds
from irsa_inferenceserver.geotools import polygon2box
from shapely.geometry import Polygon
from copy import deepcopy
from utile import *
from loftr_dyp.loftr import LoFTR
from loftr_dyp.misc import lower_config
from loftr_dyp.config import get_cfg_defaults
from irsa_inferenceserver.geotools import imgrect2geobox1, read_info
import geopandas as gpd
from rasterio.crs import CRS
import shutil

def transfromH(H, orgpts):
    dst = cv2.perspectiveTransform(orgpts, H)[:,0,:]
    # print(dst)
    xlist = [d[0] for d in dst]
    ylist = [d[1] for d in dst]
    sx0, sy0 = min(xlist), min(ylist)
    sx1, sy1 = max(xlist), max(ylist)
    return [sx0, sy0, sx1, sy1]

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

class Get_local(Dataset):
    def __init__(self, img_dir, nowscale, scpath, scpoly, rangepd, maxrange=10, dfactor=8):
        self.img_dir = img_dir
        self.nowscale = nowscale
        self.dataset = None
        self.nowimg = None
        self.scpoly = scpoly
        self.rangepd = rangepd
        self.maxrange = maxrange
        self.dfactor = dfactor
        self.scpath = scpath
        self.searchname = os.path.basename(scpath)
        self.gettrans(scpath, scpoly)
        self.getinterbox()
    
    def gettrans(self, imgpath, scpoly):
        self.scdataset = gdal.Open(imgpath)
        self.scwidth = self.scdataset.RasterXSize  # 栅格矩阵的列数
        self.scheight = self.scdataset.RasterYSize
        self.sc_layers = []
        for bandindex in range(self.scdataset.RasterCount):
            self.sc_layers.append(self.scdataset.GetRasterBand(bandindex+1))
        bound = scpoly.bounds
        pixel_width = (bound[2] - bound[0]) / self.scwidth
        pixel_height = (bound[3] - bound[1]) / self.scheight
        self.sctransform = (bound[0], pixel_width, 0, bound[3], 0, -pixel_height)
        self.scrangebox = [[0, 0],            # Top-left
                            [self.scwidth-1, 0],    # Top-right
                            [self.scwidth-1, self.scheight-1],  # Bottom-right
                            [0, self.scheight-1]]
    
    def covertbox(self, box):
        x0, y0, x1, y1 = box
        if x0<0: x0=0
        if y0<0: y0=0
        if x1>self.scwidth-1: x1 = self.scwidth-1
        if y1>self.scheight-1: y1 = self.scheight-1

        w = x1 - x0
        w = w // self.dfactor * self.dfactor
        h = y1 - y0
        h = h // self.dfactor * self.dfactor

        return [x0, y0, x0+w, y0+h]


    def initdata(self, imname):
        del self.dataset
        print(os.path.join(self.img_dir, imname))
        self.dataset = gdal.Open(os.path.join(self.img_dir, imname))
        self.band_count = self.dataset.RasterCount
        self.im_proj = self.dataset.GetProjection()
        self.nowimg = imname
        self.band_layers = []
        if self.nowscale>0:
            for bandindex in range(self.band_count):
                self.band_layers.append(self.dataset.GetRasterBand(bandindex+1).GetOverview(self.nowscale-1))
        else:
            for bandindex in range(self.band_count):
                self.band_layers.append(self.dataset.GetRasterBand(bandindex+1))

    def getdatafromrow(self, row):
        imname = row['feaname']
        stx = int(row['stx']/(2 ** self.nowscale))
        sty = int(row['sty']/(2 ** self.nowscale))
        self.initdata(imname)
        out = []
        for ly in self.band_layers:
            out.append(ly.ReadAsArray(stx, sty, 512, 512)[None])
        image = np.concatenate(out, 0)
        return image

    def getscdatafrombox(self, box):
        out = []
        for ly in self.sc_layers:
            out.append(ly.ReadAsArray(box[0], box[1], box[2], box[3])[None])
        image = np.concatenate(out, 0)
        return image


    def getinterbox(self):
        sindex_d1 = self.rangepd.sindex
        possible_matches_index = list(sindex_d1.intersection(self.scpoly.bounds))
        self.possible_matches = self.rangepd.iloc[possible_matches_index]
        # 排序
        if len(self.possible_matches)>self.maxrange:
            self.possible_matches = self.possible_matches.sample(n=self.maxrange, random_state=42)

        self.cutlist = []
        for idx, row in self.possible_matches.iterrows():
            cutbox = polygon2box(self.sctransform, row['geometry'])
            x,y,w,h = cutbox
            sd = self.covertbox([x,y,x+w,y+h])
            sw = sd[2] - sd[0]
            sh = sd[3] - sd[1]
            newtrans = list(deepcopy(self.sctransform))
            newtrans[0] = self.sctransform[0] + sd[0]*self.sctransform[1]
            newtrans[3] = self.sctransform[3] + sd[1]*self.sctransform[5]

            pts = [[pt[0] - sd[0], pt[1] - sd[1]]for pt in self.scrangebox]
            
            finepts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            
            if sw>100 and sh>100:
                self.cutlist.append(([sd[0], sd[1], sw, sh], row, newtrans, finepts, (sd[0], sd[1])))


    def __len__(self):
        return len(self.cutlist)

    def __getitem__(self, i):
        scbox, row, transform, scpartpts, biasxy = self.cutlist[i]

        bigimg = self.getdatafromrow(row)
        scimg = self.getscdatafrombox(scbox)
        simplr = {'image1': (torch.from_numpy(scimg) / 255.0).float(),
                  'image0': (torch.from_numpy(bigimg) / 255.0).float(),
                  'transform': transform,
                  'scpartpts': scpartpts,
                  'biasxy': biasxy}

        return simplr
    

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
        if iou is None: return None, None, None
        return iou, tranbox, H


def rotate_and_recenter(points, center, angle):
    # 检查角度是否有效
    if angle not in [90, 180, 270]:
        raise ValueError("旋转角度只能是90, 180, 270度")
    
    # 将角度转为弧度
    radians = np.radians(angle)
    
    # 定义旋转矩阵
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    
    # 转换点集为 numpy 数组
    points_array = np.array(points)
    center = np.array(center)
    
    # 平移点到原点（以 center 为原点）
    translated_points = points_array - center
    
    # 应用旋转矩阵
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    
    # 将中心设为新的原点（不需要平移回去）
    recentered_points = rotated_points
    return recentered_points.tolist()

def finematch(cfg, model, searchname, corspoly, rangepd, fine_pysacle, device, rot_id=0, outshp=True):
    scpath = os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],f'SC_{fine_pysacle}', searchname)
    if rot_id!=0:
        scpath = os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],f'SC_{fine_pysacle}_{rot_id}', searchname)
    
    biggetdataset = Get_local(cfg['img_dir'], fine_pysacle, scpath, corspoly, rangepd, maxrange = 10)
    lsc = 2**fine_pysacle
    bestiou = 0
    besttranbox = None
    besttransform = None
    bestbiasxy = None
    bestH = None
    for datainfo in biggetdataset:
        iou, tranbox, H = finematchsigle(datainfo, model, device)
        if iou is None: continue
        if iou>bestiou:
            bestiou = iou
            besttranbox = tranbox
            besttransform = datainfo['transform']
            bestbiasxy = datainfo['biasxy']
            bestH = H
        if iou>0.99: break
    
    if bestiou<0.4: return False
    searchnamelist = searchname.split('-')
    if len(searchnamelist)>1: orgname = searchnamelist[1]
    else: orgname = searchname
    
    npypath = os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],'NTest', searchname + '.npy')
    finenpy = np.load(npypath)
    scdataset = gdal.Open(os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],'NTest', searchname))
    scwidth = scdataset.RasterXSize-1  # 栅格矩阵的列数
    scheight = scdataset.RasterYSize-1


    finepts = [[finenpy[0], finenpy[1]],            # Top-left
            [finenpy[2], finenpy[1]],    # Top-right
            [finenpy[2], finenpy[3]],  # Bottom-right
            [finenpy[0], finenpy[3]]]
    print(finepts, scwidth, scheight)
    # if rot_id==2:
    #     finepts = rotate_and_recenter(finepts, (scwidth, scheight), 180)
    # elif rot_id==1:
    #     finepts = rotate_and_recenter(finepts, (scwidth, 0), 90)
    # elif rot_id==3:
    #     finepts = rotate_and_recenter(finepts, (0, scheight), 270)

    if rot_id==2:
        finepts = rotate_and_recenter(finepts, (scwidth, scheight), 180)
    elif rot_id==1:
        finepts = rotate_and_recenter(finepts, (0, scheight), 90)
    elif rot_id==3:
        finepts = rotate_and_recenter(finepts, (scwidth, 0), 270)
    print(finepts, '========')


    pts = [[pt[0]/lsc - bestbiasxy[0], pt[1]/lsc - bestbiasxy[1]]for pt in finepts]
    finepts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    finepts = transfromH(bestH, finepts)

    conbox = imgrect2geobox1(besttransform, finepts)
    gdf = gpd.GeoDataFrame({'geometry':[conbox]})
    gdf.crs = CRS.from_wkt(biggetdataset.im_proj)  
    if bestiou>0.9:
        print('find --- ', bestiou)
        os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson']), exist_ok=True)
        gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson'], f'{orgname}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
        if os.path.exists(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname)):
            shutil.rmtree(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname))
        return True
    elif bestiou>0.4:
        os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson_temp'],orgname), exist_ok=True)
        gdf.to_file(os.path.join(cfg['workdir'], cfg['output_geojson_temp'], orgname, f'{int(bestiou*100)}.geojson'), driver='GeoJSON', encoding='utf-8', engine='pyogrio')
        print("Notice -> fine match", searchname, orgname, bestiou)



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
    cfg['fine_pysacle']

    boxpath = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/Output/Vec/C_11_SAT_196.png.geojson'
    scpath = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/TEST/testdata-0/SC_3/30_SAT_198.png'
    shppath = f'/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/IRSA_DT/Range/range_py2.shp'
    # img_dir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/hebei/bigbase'

    boxpolygon = gpd.read_file(boxpath)
    scpoly = boxpolygon.geometry.iloc[0]
    rangepd = gpd.read_file(shppath)

    # cfg['processtestimg-10']['save_path'] = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/TEST/testdata-0'

    finematch(cfg, model, '2-11_SAT_196.png', scpoly, rangepd, cfg['fine_pysacle'],  device, 3)
    

 



        





