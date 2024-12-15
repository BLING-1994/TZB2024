from torch.utils.data import Dataset
from utile import *
import geopandas as gpd
import os
from osgeo import gdal
import numpy as np
import cv2
import torch
import tqdm


import multiprocessing as mp  
import time



class Get_local(Dataset):
    def __init__(self, img_dir, shp_dir, nowscale, dino_savepath):
        self.img_dir = img_dir
        self.nowscale = nowscale
        self.reuescount = 0
        self.dino_savepath = dino_savepath
        self.gdf = gpd.read_file(shp_dir)
        self.filterexsit()
        self.dataset = None
        self.nowimg = None
    
    def filterexsit(self):
        imlist = os.listdir(self.dino_savepath)
        self.runlist = []
        for index, row in tqdm.tqdm(self.gdf.iterrows()):
            id = row['id']
            npyname = f'{self.nowscale}_{id}.npy'
            if npyname in imlist: continue
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
        if nowpy>0:
            for bandindex in range(self.band_count):
                self.band_layers.append(self.dataset.GetRasterBand(bandindex+1).GetOverview(nowpy-1))
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
            # im_width = ly.XSize  # 栅格矩阵的列数
            # im_height = ly.XSize
            # print(id, im_width, im_height, stx, sty)
            out.append(ly.ReadAsArray(stx, sty, 512, 512)[None])
        image = np.concatenate(out, 0)# .transpose(1,2,0)
        image = torch.from_numpy(np.ascontiguousarray(image / 255.0)).float()
        sample = {'image':image, 'id': id}
        return sample
    




if __name__ == '__main__':

    datasetpath = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/JL_2023_4326/warp_JL1KF01B_PMSL4_20230416104042_200149955_101_0058_001_L1.tif'
    # datasetpath = '/irsa_public/大图幅测试数据/采购数据全/L3D/JL1KF01B_PMSL4_20230416104042_200149955_101_0058_001_L3D_PSH.tif'
    save_path = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/cudtdata'
    os.makedirs(save_path, exist_ok=True)
    # datasetpath = '/irsa_public/大图幅测试数据/采购数据全/L3D/JL1KF01B_PMSL1_20220317104342_200078402_102_0041_001_L3D_PSH.tif'
    dataset = gdal.Open(datasetpath)
    band_count = dataset.RasterCount

    nowpy = 3

    band_layers = []
    for bandindex in range(band_count):
        band_layers.append(dataset.GetRasterBand(bandindex+1).GetOverview(nowpy-1))

    out = []
    for ly in band_layers:
        ly = band_layers[0]
        im_width = ly.XSize  # 栅格矩阵的列数
        im_height = ly.YSize
        print(im_width, im_height)
        out.append(ly.ReadAsArray(1500, 1500, 152, 212)[None])

    image = np.concatenate(out, 0).transpose(1,2,0)[:,:,::-1]
    cv2.imwrite(os.path.join(save_path,'0058qs_SAT_400.png'), image)
