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
            # im_width = ly.XSize  # 栅格矩阵的列数
            # im_height = ly.XSize
            # print(id, im_width, im_height, stx, sty)
            out.append(ly.ReadAsArray(stx, sty, 512, 512)[None])
        image = np.concatenate(out, 0)# .transpose(1,2,0)
        image = torch.from_numpy(np.ascontiguousarray(image / 255.0)).float()
        sample = {'image':image, 'id': id}
        return sample
    

class MultiPostprocess:
    def __init__(self, processescount=4, debug=False):
        self.debug = debug
        self.processescount = processescount
        self.manager = mp.Manager  
        self.works = self.manager().list()
        self.works_nop = self.manager().Value(bool, True)
        self.send_finish = self.manager().Value(bool, False)
        if self.debug is False:
            print("multi procese")
            pool = mp.Pool(processes=processescount)
            for index in range(processescount):
                pool.apply_async(self.threadporcess)      
            pool.close() 
        else:
            print("single procese", debug)
       
    
    def spr(self, gimdata, gimpath, dinodata, dinopath):
        torch.save(gimdata, gimpath)
        np.save(dinopath, dinodata)
    
    def threadporcess(self):
        while True:
            if len(self.works)==0: 
                self.works_nop.value = True
                if self.send_finish.value==True: 
                    current_process = mp.current_process()
                    print(current_process.pid, '进程任务完成退出')
                    break
                continue
            self.works_nop.value = False
            gimdata, gimpath, dinodata, dinopath = self.works.pop()
            self.spr(gimdata, gimpath, dinodata, dinopath)
    
    def inputwork(self, gimdata, gimpath, dinodata, dinopath):
        if self.debug:
            self.spr(gimdata, gimpath, dinodata, dinopath)
        else:
            while len(self.works)>self.processescount*5:
                # print(len(self.works))
                time.sleep(0.1)
            self.works.append([gimdata, gimpath, dinodata, dinopath])

    def waitfinish(self):
        self.send_finish.value = True
        while self.works_nop.value==False:
            print(len(self.works))
            time.sleep(0.5)
            pass
        print('save finish')

            
def init_model(dinopath, gimpath, device='cuda'):
    dinov2model = DINOExtract(dinopath, use_half=False, device=device)
    gimmodel = LoFTR(lower_config(get_cfg_defaults())['loftr'])
    state_dict = torch.load(gimpath, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    gimmodel.load_state_dict(state_dict)
    # if use_half:
    gimmodel.half()
    gimmodel.to(device)
    gimmodel.eval()
    return dinov2model, gimmodel


if __name__ == '__main__':
    import sys
    cfg = load_config(sys.argv[1])
    
    tcfg = cfg['genfeature-02']
    workdir = cfg['workdir']
    pyscale = cfg['pyscale']# 5层金字塔
    img_dir = cfg['img_dir']
    device = cfg['device']


    for nowpy in pyscale:
        print('py -> ', nowpy)

        gim_savepath = os.path.join(workdir, tcfg['save_path'], f'SC_{nowpy}', 'Match')
        dino_savepath = os.path.join(workdir, tcfg['save_path'], f'SC_{nowpy}', 'Search')
        os.makedirs(gim_savepath, exist_ok=True)
        os.makedirs(dino_savepath, exist_ok=True)

        base_indexsavepath = os.path.join(workdir, tcfg['save_path'], f'SC_{nowpy}', 'BeseIndex.pkl')
        base_infosavepath = os.path.join(workdir, tcfg['save_path'], f'SC_{nowpy}', 'BeseInfo.pkl')

        shp_dir = os.path.join(workdir, cfg['buildbox-01']['save_path'], f'range_py{nowpy}.shp')

        dinov2model, gimmodel = init_model(tcfg['dinopath'], tcfg['gimpath'], device)

        dataset = Get_local(img_dir, shp_dir, nowpy, dino_savepath)
        test_load = torch.utils.data.DataLoader(dataset, batch_size=tcfg['batch_size'], pin_memory=True, num_workers=tcfg['num_workers'])

        allindex = []
        allinfo = {}

        mprocess = MultiPostprocess(tcfg['processworks'], cfg['debug'])

        isd = 0
        
        with torch.no_grad():
            for data in tqdm.tqdm(test_load):
                isd += 1
                image_ts = data['image'].to(device)

                mean_fe = dinov2model(image_ts).cpu().numpy()
                feat_c0, feat_f0 = gimmodel.backbone(image_ts.half())
                feat_c0 = feat_c0.cpu()
                feat_f0 = feat_f0.cpu()

                for index in range(mean_fe.shape[0]):
                    nowid = data['id'][index].numpy()
                    df = mean_fe[index]

                    mprocess.inputwork(feat_c0[index], 
                                    os.path.join(gim_savepath, f'C_{nowpy}_{nowid}.IP'), 
                                    df, 
                                    os.path.join(dino_savepath, f'{nowpy}_{nowid}.npy'))

        mprocess.waitfinish()
        allnpylist = os.listdir(dino_savepath)
        for index, npyname in enumerate(tqdm.tqdm(allnpylist)):
            allindex.append(np.load(os.path.join(dino_savepath, npyname)))
            id = npyname.split('_')[1].replace('.npy', '')
            allinfo[index] = id

        with open(base_indexsavepath, 'wb') as f:
            pickle.dump(allindex, f)
        with open(base_infosavepath, 'wb') as f:
            pickle.dump(allinfo, f)



