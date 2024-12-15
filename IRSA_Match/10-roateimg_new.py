
import os
import numpy as np
import cv2
from utile import *
import tqdm
# 注意  是否需要去除一行
from torch.utils.data import Dataset



def splitimg(img, save_path, save_name, cutsize=1024):
    imh, imw, _ = img.shape
    
    cutlist = []
    for x in range(0, imw, cutsize):
        for y in range(0, imh, cutsize):
            print(x, y, imw, imh, cutsize)
            endx = x+cutsize
            endy = y+cutsize
            if endx>imw:
                if (endx - imw)/cutsize>0.5: continue
                endx = imw-1
                x = imw - cutsize-1
            if endy>imh:
                if (endy - imh)/cutsize>0.5: continue
                endy = imh-1
                y = imh - cutsize-1
            cutlist.append([x, y, endx, endy])

    for index, cut in enumerate(cutlist):
        imgcut = img[cut[1]:cut[3], cut[0]:cut[2]]
        cs = np.array([-cut[0], -cut[1], imw - cut[0]-1, imh - cut[1]-1])
        np.save(os.path.join(save_path, f'{index}-{save_name}.npy'), cs)
        cv2.imwrite(os.path.join(save_path, f'{index}-{save_name}'), imgcut)

class Get_Img(Dataset):
    def __init__(self, img_dir, imlist, pyscale, baseresultion, fine_cutsize, fine_pysacle, savedir):
        self.img_dir = img_dir
        self.imlist = imlist
        self.pyscale = pyscale
        self.baseresultion = baseresultion
        self.fine_cutsize = fine_cutsize
        self.fine_pysacle = fine_pysacle
        self.savedir = savedir
        self.finedict = {}
   
    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, i):
        imname = self.imlist[i]
        trans_image = cv2.imread(os.path.join(self.img_dir, imname))
        print(trans_image.shape)
        os.makedirs(os.path.join(self.savedir, f'NTest1'), exist_ok=True)
        splitimg(trans_image, os.path.join(self.savedir, f'NTest1'), imname, 512)
        return 1



if __name__ == '__main__':

    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')
    img_dir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/TEST/testdata/OTest'  # 旋转完成的数据地址
    workdir = cfg['workdir']
    pyscale = cfg['pyscale']
    extension = cfg['test_extension']
    baseresultion = cfg['baseresultion']
    tcfg = cfg['processtestimg-10']
    fine_pysacle = cfg['fine_pysacle']
    savedir = os.path.join(workdir, tcfg['save_path'])
    for pyindex in pyscale: os.makedirs(os.path.join(savedir, f'SC_{pyindex}'), exist_ok=True)
    os.makedirs(os.path.join(savedir, f'Fine_sc{fine_pysacle}'), exist_ok=True)

    imlist = list_files_with_extension(img_dir, extension)[:10]
    dataset = Get_Img(img_dir, imlist, pyscale, baseresultion, cfg['fine_cutsize'], fine_pysacle, savedir)
    test_load = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=2)

    for da in tqdm.tqdm(test_load):
        pass
#







