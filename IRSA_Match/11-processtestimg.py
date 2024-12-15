
import os
import numpy as np
import cv2
from utile import *
import tqdm
# 注意  是否需要去除一行
from torch.utils.data import Dataset

def roateimg2img(imgpath):
    img = cv2.imread(imgpath)
    img, corners = getimgbound(img)
    pt0 = corners[0]
    pt1 = corners[1]
    pt2 = corners[2]
    width = round(np.sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2))
    height =  round(np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))
    destination_corners = np.array([
        [0, 0],            # Top-left
        [width-1, 0],    # Top-right
        [width - 1, height - 1],  # Bottom-right
        [0, height - 1]    # Bottom-left
    ], dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(corners, destination_corners)
    # Apply the perspective transformation to the image
    transformed_image = cv2.warpPerspective(img, transformation_matrix, (width, height))
    return np.ascontiguousarray(transformed_image)




def clipdata(img, resul, fine_pyscale=1, cutsize=768, bigbox=None, dfactor=16):
    tagres = baseresultion*(2**fine_pyscale)
    scale = resul/tagres
    if bigbox is not None:
        bigbox = [x*scale for x in bigbox]
    resized_image = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width, _ = resized_image.shape
    cuth = cutsize
    if height< cutsize: 
        cuth = height // dfactor * dfactor
    cutw = cutsize
    if width< cutsize: 
        cutw = width // dfactor * dfactor
    if bigbox is not None:
        return resized_image[:cuth, :cutw], np.array([[0,0,cutw,cuth], bigbox])
    return resized_image[:cuth, :cutw], np.array([[0,0,cutw,cuth], [0, 0, width, height]])


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
        resul = float(os.path.splitext(imname)[0].split('_')[2])/100
        # trans_image = roateimg2img(os.path.join(self.img_dir, imname))
        trans_image = cv2.imread(os.path.join(self.img_dir, imname))
        print('==== ', imname)
        npybox = None
        if os.path.exists(os.path.join(self.img_dir, imname+'.npy')):
            npybox = np.load(os.path.join(self.img_dir, imname+'.npy'))
        fineimg, rnp = clipdata(trans_image, resul, self.fine_pysacle, cfg['fine_cutsize'], npybox)
        cv2.imwrite(os.path.join(self.savedir, f'Fine_sc{self.fine_pysacle}', imname), fineimg)
        np.save(os.path.join(self.savedir, f'Fine_sc{self.fine_pysacle}', imname + '.npy'), rnp)
        
        for pyindex in self.pyscale:
            tagres = self.baseresultion*(2**pyindex)
            scale = resul/tagres
            if scale>1: continue
            resized_image = cv2.resize(trans_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(self.savedir, f'SC_{pyindex}', imname), resized_image)
            rw = resized_image.shape[1]  if resized_image.shape[0]>resized_image.shape[1] else resized_image.shape[0] 
            if rw<250: break
        return 1



if __name__ == '__main__':

    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')
    
    workdir = cfg['workdir']
    pyscale = cfg['pyscale']
    extension = cfg['test_extension']
    baseresultion = cfg['baseresultion']
    tcfg = cfg['processtestimg-10']
    fine_pysacle = cfg['fine_pysacle']
    savedir = os.path.join(workdir, tcfg['save_path'])
    img_dir = os.path.join(savedir, f'NTest')
    for pyindex in pyscale: os.makedirs(os.path.join(savedir, f'SC_{pyindex}'), exist_ok=True)
    os.makedirs(os.path.join(savedir, f'Fine_sc{fine_pysacle}'), exist_ok=True)

    imlist = list_files_with_extension(os.path.join(savedir, f'NTest'), extension)
    dataset = Get_Img(img_dir, imlist, pyscale, baseresultion, cfg['fine_cutsize'], fine_pysacle, savedir)
    # test_load = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=1)

    for da in tqdm.tqdm(dataset):
        pass
        

    for pyindex in pyscale:
        imlist = os.listdir(os.path.join(os.path.join(savedir, f'SC_{pyindex}')))
        print(pyindex, len(imlist))



