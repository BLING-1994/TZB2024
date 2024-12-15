
import os
import numpy as np
import cv2
from utile import *
import tqdm
# 注意  是否需要去除一行
from torch.utils.data import Dataset



def splitimg(img, save_path, save_name):
    h, w, _ = img.shape
    cutlist = []
    if h>w: # 垂直切分
        if h/w<1.7: 
            cv2.imwrite(os.path.join(save_path, save_name), img)
            return
        part = np.int32(np.round(h/w))
        for i in range(part):
            start_x = 0
            start_y = i*w
            end_x = w
            end_y = (i+1)*w
            if end_y>h:
                end_y = h
                start_y = h - w
            cutlist.append([start_x, start_y, end_x, end_y])   
    else:
        if w/h<1.7: 
            cv2.imwrite(os.path.join(save_path, save_name), img)
            return
        part = np.int32(np.round(w/h))
        for i in range(part):
            start_x = i*h
            start_y = 0
            end_x = (i+1)*h
            end_y = h
            if end_x>w:
                end_x = w
                start_x = w - h
            cutlist.append([start_x, start_y, end_x, end_y])

    for index, cut in enumerate(cutlist):
        imgcut = img[cut[1]:cut[3], cut[0]:cut[2]]
        cs = np.array([-cut[0], -cut[1], w - cut[0], h - cut[1]])
        np.save(os.path.join(save_path, f'{index}-{save_name}.npy'), cs)
        cv2.imwrite(os.path.join(save_path, f'{index}-{save_name}'), imgcut)


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
        trans_image = roateimg2img(os.path.join(self.img_dir, imname))
        # os.makedirs(os.path.join(self.savedir, f'OTest'), exist_ok=True)
        # cv2.imwrite(os.path.join(self.savedir, f'OTest', imname), trans_image)
        # trans_image = cv2.rotate(trans_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # os.makedirs(os.path.join(self.savedir, f'NTest'), exist_ok=True)
        splitimg(trans_image, os.path.join(self.savedir, f'NTest'), imname)
        return 1



if __name__ == '__main__':

    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')
    img_dir = cfg['test_dir']
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
        




