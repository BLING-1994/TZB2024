#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo script for performing OmniGlue inference."""

import os
import sys
import time
import numpy as np
import torch
import tqdm
import pickle
import cv2
from Dinov2.Dino_feature import Dinofeature
from PIL import Image
import faiss
from shapely import Polygon
from Dinov2.dino_extract import DINOExtract
from utile import *
from torch.utils.data import Dataset



class Get_Img(Dataset):
    def __init__(self, img_dir, imlist):
        self.img_dir = img_dir
        self.imlist = imlist
   
    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, i):
        name = self.imlist[i]
        image = cv2.imread(os.path.join(self.img_dir, name))[:,:,::-1].transpose(2,0,1) # RGB
        image = torch.from_numpy(np.ascontiguousarray(image / 255.0)).float()
        sample = {'image':image, 'name': name}
        return sample

def init_model(dinopath, device='cuda'):
    dinov2model = DINOExtract(dinopath, use_half=False, device=device, fixsize=None)
    return dinov2model


def findimgmatch(cfg, usepysacle, datalist):
    print("feature start")
    workdir = cfg['workdir']
    # pyscale = cfg['pyscale']# 5层金字塔
    img_dir = cfg['img_dir']
    device = cfg['device']
    picbase_dir = os.path.join(workdir, cfg['processtestimg-10']['save_path'], f'SC_{usepysacle}')

    # 加载向量库
    with open(os.path.join(workdir, cfg['genfeature-02']['save_path'], f'SC_{usepysacle}/BeseIndex.pkl'), 'rb') as file:
        BaseIndexes = pickle.load(file)

    with open(os.path.join(workdir, cfg['genfeature-02']['save_path'], f'SC_{usepysacle}/BeseInfo.pkl'), 'rb') as file:
        BeseInfo = pickle.load(file)

    all_descriptors = np.vstack(BaseIndexes).astype('float32')
    all_descriptors = all_descriptors / np.linalg.norm(all_descriptors, axis=1, keepdims=True)
    print(all_descriptors.shape)
    d = all_descriptors.shape[1]  # Dimension of the feature vectors
    faissindex = faiss.IndexFlatL2(d)  # L2 distance index
    faissindex.add(all_descriptors)

    top_k = faissindex.ntotal
    print("===========", faissindex.ntotal)
    
    model = init_model(cfg['genfeature-02']['dinopath'], device)
    picbase_dirlist = datalist

    dataset = Get_Img(picbase_dir, picbase_dirlist)
    test_load = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4)

    outdata = {}
    with torch.no_grad():
        for data in tqdm.tqdm(test_load):
            image_ts = data['image'].to(device)
            imname = data['name'][0]
            mean_fe = model(image_ts).cpu().numpy()[0]
            
            sefe = np.array([mean_fe])
            sefe = sefe / np.linalg.norm(sefe, axis=1, keepdims=True)
            D, I = faissindex.search(sefe, k=top_k)    
            outdata[imname] = []
            for i in range(top_k):
                similar_image_index = I[0][i]#  // all_descriptors.shape[0]
                simindex= BeseInfo[similar_image_index]
                outdata[imname].append(int(simindex))
            # print(imname, len(outdata[imname]))
    return outdata


if __name__ == "__main__":
    # main()

    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')
    workdir = cfg['workdir']

    usepysacle = 3

    datalist = os.listdir(os.path.join(workdir, cfg['processtestimg-10']['save_path'], f'SC_{usepysacle}'))
    
    findimgmatch(cfg, usepysacle, datalist)
