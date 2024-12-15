import os
from osgeo import gdal
from utile import *
import time
import subprocess
from multiprocessing import Pool


def buildoverviews(indata):
    path, overview_levels = indata
    dataset = gdal.Open(path, gdal.GA_Update)
    print(path,os.path.exists(path), dataset)
    overview_count = dataset.GetRasterBand(1).GetOverviewCount()
    if overview_count>0:
        print(os.path.basename(path), 'overview_count is ', overview_count)
        return
    # 创建金字塔
    print(overview_levels)
    subprocess.run(['gdaladdo', '-r', 'nearest', '-ro', '--config', 'BIGTIFF_OVERVIEW', 'IF_NEEDED', path] + overview_levels)

if __name__ == '__main__':
    # path = '/irsa/河北省变化监测/2mcog/衡水.tif'
    # dataset = gdal.Open(path, gdal.GA_Update)
    # print(path,os.path.exists(path), dataset)

    import sys
    print(sys.argv[1])
    cfg = load_config(sys.argv[1])

    img_dir = cfg['img_dir']
    extension = cfg['extension']
    imlist = list_files_with_extension(img_dir, extension)[:1]
    workdir = cfg['workdir']
    pyscale = cfg['pyscale']# 5层金字塔
    overview_levels = []
    for n in pyscale:
        if n==0: continue
        overview_levels.append(str(2**n))

    alltask = []
    for imname in imlist:
        alltask.append((os.path.join(img_dir, imname), overview_levels))
    
    st = time.time()
    with Pool(2) as p:  # 设置进程池大小，4个进程
        p.map(buildoverviews, alltask)
    print(time.time() - st)

        
        

# gdaladdo -r nearest  testdata/JL_2023_4326/warp_JL1KF01C_PMSR3_20240718110924_200283670_101_0076_001_L1.tif  2 4 8 16
# gdaladdo -r none JL1KF01B_PMSL1_20220317104342_200078402_102_0041_001_L3D_PSH.tif