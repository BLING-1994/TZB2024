import os
from osgeo import gdal
from utile import *


if __name__ == '__main__':

    # todo 需要考虑旋转和超出边界的情况

    import sys
    # cfg = load_config(sys.argv[1])
    cfg = load_config('config/hebei.yaml')

    ntestpath = os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],'NTest')
    outlist = list_files_with_extension(ntestpath, cfg['test_extension'])

    for imname in outlist:
        npypath = os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],'NTest', imname + '.npy')
        if os.path.exists(npypath) is False:
            scdataset = gdal.Open(os.path.join(cfg['workdir'], cfg['processtestimg-10']['save_path'],'NTest', imname))
            scwidth = scdataset.RasterXSize  # 栅格矩阵的列数
            scheight = scdataset.RasterYSize
            cs = np.array([0, 0, scwidth, scheight])
            np.save(npypath, cs)