import os
from osgeo import gdal

imdir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/baseimage'
for imname in os.listdir(imdir):

    # 打开栅格文件
    dataset = gdal.Open(os.path.join(imdir, imname), gdal.GA_Update)

    # 删除金字塔层
    dataset.BuildOverviews(None)

    # 关闭数据集
    dataset = None