import os
from osgeo import gdal
from utile import *


# input_files = ["input1.tif", "input2.tif", "input3.tif"]


inputdir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/hebei/bigbase'
imlist = list_files_with_extension(inputdir, ['.tif'])
input_files = []
for imname in imlist:
    input_files.append(os.path.join(inputdir, imname))


# 创建 VRT 文件
vrt_options = gdal.BuildVRTOptions(srcNodata=0)  # 设置 NoData 值为 0
vrt = gdal.BuildVRT("output.vrt", input_files, options=vrt_options)

# 关闭并保存文件
vrt = None
