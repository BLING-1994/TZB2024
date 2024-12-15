
import os
from osgeo import gdal, osr
import tqdm
from  multiprocessing import Pool


def prjis3857(filepath):
    print(filepath)
    # 打开栅格文件
    dataset = gdal.Open(filepath)
    # 获取投影信息（WKT格式）
    proj = dataset.GetProjection()
    print(proj)
    # 创建 SpatialReference 对象并导入投影信息
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    # 检查 EPSG 代码
    if srs.IsProjected():
        epsg_code = srs.GetAttrValue('AUTHORITY', 1)
        if epsg_code == '3857':
            print("该栅格使用 EPSG:3857 坐标系 (Web Mercator)")
            return True
        else:
            print(f"该栅格使用的 EPSG 代码是: {epsg_code}")
            return False
    else:
        print("该栅格没有投影或不使用投影坐标系")
        return None


# 定义进度回调函数
def progress_callback(progress, message, user_data):
    print(f"Progress: {progress*100:.2f}%, Message: {message}")
    return 1  # 返回 1 继续处理, 返回 0 中止处理

def translate_to_cog(input_filename, output_filename):
    # 打开原始数据集
    # src_ds = gdal.Open(input_filename)
    outputdir = os.path.dirname(output_filename)
    os.makedirs(outputdir, exist_ok=True)
    outputname = os.path.basename(output_filename)
    print(input_filename)

    wrap_path = os.path.join(outputdir, 'warp_' + outputname)

    wopt = gdal.WarpOptions(dstSRS='EPSG:4326', resampleAlg='near', callback=progress_callback, callback_data='.')
    # 设置进度回调函数
    # gdal.TermProgress = progress_callback
    print('start warp')
    # todo 判断input的坐标系
    gdal.Warp(wrap_path, input_filename, options=wopt)





def poolrun(src_path, dst_path):
    translate_to_cog(src_path, dst_path)

if __name__ == '__main__':

   
    # bigpath = '/irsa_public/大图幅测试数据/JL_2023'
    # save_path = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/JL_2023_4326'
    # os.makedirs(save_path, exist_ok=True)
    # filterhz = ['.tif', '.tiff', '.img']
    # for tifname in os.listdir(bigpath):
    #     _, hz = os.path.splitext(tifname)
        
    #     if hz.lower() not in filterhz: continue

    #     translate_to_cog(os.path.join(bigpath, tifname), os.path.join(save_path, tifname))




    inpath = '/irsa/河北省变化监测/DOM0508-0514zz/保定1.tif'
    outpath = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/hebei/bigtest/保定1.tif'

    translate_to_cog(inpath, outpath)