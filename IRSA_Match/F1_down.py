import os
from osgeo import gdal
import tqdm
def progress_callback(progress, message, user_data):
    print(f"Progress: {progress*100:.2f}%, Message: {message}")
    return 1  # 返回 1 继续处理, 返回 0 中止处理

def downsample_with_translate(input_path, output_path, scale_factor):
    """
    使用 gdal.Translate 下采样栅格数据。
    
    :param input_path: 输入栅格文件路径
    :param output_path: 输出栅格文件路径
    :param scale_factor: 缩小比例（> 1 表示下采样）
    """
    # 打开输入文件
    src = gdal.Open(input_path)
    
    # 获取原始分辨率
    original_width = src.RasterXSize
    original_height = src.RasterYSize

    # 计算新的宽度和高度
    new_width = int(original_width / scale_factor)
    new_height = int(original_height / scale_factor)

    # 配置 Translate 参数
    translate_options = gdal.TranslateOptions(
        width=new_width,
        height=new_height,
        resampleAlg="average",  # 重采样方法
        callback=progress_callback
    )
    
    # 执行下采样并保存结果
    gdal.Translate(output_path, src, options=translate_options)


scale = 8  # 缩小比例0.5->4m

inputdir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/JL_2023_4326'
outdir = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/JL_2023_4326_down8'
os.makedirs(outdir, exist_ok=True)

sdsd = []
for imname in os.listdir(inputdir):
    sdfim = imname.split('.')[-1]
    if sdfim=='tif':
        sdsd.append(imname)

for imname in tqdm.tqdm(sdsd):

    downsample_with_translate(os.path.join(inputdir, imname), os.path.join(outdir, imname), scale)
