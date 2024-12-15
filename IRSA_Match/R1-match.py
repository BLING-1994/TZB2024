import os
from R11search import findimgmatch
from R12croasematch_new import init_model, singelpicfind
from R13finematch_new import finematch
from utile import *
import geopandas as gpd
import logging
import time



if __name__ == '__main__':
    import sys

    cfg = load_config('config/hebei.yaml')
    workdir = cfg['workdir']
    device = cfg['device']
    pyscale = 3

    local_time = time.localtime(time.time())
    formatted_time = time.strftime('%d-%H:%M:%S', local_time)

    os.makedirs(os.path.join(cfg['workdir'], cfg['output_log']), exist_ok=True)
    os.makedirs(os.path.join(cfg['workdir'], cfg['output_geojson']), exist_ok=True)
    # # 区域读取
    boxshppath = os.path.join(workdir, cfg['buildbox-01']['save_path'], f'range_py{pyscale}.shp') 
    rangegdf = gpd.read_file(boxshppath)

    # 去除已经检索到区域
    outvecdir = os.path.join(cfg['workdir'], cfg['output_geojson'])
    testdir = os.path.join(workdir, cfg['processtestimg-10']['save_path'], f'SC_{pyscale}')
    outlist = list_files_with_extension(outvecdir, '.geojson')
    testlist = list_files_with_extension(testdir, cfg['test_extension'])
    datalist = [item for item in testlist if item not in outlist]
    datalistf = []
    # for search_name in datalist:
    #     if '-' in search_name:
    #         if '0-' not in search_name: continue
    #     datalistf.append(search_name)
    
    # datalistf = ['0-11_SAT_196.png']
        
    serachallinfo = findimgmatch(cfg, pyscale, datalist)
    # 匹配 同时创建多个进程加速完成
    model = init_model(cfg['genfeature-02']['gimpath'], device)

    # log_stream = IRSALogger(os.path.join(cfg['workdir'], cfg['output_log'], 'Shell_' + f'SC{pyscale}_{formatted_time}.log'), fmt='%(asctime)s  - %(levelname)s: : %(message)s')
    # sys.stdout = log_stream
    # sys.stderr = log_stream

    for search_name, search_list in serachallinfo.items():
        # # 判断是否已经检索到
        # if '-' in search_name:
        #     if '0-' not in search_name: continue
        
        singelpicfind(model, search_name, pyscale, cfg, rangegdf, search_list, finematch)
    
    

