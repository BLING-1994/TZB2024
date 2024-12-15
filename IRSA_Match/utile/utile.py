import yaml
import gzip
import pickle
import io
from pathlib import Path
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
import torchvision.transforms.functional as F

def list_files_with_extension(folder_path, extension, rel=''):
    # 使用 Path 对象来表示目录
    path = Path(folder_path)
    # 遍历目录并筛选出符合后缀条件的文件
    alllist = []
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in extension:
            alllist.append(file.name.replace(rel, ''))
    return alllist

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # 使用 safe_load 安全地加载 YAML
    return config


def tensor2gzsave(tensor,path):
    # 压缩保存
    with gzip.open(path + '.gz', 'wb') as f:
        pickle.dump(tensor, f)

def readgz2tensor(path):
    with gzip.open(path, 'rb') as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor


def getimgbound(img):
    if len(img.shape)==3:
      mask = np.where(np.all(img == [0, 0, 0], axis=-1), 0, 255).astype('uint8')
    else:
      mask = np.where(img==0, 0, 255).astype('uint8')
    non_zero_rows = ~np.all(mask == 0, axis=1)
    non_zero_cols = ~np.all(mask == 0, axis=0)
    mask = mask[non_zero_rows][:, non_zero_cols]
    if len(img.shape)==3:
      img = img[non_zero_rows][:, non_zero_cols, :]
    else:
      img = img[non_zero_rows][:, non_zero_cols]
    imh, imw = mask.shape
    first_x = np.argmax(mask[0,:] == 255)
    first_y = np.argmax(mask[:,0] == 255)
    if first_x==0 or first_y==0:#  or np.abs(first_x_f - first_x)>10 or np.abs(first_y_f - first_y)>10: 
      corners = np.array([[ 0,    0],
                          [   0,  imh-1],
                          [ imw-1, imh-1],
                          [ imw-1,  0]], dtype=np.float32)
      return np.ascontiguousarray(img), corners
    if first_x>first_y:
       s_y = first_y
       s_x = np.argmax(mask[0,::-1] == 255)
       corners = np.array([[0,s_y], [imw -1 - s_x, 0], [imw - 1, imh - 1 - s_y], [s_x, imh-1]], dtype=np.float32)
    else:
       s_x = first_x
       s_y = np.argmax(mask[::-1, 0] == 255)
       corners = np.array([[s_x,0], [imw - 1, s_y], [imw - 1 - s_x, imh - 1], [0, imh-1-s_y]], dtype=np.float32)
    return np.ascontiguousarray(img), corners





DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_METHOD = cv2.USAC_MAGSAC

RANSAC_ZOO = {
    "RANSAC": cv2.RANSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}



def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized




def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(
            lambda x: int(x // dfactor * dfactor),
            image.shape[-2:]))
    image = F.resize(image, size=size_new)
    return image



def getioubox(mkpts1, mkpts2, orgpts, mconf):
    # 计算单应性矩阵
    try:
        H, mask = cv2.findHomography(mkpts2, mkpts1, 
                            method=DEFAULT_RANSAC_METHOD,
                            ransacReprojThreshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
                            confidence=DEFAULT_RANSAC_CONFIDENCE,
                            maxIters=DEFAULT_RANSAC_MAX_ITER,)
    except:
        return None, None, 0,0, None

    

    if H is None: return None, None, 0, 0 , None
    # print('H', H)
    has_true = np.sum(mask)

    score = np.mean(mconf[mask])
    # score = np.mean(np.sort(mconf[mask])[-4:])
    dst = cv2.perspectiveTransform(orgpts, H)[:,0,:]
    # dst = np.round(dst)
    # print(dst)
    polygon2 = Polygon(dst)
    if polygon2.is_simple is False:
        return None, None, 0, 0, None
    # print(dst)
    xlist = [d[0] for d in dst]
    ylist = [d[1] for d in dst]
    sx0, sy0 = min(xlist), min(ylist)
    sx1, sy1 = max(xlist), max(ylist)
    polygon1 = Polygon([(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)])

    # 计算两个 Polygon 的交集区域
    intersection_area = polygon1.intersection(polygon2).area
    # 计算两个 Polygon 的并集区域
    union_area = polygon1.union(polygon2).area
    # 计算 IoU
    iou = intersection_area / (union_area + 1e-10)
    return iou, [sx0, sy0, sx1, sy1], score, has_true, H

def tran2box(orgpts, H):
    dst = cv2.perspectiveTransform(orgpts, H)[:,0,:]
    dst = np.round(dst)
    xlist = [d[0] for d in dst]
    ylist = [d[1] for d in dst]
    sx0, sy0 = min(xlist), min(ylist)
    sx1, sy1 = max(xlist), max(ylist)
    return [sx0, sy0, sx1, sy1]