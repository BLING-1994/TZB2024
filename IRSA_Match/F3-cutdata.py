import geopandas as gpd
import rasterio
from rasterio.mask import mask
import json

# 输入文件路径
vector_file = "/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/Output/Vec-0/21_SAT_196.png.geojson"  # 矢量文件路径，例如 Shapefile
raster_file = "/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/output.vrt"  # 栅格文件路径，例如 GeoTIFF
output_file = "21_SAT_196.tif"  # 裁剪后的栅格输出路径


# 步骤 1: 使用 geopandas 读取矢量文件
gdf = gpd.read_file(vector_file)

# 确保矢量和栅格文件的坐标系一致
with rasterio.open(raster_file) as src:
    raster_crs = src.crs
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

# 步骤 2: 将矢量转换为 GeoJSON 格式
shapes = [json.loads(gdf.geometry.to_json())["features"][0]["geometry"]]

# 步骤 3: 使用 rasterio 打开栅格文件
with rasterio.open(raster_file) as src:
    # 步骤 4: 使用 rio.mask.mask 进行裁剪
    out_image, out_transform = mask(src, shapes, crop=True)
    out_meta = src.meta.copy()

# 更新元数据以匹配裁剪结果
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

# 步骤 5: 将裁剪结果保存为新的文件
with rasterio.open(output_file, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"裁剪后的栅格已保存到: {output_file}")
