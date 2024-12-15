import rasterio
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt

# 1. 加载 Shapefile，假设 Shapefile 中有一个矩形 Polygon
shp_path = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/tempshp/保定定州cut.shp'  # 这里是你的 Shapefile 文件路径
gdf = gpd.read_file(shp_path)

# 2. 确保 Shapefile 的 CRS 与栅格的 CRS 一致
# 读取栅格数据
raster_path = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/testdata/hebei/bigbase/warp_保定定州.TIF'
with rasterio.open(raster_path) as src:
    crs_raster = src.crs

# 如果 Shapefile 的 CRS 与栅格的 CRS 不一致，需要进行 CRS 转换
if gdf.crs != crs_raster:
    gdf = gdf.to_crs(crs_raster)

# 3. 获取矩形多边形（假设 Shapefile 中的多边形是矩形）
# 如果 Shapefile 中有多个多边形，可以根据需要选择合适的一个
polygon = gdf.geometry.iloc[0]  # 选择第一个矩形

# 4. 使用Rasterio裁切栅格
with rasterio.open(raster_path) as src:
    # 将矩形多边形转换为 GeoJSON 格式
    geoms = [polygon.__geo_interface__]
    
    # 使用 mask 对栅格进行裁切
    out_image, out_transform = mask(src, geoms, crop=True)
    
    # 获取裁切后的栅格的元数据
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # 5. 保存裁切后的栅格
    output_raster_path = 'path_to_output_raster.tif'  # 输出的裁切后的栅格路径
    with rasterio.open(output_raster_path, 'w', **out_meta) as dest:
        dest.write(out_image)

# 6. 可视化裁切后的栅格
plt.imshow(out_image[0], cmap='gray')
plt.show()
