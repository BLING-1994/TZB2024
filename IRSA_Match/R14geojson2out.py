import os
import geopandas as gpd


impath = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/Output/Vec'

alllist = []
for geoname in os.listdir(impath):
    searchname = geoname.replace('.geojson', '')
    gdf = gpd.read_file(os.path.join(impath, geoname))
    first_polygon = gdf.geometry.iloc[0]

    coordinates = [(round(lon, 7), round(lat, 7)) for lon, lat in first_polygon.exterior.coords]

    outline = searchname
    for pt in coordinates[:4]:
        outline += f' {pt[0]} {pt[1]}'
    alllist.append(outline)

with open("RESULT.txt", "w") as file:
    for line in alllist:
        file.write(line + "\n")



