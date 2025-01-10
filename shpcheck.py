import geopandas as gpd

# Shapefileを読み込み
#regions = gpd.read_file("./N03-20220101_GML/N03-22_220101.shp")
regions = gpd.read_file("./N03-20220101_GML/N03-22_220101.shp")

# 全データの確認
print(regions.head())
