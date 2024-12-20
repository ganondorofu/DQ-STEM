# -*- coding: utf-8 -*-
# 以下は座標変換を入れた修正コード例

import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import numpy as np
import subprocess
import os
import rasterio
import base64
from io import BytesIO
from PIL import Image
from pyproj import Transformer

script_path = './Geo-getter.py'
subprocess.run(['python', script_path])

def convert_to_5_digit(code):
    # 市区町村団体コード(6桁) -> 5桁コードに変換 (後ろ1桁を除く)
    return code[:-1]

def process_designated_cities(gdf, df):
    # 政令指定都市の区をまとめて1つのジオメトリにする関数
    city_codes = df[df['市区町村団体コード_5桁'].str[2:5] == '100']['市区町村団体コード_5桁'].unique()
    for city_code in city_codes:
        ward_data = gdf[(gdf['code'].str[:3] == city_code[:3]) & (gdf['code'].str[3:5] != '00')]
        if not ward_data.empty:
            city_geometry = ward_data['geometry'].union_all()
            city_data = pd.DataFrame({
                'code': [city_code],
                'geometry': [city_geometry]
            })
            gdf = gdf[~gdf.index.isin(ward_data.index)]
            gdf = pd.concat([gdf, gpd.GeoDataFrame(city_data, crs=gdf.crs)], ignore_index=True)
    return gdf

# CSVファイルパスを指定
csv_path = './data/exported_data.csv'

# シェープファイル（全国市区町村界）をGeoDataFrameとして読み込み
gdf_init = gpd.read_file("./N03-20220101_GML/N03-22_220101.shp", encoding='shift-jis')

# CSVデータの読み込み
df = pd.read_csv(
    csv_path,
    header=None,
    names=['市区町村団体コード', '都道府県', '市区町村', 'ふるさと納税合計額'],
    dtype={'市区町村団体コード': str},
    skip_blank_lines=True,
    encoding='utf-8'
)

# 空の市区町村団体コード行をドロップ
df = df.dropna(subset=['市区町村団体コード'])

# 市区町村団体コードを6桁ゼロ埋め
df['市区町村団体コード'] = df['市区町村団体コード'].str.zfill(6)

# ふるさと納税合計額を数値型に変換し、欠損を0で埋める
df['ふるさと納税合計額'] = pd.to_numeric(df['ふるさと納税合計額'], errors='coerce', downcast='float').fillna(0)

# 5桁コードを作成
df['市区町村団体コード_5桁'] = df['市区町村団体コード'].apply(convert_to_5_digit)

# 都道府県ごとに処理を行うため、ユニークな都道府県名を取得
unique_prefectures = df['都道府県'].dropna().unique()

merged_list = []
for pref_name in unique_prefectures:
    print(f"{pref_name} の処理を開始します。")

    df_pref = df[df['都道府県'] == pref_name]
    gdf_pref = gdf_init[gdf_init['N03_001'] == pref_name].copy()

    if gdf_pref.empty:
        continue

    gdf_pref = gdf_pref[['N03_007', 'geometry']]
    gdf_pref['code'] = gdf_pref['N03_007'].replace('000nan', pd.NA).astype(str).str.zfill(5)
    gdf_pref = process_designated_cities(gdf_pref, df_pref)
    gdf_pref = gdf_pref.dissolve(by='code', aggfunc='first').reset_index()
    gdf_pref = gdf_pref.to_crs(epsg=4326)
    gdf_pref['geometry'] = gdf_pref['geometry'].simplify(tolerance=0.001)
    merged_gdf_pref = gdf_pref.merge(df_pref, left_on='code', right_on='市区町村団体コード_5桁', how='left')
    merged_gdf_pref['ふるさと納税合計額'] = merged_gdf_pref['ふるさと納税合計額'].fillna(0)
    merged_gdf_pref = merged_gdf_pref[merged_gdf_pref['code'] != '00nan']
    merged_list.append(merged_gdf_pref)

if merged_list:
    merged_gdf = gpd.GeoDataFrame(pd.concat(merged_list, ignore_index=True), crs="EPSG:4326")
else:
    print("該当するデータがありません。処理を終了します。")
    exit()

max_color_val = 3_000_000_000

merged_gdf_json = json.loads(merged_gdf.to_json())
for i, feature in enumerate(merged_gdf_json['features']):
    feature['id'] = merged_gdf.iloc[i]['code']

bounds = merged_gdf.total_bounds
center_lon = (bounds[0] + bounds[2]) / 2
center_lat = (bounds[1] + bounds[3]) / 2

fig = px.choropleth_mapbox(
    merged_gdf,
    geojson=merged_gdf_json,
    locations='code',
    color='ふるさと納税合計額',
    hover_name='市区町村',
    hover_data={'ふるさと納税合計額': True, '市区町村': True, '都道府県': True},
    color_continuous_scale='Viridis',
    title='日本全国のふるさと納税合計額 (最大3B固定)',
    labels={'ふるさと納税合計額': '合計額'},
    mapbox_style="carto-positron",
    center={"lat": center_lat, "lon": center_lon},
    zoom=4,
    range_color=(0, max_color_val)
)

fig.update_layout(height=800, margin={"r":0,"t":30,"l":0,"b":0})

# 夜間光データ処理
image_dir = './downloaded_data'
def list_tiff_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

tiff_files = list_tiff_files(image_dir)
if tiff_files:
    tiff_file = tiff_files[0]
    with rasterio.open(tiff_file) as src:
        data = src.read(1)
        if src.nodata is not None:
            data[data == src.nodata] = 0
        data = np.log1p(data)
        data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.astype(np.uint8)
        img = Image.fromarray(data, mode='L')

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        png_data = buffer.read()
        b64_str = base64.b64encode(png_data).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"

        # 元の座標系を確認
        raster_crs = src.crs
        left, bottom, right, top = src.bounds
        
        # 座標がEPSG:4326(経緯度)でない場合は変換
        if raster_crs and raster_crs != "EPSG:4326":
            # Transformerを使ってEPSG:4326へ変換
            transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
            # left,bottom,right,topは (経度,緯度) 順で渡す必要がある。
            (left, bottom) = transformer.transform(left, bottom)
            (right, top) = transformer.transform(right, top)

        # 画像レイヤー追加
        fig.update_layout(
            mapbox={
                "layers": [
                    {
                        "sourcetype": "image",
                        "source": data_url,
                        "coordinates": [
                            [left, top],
                            [right, top],
                            [right, bottom],
                            [left, bottom]
                        ],
                        "opacity": 0.6
                    }
                ]
            }
        )
else:
    print("TIFFファイルが見つかりませんでした。")

fig.write_html("furusato_nouzei_map_with_nightlight.html")
fig.show()
print("処理が完了しました。")
