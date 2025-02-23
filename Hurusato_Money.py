# -*- coding: utf-8 -*-
# 以下は日本全国のデータを扱いますが、処理中の都道府県が切り替わるタイミングでprintするようにし、
# カラースケールの最大値を3B（約30億）に固定することで、極端な高額データに引きずられず、
# 全体的な色分布を見やすくします。

import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import numpy as np
import subprocess

script_path = './Geo-getter.py'

subprocess.run(['python', script_path])

# ---------------------------------------
# ユーティリティ関数定義
# ---------------------------------------
def convert_to_5_digit(code):
    # 市区町村団体コード(6桁) -> 5桁コードに変換 (後ろ1桁を除く)
    return code[:-1]

def process_designated_cities(gdf, df):
    # 政令指定都市の区をまとめて1つのジオメトリにする関数
    # union_all() を使用 (GeoPandasの新バージョン対応)
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

# ---------------------------------------
# メイン処理
# ---------------------------------------

# CSVファイルパスを指定（適宜変更）
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

# 結果を格納するリスト
merged_list = []

for pref_name in unique_prefectures:
    # 都道府県が変わるごとにprint
    print(f"{pref_name} の処理を開始します。")

    # 該当都道府県のみ抽出
    df_pref = df[df['都道府県'] == pref_name]

    # 該当都道府県のジオメトリ抽出
    gdf_pref = gdf_init[gdf_init['N03_001'] == pref_name].copy()

    if gdf_pref.empty:
        # シェープファイル上に該当都道府県がない場合はスキップ
        continue

    gdf_pref = gdf_pref[['N03_007', 'geometry']]
    gdf_pref['code'] = gdf_pref['N03_007'].replace('000nan', pd.NA).astype(str).str.zfill(5)

    # 政令指定都市の区を統合
    gdf_pref = process_designated_cities(gdf_pref, df_pref)

    # 同じcodeでdissolve
    gdf_pref = gdf_pref.dissolve(by='code', aggfunc='first').reset_index()
    gdf_pref = gdf_pref.to_crs(epsg=4326)

    # ジオメトリ簡略化
    gdf_pref['geometry'] = gdf_pref['geometry'].simplify(tolerance=0.001)

    # マージ処理
    merged_gdf_pref = gdf_pref.merge(df_pref, left_on='code', right_on='市区町村団体コード_5桁', how='left')
    merged_gdf_pref['ふるさと納税合計額'] = merged_gdf_pref['ふるさと納税合計額'].fillna(0)
    merged_gdf_pref = merged_gdf_pref[merged_gdf_pref['code'] != '00nan']

    # 処理結果をリストに追加
    merged_list.append(merged_gdf_pref)

# 全都道府県を結合
if merged_list:
    merged_gdf = gpd.GeoDataFrame(pd.concat(merged_list, ignore_index=True), crs="EPSG:4326")
else:
    # データがなければ終了
    print("該当するデータがありません。処理を終了します。")
    exit()

# カラースケールの最大値を3Bに固定
max_color_val = 3_000_000_000  # 3B

# GeoJSON変換
merged_gdf_json = json.loads(merged_gdf.to_json())
for i, feature in enumerate(merged_gdf_json['features']):
    feature['id'] = merged_gdf.iloc[i]['code']

# 中心点計算（全国表示）
bounds = merged_gdf.total_bounds
center_lon = (bounds[0] + bounds[2]) / 2
center_lat = (bounds[1] + bounds[3]) / 2

# Choroplethマップ作成（range_colorで最大値を3Bに固定）
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
    range_color=(0, max_color_val)  # 最大値を3Bに固定
)

fig.update_layout(height=800, margin={"r":0,"t":30,"l":0,"b":0})
fig.show()

# 結果をHTMLファイルとして出力
fig.write_html("furusato_nouzei_map.html")

print("処理が完了しました。")
