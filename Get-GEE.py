import ee
import os
import requests
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd

ee.Authenticate()  # 認証を必要に応じてコメントアウト
ee.Initialize(project='stem-443415')
print("Google Earth Engine initialized successfully.")

# GEEのデータロード
def load_data(snippet, from_date, to_date, geometry, band):
    try:
        dataset = ee.ImageCollection(snippet).filter(
            ee.Filter.date(from_date, to_date)).filterBounds(geometry).select(band)
        data_list = dataset.toList(dataset.size())
        num_data = dataset.size().getInfo()
        print(f"Number of datasets found: {num_data}")
        return num_data, data_list
    except Exception as e:
        print(f"Error loading data: {e}")
        return 0, None

# ローカルに保存するためのエクスポート
def save_to_local(image, geometry, folder_path, file_name, scale):
    try:
        # エクスポート用URL生成
        url = image.getDownloadURL({
            'scale': scale,
            'region': geometry.getInfo()['coordinates'],
            'crs': 'EPSG:4326',
            'format': 'GeoTIFF'
        })

        # データのダウンロード
        print(f"Downloading {file_name} ...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, f"{file_name}.tif")
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Saved: {file_path}")
            return file_path
        else:
            print(f"Failed to download {file_name}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred while downloading {file_name}: {e}")
        return None

# TIFFデータをCSV形式に変換
def tiff_to_csv(tiff_path, regions_path, csv_output_path):
    try:
        # 地域データの読み込み（Shapefile）
        regions = gpd.read_file(regions_path)

        # 夜間光データを地域ごとに集計
        stats = zonal_stats(regions, tiff_path, stats=["mean", "sum"])
        regions['mean_light'] = [stat['mean'] for stat in stats]
        regions['sum_light'] = [stat['sum'] for stat in stats]

        # CSVとしてエクスポート
        regions[['region_name', 'mean_light', 'sum_light']].to_csv(csv_output_path, index=False)
        print(f"CSV saved at: {csv_output_path}")
    except Exception as e:
        print(f"Error converting TIFF to CSV: {e}")

# メイン処理
def main():
    # パラメーター設定
    snippet = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
    band = 'avg_rad'
    from_date = '2014-01-01'
    to_date = '2024-06-04'
    geometry = ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12])  # 日本全域
    scale = 1000
    folder_path = './downloaded_data'  # 保存先フォルダ
    regions_path = './N03-20220101_GML/N03-22_220101.shp'  # 地域境界データ（Shapefile）
    csv_output_path = './nightlight_data.csv'  # 出力CSVファイル

    # データをロード
    num_data, data_list = load_data(snippet, from_date, to_date, geometry, band)

    # データのダウンロードと変換
    if num_data > 0 and data_list is not None:
        for i in range(num_data):
            try:
                image = ee.Image(data_list.get(i))
                image_info = image.getInfo()
                file_name = image_info['id'].replace('/', '_')  # ファイル名を生成
                tiff_path = save_to_local(image, geometry, folder_path, file_name, scale)

                # ダウンロード成功時にCSV変換
                if tiff_path:
                    tiff_to_csv(tiff_path, regions_path, csv_output_path)
            except Exception as e:
                print(f"Error processing image {i}: {e}")

if __name__ == "__main__":
    main()
