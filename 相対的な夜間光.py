import ee
import os
import requests
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

def initialize_gee():
    """
    GEEを認証・初期化する関数。
    """
    try:
        ee.Authenticate()  # 必要に応じてコメントアウト
        ee.Initialize(project='stem-443415')
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        exit(1)

def download_tiff(image, geometry, folder_path, file_name, scale):
    """
    GEEのImageをGeoTIFFとしてローカルに保存する。
    """
    try:
        url = image.getDownloadURL({
            'scale': scale,
            'region': geometry.getInfo()['coordinates'],
            'crs': 'EPSG:4326',
            'format': 'GeoTIFF'
        })

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

def merge_to_prefecture_level(shapefile_path):
    """
    Shapefileを都道府県レベルに統合する。
    """
    try:
        regions = gpd.read_file(shapefile_path)
        if 'N03_001' not in regions.columns:
            raise ValueError("地域名を示すカラム N03_001 が見つかりません。")

        regions = regions.dissolve(by='N03_001').reset_index()
        regions = regions.rename(columns={"N03_001": "region_name"})
        print("Shapefile merged to prefecture level.")
        return regions
    except Exception as e:
        print(f"Error merging Shapefile: {e}")
        return None

def tiff_to_csv(tiff_path, shapefile_path, csv_output_path):
    """
    TIFFファイルをzonal_statsで集計し、CSVとして出力する。
    """
    try:
        regions = merge_to_prefecture_level(shapefile_path)
        if regions is None:
            raise ValueError("Shapefile merging failed.")

        stats = zonal_stats(regions, tiff_path, stats=["mean", "sum", "min", "max"], nodata=-9999)
        regions['mean_light'] = [stat['mean'] for stat in stats]
        regions['sum_light'] = [stat['sum'] for stat in stats]
        regions['min_light'] = [stat['min'] for stat in stats]
        regions['max_light'] = [stat['max'] for stat in stats]

        # 相対的な夜間光データを計算
        regions['relative_mean_light'] = regions['mean_light'] / regions['mean_light'].max()
        regions['relative_sum_light'] = regions['sum_light'] / regions['sum_light'].max()

        if not os.path.exists(os.path.dirname(csv_output_path)):
            os.makedirs(os.path.dirname(csv_output_path))
        regions[['region_name', 'relative_mean_light', 'relative_sum_light']].to_csv(csv_output_path, index=False)
        print(f"CSV saved at: {csv_output_path}")
    except Exception as e:
        print(f"Error converting TIFF to CSV: {e}")

def create_yearly_image(snippet, band, geometry, year, method='mean'):
    """
    start_date ~ end_date の期間を対象に、指定されたデータセットの月次データをまとめて1枚のImageにする。
    """
    try:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        dataset = (ee.ImageCollection(snippet)
                   .filter(ee.Filter.date(start_date, end_date))
                   .filterBounds(geometry)
                   .select(band))
        if method == 'mean':
            return dataset.mean()
        elif method == 'median':
            return dataset.median()
        else:
            return dataset.mean()
    except Exception as e:
        print(f"Error creating yearly image: {e}")
        return None

def main():
    initialize_gee()

    geometry = ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12])
    scale = 1000
    shp_path = './N03-20220101_GML/N03-22_220101.shp'
    tiff_folder = './downloaded_data'
    csv_folder = './GEE_csv_data'

    years = range(2013, 2016)  # 処理したい年の範囲

    for year in years:
        print(f"Processing year: {year}")

        if year < 2014:
            snippet = 'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS'
            band = 'avg_vis'
        else:
            snippet = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
            band = 'avg_rad'

        yearly_image = create_yearly_image(snippet, band, geometry, year)

        if yearly_image is None:
            continue

        file_tag = f"{year}_relative"
        tiff_file_path = os.path.join(tiff_folder, f"{file_tag}.tif")
        downloaded_path = download_tiff(yearly_image, geometry, tiff_folder, file_tag, scale)

        if downloaded_path:
            csv_file_path = os.path.join(csv_folder, f"{file_tag}.csv")
            tiff_to_csv(downloaded_path, shp_path, csv_file_path)

if __name__ == "__main__":
    main()
