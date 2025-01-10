import ee
import os
import requests
import rasterio
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_gee():
    """
    GEEを認証・初期化する関数。
    """
    try:
        ee.Authenticate()
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
        else:
            print(f"Failed to download {file_name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"An error occurred while downloading {file_name}: {e}")

def download_yearly_data(snippet, band, geometry, year, folder_path, region_name, scale=1000):
    """
    年次データを生成して保存する。
    - モザイクではなく平均値で合成するように修正。
    """
    try:
        dataset = (ee.ImageCollection(snippet)
                   .filter(ee.Filter.calendarRange(year, year, 'year'))
                   .select(band))

        # 平均値で合成
        image = dataset.mean()

        # 必要であればスケール係数を掛ける
        # image = image.multiply(0.1)

        # ファイル名に band をそのまま利用
        file_name = f"{region_name}_Yearly_{year}_{band}"
        download_tiff(image, geometry, folder_path, file_name, scale)
    except Exception as e:
        print(f"Error downloading yearly data for {region_name} in {year}: {e}")

def load_tif(file_path):
    """
    GeoTIFFファイルを読み込む関数。
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 最初のバンドを取得
        profile = src.profile  # GeoTIFF メタデータ
    return data, profile

def apply_correction_model(model, dmsp_data):
    """
    補正モデルを使用してDMSPデータを補正。
    """
    corrected_data = model.predict(dmsp_data.flatten().reshape(-1, 1))
    return corrected_data.reshape(dmsp_data.shape)

def visualize_data_as_boxplot(data_dict):
    """
    データを箱ひげ図として可視化する。
    """
    plt.figure(figsize=(12, 6))
    # data_dict.values() は各年のデータ配列
    # これらをリストにして boxplot に渡す
    sns.boxplot(data=list(data_dict.values()))
    plt.xticks(ticks=range(len(data_dict)), labels=list(data_dict.keys()), rotation=45)
    plt.title("Corrected Nighttime Light Data (2006-2021)")
    plt.xlabel("Year")
    plt.ylabel("Light Intensity")
    plt.tight_layout()
    plt.show()

def main():
    initialize_gee()

    # 日本の地域設定
    japan_geometry = ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12])  # 日本全域
    folder_path = './nighttime_data'
    years = range(2006, 2022)  # 2006年から2021年まで

    # データセット設定
    # DMSP -> 2006 ~ 2013
    # VIIRS -> 2014 ~ 2021
    datasets = {
        'DMSP': {
            'snippet': 'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS',
            'band': 'stable_lights',
            'years': range(2006, 2014)
        },
        'VIIRS': {
            'snippet': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG',
            'band': 'avg_rad',
            'years': range(2014, 2022)
        }
    }

    # 補正モデルの読み込み
    model_path = './models/dmsp_viirs_correction_model.pkl'
    with open(model_path, 'rb') as f:
        correction_model = pickle.load(f)
    print(f"補正モデルを読み込みました: {model_path}")

    # 年次データを補正し、箱ひげ図用のデータを収集
    corrected_data_dict = {}

    for year in years:
        print(f"Processing year: {year}")
        # 年によって DMSP or VIIRS を選択
        if year < 2014:
            dataset = datasets['DMSP']
        else:
            dataset = datasets['VIIRS']

        snippet = dataset['snippet']
        band = dataset['band']
        region_name = "Japan"

        file_name = f"{region_name}_Yearly_{year}_{band}"
        output_path = os.path.join(folder_path, f"{file_name}.tif")

        # データをダウンロード
        download_yearly_data(snippet, band, japan_geometry, year, folder_path, region_name)

        # ダウンロードした GeoTIFF を読み込む
        data, _ = load_tif(output_path)

        # DMSPデータは補正モデルを適用、VIIRS はそのまま
        if year < 2014:
            corrected_data = apply_correction_model(correction_model, data)
        else:
            corrected_data = data

        # グラフ用に平坦化して格納（箱ひげ図を描くため）
        corrected_data_dict[str(year)] = corrected_data.flatten()

    # 箱ひげ図を表示
    visualize_data_as_boxplot(corrected_data_dict)

if __name__ == "__main__":
    main()
