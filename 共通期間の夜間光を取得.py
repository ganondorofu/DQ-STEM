import ee
import os
import requests
from yoneyone import send_email

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
    """
    try:
        dataset = ee.ImageCollection(snippet) \
                    .filter(ee.Filter.calendarRange(year, year, 'year')) \
                    .select(band)

        # 年次データを取得
        image = dataset.mosaic()  # 年次データを1つの画像に統合
        file_name = f"{region_name}_Yearly_{year}_{band.split('_')[0]}"
        download_tiff(image, geometry, folder_path, file_name, scale)
    except Exception as e:
        print(f"Error downloading yearly data for {region_name} in {year}: {e}")

def main():
    initialize_gee()

    # 設定
    datasets = [
        {'snippet': 'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS', 'band': 'stable_lights'},  # DMSP-OLS の主なバンド
        {'snippet': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG', 'band': 'avg_rad'}  # VIIRS の主なバンド
    ]

    # 複数地域の設定
    regions = {
        'Japan': ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12]),  # 日本全域
        'USA_East': ee.Geometry.Rectangle([-80.0, 30.0, -75.0, 35.0]),  # アメリカ東部の一部
        'USA_West': ee.Geometry.Rectangle([-120.0, 35.0, -115.0, 40.0]),  # アメリカ西部の一部
        'India': ee.Geometry.Rectangle([68.7, 8.0, 80.0, 25.0]),  # インド北部
        'China': ee.Geometry.Rectangle([105.0, 25.0, 115.0, 40.0])  # 中国中部
    }

    folder_path = './nighttime_data'
    years = [2012, 2013]  # 対象年

    for region_name, geometry in regions.items():
        print(f"Processing region: {region_name}")
        for dataset in datasets:
            snippet = dataset['snippet']
            band = dataset['band']
            for year in years:
                print(f"Downloading {year} data for {region_name} using dataset {snippet}")
                download_yearly_data(snippet, band, geometry, year, folder_path, region_name)

if __name__ == "__main__":
    main()
    send_email(
        subject="ダウンロード完了",
        body="GeoTIFFのダウンロードが完了しました（2012年および2013年の年次データを取得）。",
        to_email="ganondorofu3143@outlook.com",
        sender_email="ganondorofu123@gmail.com",
        sender_password="cixg vnmb gvfn ltwi"
    )
