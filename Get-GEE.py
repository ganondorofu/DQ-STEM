import ee
import os
import requests

# GEEの認証・初期化
ee.Authenticate()
ee.Initialize(project='stem-443415')

# GEEのデータロード
def load_data(snippet, from_date, to_date, geometry, band):
    dataset = ee.ImageCollection(snippet).filter(
        ee.Filter.date(from_date, to_date)).filterBounds(geometry).select(band)
    data_list = dataset.toList(dataset.size())
    return dataset.size().getInfo(), data_list

# ローカルに保存するためのエクスポート
def save_to_local(image, geometry, folder_path, file_name, scale):
    try:
        # エクスポート設定
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
        else:
            print(f"Failed to download {file_name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"An error occurred while downloading {file_name}: {e}")

# パラメーターの指定
snippet = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
band = 'avg_rad'
from_date = '2014-01-01'
to_date = '2024-06-04'
geometry = ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12])  # 日本全域
scale = 1000
folder_path = './downloaded_data'  # 保存先フォルダ

# データ処理の実行
num_data, data_list = load_data(snippet=snippet, from_date=from_date, to_date=to_date, geometry=geometry, band=band)
print('#Datasets:', num_data)

# データのダウンロード
for i in range(num_data):
    image = ee.Image(data_list.get(i))
    file_name = image.getInfo()['id'].replace('/', '_')  # ファイル名に適切な文字列を使用
    save_to_local(image, geometry, folder_path, file_name, scale)
