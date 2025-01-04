import ee
import os
import requests

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

def create_yearly_image(snippet, geometry, year, method='mean'):
    """
    start_date ~ end_date の期間を対象に、指定されたデータセットの月次データをまとめて1枚のImageにする。
    """
    try:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        dataset = (ee.ImageCollection(snippet)
                   .filter(ee.Filter.date(start_date, end_date))
                   .filterBounds(geometry))
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

    # カリフォルニア州のジオメトリ
    california_geometry = ee.Geometry.Polygon(
        coords=[
            [-124.482003, 32.528832],
            [-114.131211, 32.528832],
            [-114.131211, 42.009518],
            [-124.482003, 42.009518],
            [-124.482003, 32.528832]
        ],
        proj='EPSG:4326',
        geodesic=False
    )

    # ミシシッピ州のジオメトリ
    mississippi_geometry = ee.Geometry.Polygon(
        coords=[
            [-91.655009, 30.173943],
            [-88.097888, 30.173943],
            [-88.097888, 34.996053],
            [-91.655009, 34.996053],
            [-91.655009, 30.173943]
        ],
        proj='EPSG:4326',
        geodesic=False
    )

    scale = 1000
    california_folder = './California'
    mississippi_folder = './Mississippi'

    years = range(2006, 2022)

    for year in years:
        print(f"Processing year: {year}")
        for state, geometry, folder in [
            ("California", california_geometry, california_folder),
            ("Mississippi", mississippi_geometry, mississippi_folder)
        ]:
            try:
                if year < 2014:
                    # DMSP データを使用
                    snippet = 'projects/sat-io/open-datasets/Harmonized_NTL/dmsp'
                    band = 'stable_lights'
                else:
                    # VIIRS データを使用
                    snippet = 'projects/sat-io/open-datasets/Harmonized_NTL/viirs'
                    band = 'avg_rad'

                yearly_image = create_yearly_image(snippet, geometry, year)

                if yearly_image is None:
                    continue

                file_tag = f"{year}_harmonized_{state}"
                tiff_file_path = os.path.join(folder, f"{file_tag}.tif")
                downloaded_path = download_tiff(yearly_image, geometry, folder, file_tag, scale)

                if downloaded_path:
                    print(f"GeoTIFF for {year}, {state} saved successfully.")
            except Exception as e:
                print(f"An error occurred while processing {year}, {state}: {e}")

if __name__ == "__main__":
    main()
