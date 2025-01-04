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

    # アメリカ全体を分割したジオメトリ (4つのサブリージョン)
    regions = [
        ee.Geometry.Polygon(coords=[[-125.0, 24.396308], [-96.0, 24.396308], [-96.0, 37.0], [-125.0, 37.0], [-125.0, 24.396308]], proj='EPSG:4326', geodesic=False),
        ee.Geometry.Polygon(coords=[[-96.0, 24.396308], [-66.93457, 24.396308], [-66.93457, 37.0], [-96.0, 37.0], [-96.0, 24.396308]], proj='EPSG:4326', geodesic=False),
        ee.Geometry.Polygon(coords=[[-125.0, 37.0], [-96.0, 37.0], [-96.0, 49.384358], [-125.0, 49.384358], [-125.0, 37.0]], proj='EPSG:4326', geodesic=False),
        ee.Geometry.Polygon(coords=[[-96.0, 37.0], [-66.93457, 37.0], [-66.93457, 49.384358], [-96.0, 49.384358], [-96.0, 37.0]], proj='EPSG:4326', geodesic=False)
    ]

    scale = 1000
    tiff_folder = './USA'

    years = range(2006, 2022)

    for year in years:
        print(f"Processing year: {year}")
        for idx, geometry in enumerate(regions):
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

                file_tag = f"{year}_harmonized_region{idx+1}"
                tiff_file_path = os.path.join(tiff_folder, f"{file_tag}.tif")
                downloaded_path = download_tiff(yearly_image, geometry, tiff_folder, file_tag, scale)

                if downloaded_path:
                    print(f"GeoTIFF for {year}, region {idx+1} saved successfully.")
            except Exception as e:
                print(f"An error occurred while processing {year}, region {idx+1}: {e}")

if __name__ == "__main__":
    main()
