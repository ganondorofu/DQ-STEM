import ee
import os
import requests
import rasterio
import numpy as np
import pandas as pd
import pickle

from rasterstats import zonal_stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# yoneyone というモジュールから send_email 関数をインポート
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

def download_yearly_data(year, geometry, folder_path, region_name, scale=1000):
    """
    指定した year に対して:
      - 2013年までは DMSP (stable_lights)
      - 2014年以降は VIIRS (avg_rad)
    を年間平均で合成した TIFF をダウンロード。
    """
    try:
        if year < 2014:
            snippet = 'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS'
            band = 'stable_lights'
        else:
            snippet = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG'
            band = 'avg_rad'

        dataset = (
            ee.ImageCollection(snippet)
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .select(band)
        )
        image = dataset.mean()

        file_name = f"{region_name}_Yearly_{year}_{band}"
        download_tiff(image, geometry, folder_path, file_name, scale)
    except Exception as e:
        print(f"Error downloading data for {region_name} in {year}: {e}")

def load_tif(file_path):
    """
    GeoTIFFファイルを読み込む関数。(rasterio)
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile

def apply_dmsp_correction(dmsp_data, correction_model_path):
    """
    学習済みの補正モデル (dmsp_viirs_correction_model.pkl) を読み込んで
    DMSPデータをVIIRS相当に補正。
    """
    with open(correction_model_path, 'rb') as f:
        correction_model = pickle.load(f)
    dmsp_flat = dmsp_data.flatten().reshape(-1, 1)
    corrected_flat = correction_model.predict(dmsp_flat)
    corrected_data = corrected_flat.reshape(dmsp_data.shape)
    return corrected_data

def compute_zonal_stats(shapefile_path, tif_path, year):
    """
    shapefile + rasterstats で都道府県単位に mean, sum を取得。
    戻り値: pd.DataFrame(['都道府県','mean_light','sum_light','Year'])
    """
    # geojson_out=True だと shapefile を GeoJSON として扱うため、属性を properties に入れてくれる
    stats = zonal_stats(
        shapefile_path,
        tif_path,
        stats=["mean", "sum"],
        geojson_out=True,
        nodata=-999
    )

    records = []
    for feature in stats:
        props = feature["properties"]
        # N03_001 や N03_004など、shapefileの実態に合わせて修正
        # 今回は N03_001 に都道府県名が入っている想定
        pref_name = props.get("N03_001", "")
        mean_val = props.get("mean", None)
        sum_val = props.get("sum", None)
        records.append({
            "都道府県": pref_name,
            "mean_light": mean_val,
            "sum_light": sum_val
        })
    df = pd.DataFrame(records)
    df["Year"] = year
    return df

def merge_with_gdp_and_compute_ratio(df_light, df_gdp, baseline_region="東京都"):
    """
    df_light: columns=['都道府県','mean_light','sum_light','Year']
    df_gdp:   columns=['都道府県','gdp']
    """
    df_merged = pd.merge(df_light, df_gdp, on='都道府県', how='inner')

    baseline = df_merged[df_merged['都道府県'] == baseline_region]
    if baseline.empty:
        raise ValueError(f"基準都道府県 {baseline_region} が見つかりません。")

    base_mean = baseline['mean_light'].values[0]
    base_sum = baseline['sum_light'].values[0]
    base_gdp = baseline['gdp'].values[0]

    df_merged['mean_light_ratio'] = df_merged['mean_light'] / base_mean if base_mean else np.nan
    df_merged['sum_light_ratio'] = df_merged['sum_light'] / base_sum if base_sum else np.nan
    df_merged['gdp_ratio'] = df_merged['gdp'] / base_gdp if base_gdp else np.nan
    return df_merged

def train_and_evaluate(df_merged, year, results_folder):
    """
    df_merged: ['都道府県','mean_light_ratio','sum_light_ratio','gdp_ratio','Year']
    """
    X = df_merged[['mean_light_ratio','sum_light_ratio']].copy()
    y = df_merged['gdp_ratio'].copy()

    mask = ~X.isnull().any(axis=1) & ~y.isnull()
    mask &= np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Year': year,
            'Model': model_name,
            'MSE': mse,
            'R2': r2
        })
        print(f"[{year}] {model_name}: MSE={mse:.4f}, R2={r2:.4f}")

    df_results = pd.DataFrame(results)
    os.makedirs(results_folder, exist_ok=True)
    out_csv = os.path.join(results_folder, f"model_results_{year}.csv")
    df_results.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Saved CSV to {out_csv}\n")

def main():
    initialize_gee()

    # 1) ダウンロードの設定
    folder_path = './nighttime_data'
    os.makedirs(folder_path, exist_ok=True)

    years = range(2006, 2022)
    region_name = 'Japan'
    geometry = ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12])
    correction_model_path = './models/dmsp_viirs_correction_model.pkl'
    # N03-20220101_GML: 国交省の「国土数値情報　行政区域データ」などを想定
    shapefile_path = './N03-20220101_GML/N03-22_220101.shp'
    results_folder = './Chowa_model'

    # 2) ダウンロード (2006～2013: DMSP, 2014～: VIIRS)
    print("=== Downloading TIF files ===")
    for year in years:
        print(f"Year: {year}")
        download_yearly_data(year, geometry, folder_path, region_name)

    # 3) 各年のファイルを補正 + zonal_stats + GDPマージ + 学習・評価
    for year in years:
        print(f"\n=== Processing year {year} ===")

        if year < 2014:
            tif_path = os.path.join(folder_path, f"{region_name}_Yearly_{year}_stable_lights.tif")
            if not os.path.isfile(tif_path):
                print(f"Missing TIFF for {year}: {tif_path}, skipping.")
                continue
            # DMSP load & correct
            dmsp_data, dmsp_profile = load_tif(tif_path)
            corrected_dmsp = apply_dmsp_correction(dmsp_data, correction_model_path)
            combined_data = corrected_dmsp
            band_label = "stable_lights_corrected"
        else:
            tif_path = os.path.join(folder_path, f"{region_name}_Yearly_{year}_avg_rad.tif")
            if not os.path.isfile(tif_path):
                print(f"Missing TIFF for {year}: {tif_path}, skipping.")
                continue
            # VIIRS => no correction
            viirs_data, viirs_profile = load_tif(tif_path)
            combined_data = viirs_data
            band_label = "avg_rad"

        # 4) 補正orそのままのTIFをローカルに書き出し
        out_tif = os.path.join(folder_path, f"corrected_{year}_{band_label}.tif")
        if year < 2014:
            out_profile = dmsp_profile.copy()
        else:
            out_profile = viirs_profile.copy()
        out_profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(out_tif, 'w', **out_profile) as dst:
            dst.write(combined_data.astype(rasterio.float32), 1)
        print(f"Saved corrected TIF: {out_tif}")

        # 5) shapefile + rasterstats で都道府県ごとに mean, sum を計算
        print("Calculating zonal stats for each prefecture...")
        df_light = compute_zonal_stats(shapefile_path, out_tif, year)
        # df_light: ['都道府県','mean_light','sum_light','Year']

        # 6) GDPファイルを読み込み
        gdp_path = f"./gdp_by_prefecture{year}.csv"
        if not os.path.isfile(gdp_path):
            print(f"Missing GDP file for {year}: {gdp_path}, skipping.")
            continue
        df_gdp = pd.read_csv(gdp_path)  # columns=['都道府県','gdp']

        # 7) ratio計算 & モデル評価
        df_ratio = merge_with_gdp_and_compute_ratio(df_light, df_gdp, baseline_region="東京都")
        train_and_evaluate(df_ratio, year, results_folder)

if __name__ == "__main__":
    main()

    send_email(
        subject="処理完了",
        body="DMSP/VIIRSダウンロード＆補正＆都道府県別集計→GDP比較が完了しました。",
        to_email="example@example.com",
        sender_email="sender@gmail.com",
        sender_password="some-app-password"
    )
