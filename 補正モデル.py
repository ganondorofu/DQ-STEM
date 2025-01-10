import ee
import os
import requests
import rasterio
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
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

def download_yearly_data(snippet, band, geometry, year, folder_path, region_name, scale=1000):
    """
    年次データを生成して保存する。(同一年内の画像を mean() で合成)
    """
    try:
        dataset = (ee.ImageCollection(snippet)
                   .filter(ee.Filter.calendarRange(year, year, 'year'))
                   .select(band))

        # 単純平均で合成
        image = dataset.mean()
        # 必要に応じてスケール係数を掛ける（例: image = image.multiply(0.1)）

        file_name = f"{region_name}_Yearly_{year}_{band}"
        download_tiff(image, geometry, folder_path, file_name, scale)
    except Exception as e:
        print(f"Error downloading yearly data for {region_name} in {year}: {e}")

def load_tif(file_path):
    """
    GeoTIFFファイルを読み込む関数。
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  # バンド1を読む
        profile = src.profile
    return data, profile

def clean_and_sample_data(dmsp_data, viirs_data, sample_ratio=0.1, clip_max=None):
    """
    DMSP データと VIIRS データをクレンジング・サンプリングする関数。
    - 欠損値(NoData)や負値を除外し、外れ値をクリップし、一定割合でランダムサンプリング。
    引数:
      sample_ratio: 画素の何割を学習に使うか (0 < sample_ratio <= 1)
      clip_max: 上限値。たとえば clip_max=300 なら 300 でクリップ。
    戻り値:
      dmsp_flat, viirs_flat (同じ長さの配列)
    """
    # 欠損値や Inf を除外
    dmsp_flat = dmsp_data.flatten()
    viirs_flat = viirs_data.flatten()

    mask = ~np.isnan(dmsp_flat) & ~np.isnan(viirs_flat)
    mask &= np.isfinite(dmsp_flat) & np.isfinite(viirs_flat)

    # 0未満を除外したい場合
    mask &= (dmsp_flat >= 0) & (viirs_flat >= 0)

    dmsp_valid = dmsp_flat[mask]
    viirs_valid = viirs_flat[mask]

    # clip_max が指定されていれば上限クリップ
    if clip_max is not None:
        dmsp_valid = np.clip(dmsp_valid, 0, clip_max)
        viirs_valid = np.clip(viirs_valid, 0, clip_max)

    # ランダムサンプリング
    n = len(dmsp_valid)
    if sample_ratio < 1.0 and sample_ratio > 0.0:
        sample_size = int(n * sample_ratio)
        idx = np.random.choice(n, sample_size, replace=False)
        dmsp_valid = dmsp_valid[idx]
        viirs_valid = viirs_valid[idx]

    return dmsp_valid, viirs_valid

def build_correction_model(dmsp_data, viirs_data, model_type='random_forest'):
    """
    DMSP と VIIRS データを使って補正モデルを構築し、単一オブジェクトとして返す。
    model_type は 'linear' / 'polynomial' / 'random_forest' のいずれか。
    """
    # -- 1) データクレンジング＆サンプリング --
    # 必要に応じてパラメータを調整 (sample_ratio など)
    dmsp_valid, viirs_valid = clean_and_sample_data(dmsp_data, viirs_data,
                                                    sample_ratio=0.1,
                                                    clip_max=500)

    # -- 2) train / test 分割で汎化性能を確認 --
    X_all = dmsp_valid.reshape(-1, 1)
    y_all = viirs_valid

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                        test_size=0.2,
                                                        random_state=42)

    # -- 3) モデルの生成 --
    if model_type == 'linear':
        model = LinearRegression()

    elif model_type == 'polynomial':
        # 次数を 2→3 に上げればさらに複雑化可能
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression())
        ])

    elif model_type == 'random_forest':
        # 高精度寄りのパラメータ
        model = RandomForestRegressor(
            n_estimators=100,  # 木の本数を増やす
            max_depth=None,    # 深さ制限をしない
            random_state=42,
            n_jobs=-1          # 複数CPUコアを使えるなら
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # -- 4) 学習 --
    model.fit(X_train, y_train)

    # -- 5) 学習データ・テストデータ双方の R² を表示
    r2_train = model.score(X_train, y_train)
    r2_test  = model.score(X_test,  y_test)
    print(f"[{model_type} モデル]")
    print(f"  R² (train) = {r2_train:.3f}, R² (test) = {r2_test:.3f}")

    return model

def apply_correction_model(model, dmsp_data):
    """
    補正モデルを使用してDMSP データを補正する。
    model は単一のオブジェクト (Pipeline or Regressor)。
    """
    X = dmsp_data.flatten().reshape(-1, 1)
    # 欠損値や負値はそのままになっているので、簡易的に nan→0 にする等の対処もあり
    pred = model.predict(X)
    return pred.reshape(dmsp_data.shape)

def save_corrected_tif(corrected_data, profile, output_path):
    """
    補正後のデータをGeoTIFFとして保存。
    """
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(corrected_data.astype(rasterio.float32), 1)

def process_tif_files(dmsp_path, viirs_path, output_path, model_path, model_type='random_forest'):
    """
    DMSP と VIIRS のファイルから補正モデルを作り、
    DMSP を補正して GeoTIFF に保存し、モデルも pickle 化する。
    """
    dmsp_data, dmsp_profile = load_tif(dmsp_path)
    viirs_data, _ = load_tif(viirs_path)

    # モデルを構築 (単一オブジェクト)
    model = build_correction_model(dmsp_data, viirs_data, model_type=model_type)

    # モデルを保存 (pickle)
    os.makedirs('./models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"補正モデルを保存しました: {model_path}")

    # DMSP データを補正
    corrected_dmsp = apply_correction_model(model, dmsp_data)

    # 補正後データを保存
    save_corrected_tif(corrected_dmsp, dmsp_profile, output_path)

def main():
    initialize_gee()

    # ダウンロード設定
    datasets = [
        {'snippet': 'NOAA/DMSP-OLS/NIGHTTIME_LIGHTS', 'band': 'stable_lights'}, 
        {'snippet': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG', 'band': 'avg_rad'}
    ]

    regions = {
        'Japan': ee.Geometry.Rectangle([128.60, 29.97, 148.43, 46.12])
    }

    folder_path = './nighttime_data'
    years = [2012, 2013]  # 対象年

    # データをダウンロード
    for region_name, geometry in regions.items():
        print(f"Processing region: {region_name}")
        for dataset in datasets:
            snippet = dataset['snippet']
            band = dataset['band']
            for year in years:
                print(f"Downloading {year} data for {region_name} using dataset {snippet}")
                download_yearly_data(snippet, band, geometry, year, folder_path, region_name)

    # 例: 2012 年の DMSP / VIIRS でモデル構築
    dmsp_path = './nighttime_data/Japan_Yearly_2012_stable_lights.tif'
    viirs_path = './nighttime_data/Japan_Yearly_2012_avg_rad.tif'

    # 出力
    output_path = './nighttime_data/Japan_Corrected_2012.tif'
    model_path = './models/dmsp_viirs_correction_model.pkl'

    # モデル手法: 'linear', 'polynomial', 'random_forest'
    chosen_model_type = 'polynomial'  # 好みに応じて

    # TIFF を処理 & 補正モデルを保存 & 補正GeoTIFFを作成
    process_tif_files(
        dmsp_path=dmsp_path,
        viirs_path=viirs_path,
        output_path=output_path,
        model_path=model_path,
        model_type=chosen_model_type
    )

if __name__ == "__main__":
    main()

    # ダウンロード完了メール送信
    send_email(
        subject="ダウンロード完了",
        body="GeoTIFFのダウンロード & モデル作成（精度向上版）が完了しました。",
        to_email="ganondorofu3143@outlook.com",
        sender_email="ganondorofu123@gmail.com",
        sender_password="cixg vnmb gvfn ltwi"
    )
