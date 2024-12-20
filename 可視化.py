# 必要なライブラリをインポート
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 画像フォルダのパスを指定
image_dir = './downloaded_data'  # ローカルに保存されたGeoTIFFファイルのパス

# フォルダ内のすべてのGeoTIFF画像をリストアップ
def list_tiff_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

# GeoTIFF画像を読み込み地図上で可視化する関数
def visualize_tiff_with_bounds(file_path, vmin=50, vmax=150):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        bounds = src.bounds  # ファイルの地理的境界（経度・緯度）

        # 欠損値（NoData）の処理
        if src.nodata is not None:
            data[data == src.nodata] = 0

        # ピクセル値のスケールを強調（正規化とログスケールの組み合わせ）
        data = np.log1p(data)  # log(1 + x) でスケール調整
        data = (data - data.min()) / (data.max() - data.min()) * 255  # 正規化して0〜255の範囲にスケール

        plt.figure(figsize=(10, 10))
        plt.title(os.path.basename(file_path))
        plt.imshow(data, cmap='viridis', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], vmin=vmin, vmax=vmax)
        plt.colorbar(label='Enhanced Pixel Values')
        plt.xlabel('Longitude')  # 経度
        plt.ylabel('Latitude')  # 緯度
        plt.show()

# 保存されたすべての画像を地理座標で可視化
def visualize_all_tiff_files_with_bounds(image_dir, vmin=50, vmax=200):
    tiff_files = list_tiff_files(image_dir)
    print(f"Found {len(tiff_files)} TIFF files.")
    for tiff_file in tiff_files:
        visualize_tiff_with_bounds(tiff_file, vmin, vmax)

# 実行
visualize_all_tiff_files_with_bounds(image_dir, vmin=30, vmax=180)
