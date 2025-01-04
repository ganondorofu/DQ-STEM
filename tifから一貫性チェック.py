import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os  # ファイル名操作に必要

# GeoTIFFデータを結合
tif_folder = './USA'  # GeoTIFFファイルの保存先フォルダ
tif_files = glob.glob(f"{tif_folder}/*.tif")

# データを保存するリスト
data_frames = []

for file in tif_files:
    try:
        # ファイル名から年を抽出 (xxxx_harmonized_regionX.tif の形式を想定)
        file_name = os.path.basename(file)
        year = int(file_name.split('_')[0])  # 年を取得
        region = file_name.split('_')[2]  # 地域情報を取得（regionX）
    except ValueError:
        print(f"Skipping invalid file: {file}")
        continue

    # GeoTIFFを開く
    with rasterio.open(file) as src:
        # データを読み込み
        band_data = src.read(1)  # バンド1を読み込み
        band_data = band_data[band_data > 0]  # 有効なピクセル値のみを抽出 (0以下を無視)

    # データを保存
    for value in band_data:
        data_frames.append({'year': year, 'region': region, 'mean_light': value})

# データフレームに変換
if data_frames:
    combined_data = pd.DataFrame(data_frames)

    # データの確認
    print(combined_data.head())

    # ヒストグラムを作成
    plt.figure(figsize=(12, 6))
    for year, group in combined_data.groupby('year'):
        plt.hist(group['mean_light'], bins=30, alpha=0.5, label=f"Year {year}")
    plt.title("Distribution of Mean Light by Year (Direct from GeoTIFF)")
    plt.xlabel("Mean Light")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # 箱ひげ図を作成
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year', y='mean_light', data=combined_data)
    plt.title("Mean Light Distribution by Year (Direct from GeoTIFF)")
    plt.xlabel("Year")
    plt.ylabel("Mean Light")
    plt.show()
else:
    print("No valid GeoTIFF files found.")
