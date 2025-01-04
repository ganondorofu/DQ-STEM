import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

# フォルダ設定
folders = {
    "California": "./California",
    "Mississippi": "./Mississippi"
}

# データを保存するリスト
data_frames = []

for state, folder in folders.items():
    tif_files = glob.glob(f"{folder}/*.tif")

    for file in tif_files:
        try:
            # ファイル名から年を抽出
            file_name = os.path.basename(file)
            year = int(file_name.split('_')[0])  # 年を取得
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
            data_frames.append({'year': year, 'state': state, 'mean_light': value})

# データフレームに変換
if data_frames:
    combined_data = pd.DataFrame(data_frames)

    # データの確認
    print(combined_data.head())

    # ヒストグラムを作成
    plt.figure(figsize=(12, 6))
    for (state, year), group in combined_data.groupby(['state', 'year']):
        plt.hist(group['mean_light'], bins=30, alpha=0.5, label=f"{state} {year}")
    plt.title("Distribution of Mean Light by State and Year (Direct from GeoTIFF)")
    plt.xlabel("Mean Light")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # 箱ひげ図を作成
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year', y='mean_light', hue='state', data=combined_data)
    plt.title("Mean Light Distribution by State and Year (Direct from GeoTIFF)")
    plt.xlabel("Year")
    plt.ylabel("Mean Light")
    plt.legend(title="State")
    plt.show()
else:
    print("No valid GeoTIFF files found.")
