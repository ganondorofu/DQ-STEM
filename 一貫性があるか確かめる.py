import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os  # ファイル名操作に必要

# CSVデータを結合
csv_folder = './Chowa_CSV'  # CSVファイルの保存先フォルダ
csv_files = glob.glob(f"{csv_folder}/*.csv")

# CSVファイルを結合
data_frames = []
for file in csv_files:
    try:
        # ファイル名から年を抽出 (xxxx_harmonized.csv の形式を想定)
        file_name = os.path.basename(file)  # パスからファイル名を取得
        year = int(file_name.split('_')[0])  # 年を取得
    except ValueError:
        print(f"Skipping invalid file: {file}")
        continue

    df = pd.read_csv(file)
    df['year'] = year  # 年を追加
    data_frames.append(df)

# すべてのデータを結合
if data_frames:
    combined_data = pd.concat(data_frames, ignore_index=True)

    # データの確認
    print(combined_data.head())

    # ヒストグラムを作成
    plt.figure(figsize=(12, 6))
    for year, group in combined_data.groupby('year'):
        plt.hist(group['mean_light'], bins=30, alpha=0.5, label=f"Year {year}")
    plt.title("Distribution of Mean Light by Year")
    plt.xlabel("Mean Light")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # 箱ひげ図を作成
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year', y='mean_light', data=combined_data)
    plt.title("Mean Light Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean Light")
    plt.show()
else:
    print("No valid CSV files found.")
