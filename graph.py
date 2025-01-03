import pandas as pd
import matplotlib.pyplot as plt
import glob

# ファイルパスを取得（各年の結果ファイル）
result_files = glob.glob('./model/model_results_*.csv')

# 全データを保存するリスト
data_frames = []

# 各ファイルを読み込み、年次を抽出してデータフレームに追加
for file in result_files:
    df = pd.read_csv(file)
    year = file.split('_')[-1].split('.')[0]  # ファイル名から年次を抽出
    df['Year'] = year
    data_frames.append(df)

# 全データを結合
all_data = pd.concat(data_frames, ignore_index=True)

# グラフの作成
models = all_data['Model'].unique()
years = sorted(all_data['Year'].unique())

# R^2 スコアの棒グラフ
plt.figure(figsize=(12, 6))
for model in models:
    model_data = all_data[all_data['Model'] == model]
    plt.plot(model_data['Year'], model_data['R^2 Score'], marker='o', label=model)

plt.title('R^2 Score by Model and Year')
plt.xlabel('Year')
plt.ylabel('R^2 Score')
plt.legend()
plt.grid(True)
plt.savefig('./R2_score_comparison.png')
plt.show()

# Mean Squared Error (MSE) の棒グラフ
plt.figure(figsize=(12, 6))
for model in models:
    model_data = all_data[all_data['Model'] == model]
    plt.plot(model_data['Year'], model_data['Mean Squared Error'], marker='o', label=model)

plt.title('Mean Squared Error by Model and Year')
plt.xlabel('Year')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig('./MSE_comparison.png')
plt.show()

print("Graphs saved as R2_score_comparison.png and MSE_comparison.png")
