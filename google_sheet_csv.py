import pandas as pd
import glob

# ./modelフォルダ内の全てのCSVファイルを取得
result_files = glob.glob('./Chowa_model/model_results_*.csv')

# 全データを保存するリスト
data_frames = []

# 各ファイルを読み込み、年次を抽出してデータフレームに追加
for file in result_files:
    df = pd.read_csv(file)
    year = file.split('_')[-1].split('.')[0]  # ファイル名から年次を抽出
    df['年'] = year
    data_frames.append(df)

# 全データを結合
all_data = pd.concat(data_frames, ignore_index=True)

# 必要なフォーマットに変換
pivot_r2 = all_data.pivot(index='年', columns='Model', values='R^2 Score').reset_index()
pivot_mse = all_data.pivot(index='年', columns='Model', values='Mean Squared Error').reset_index()

# 日本語表記への変更
model_mapping = {
    'Random Forest': 'ランダムフォレスト',
    'Linear Regression': '線形回帰',
    'Ridge Regression': 'リッジ回帰',
    'Lasso Regression': 'ラッソ回帰',
    'Gradient Boosting': '勾配ブースティング'
}

pivot_r2.rename(columns=model_mapping, inplace=True)
pivot_mse.rename(columns=model_mapping, inplace=True)

# 結合データフレームの作成
final_df = pd.DataFrame()
final_df['年'] = pivot_r2['年']
for col in pivot_r2.columns[1:]:
    final_df[f"決定係数 ({col})"] = pivot_r2[col]
for col in pivot_mse.columns[1:]:
    final_df[f"平均二乗誤差 ({col})"] = pivot_mse[col]

# CSVに保存
output_file = './formatted_model_results_jp.csv'
final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Formatted data saved to {output_file}")
