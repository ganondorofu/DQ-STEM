import chardet
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# ここで日本語フォントを設定します（Windowsなら"Meiryo"など）
plt.rcParams['font.family'] = 'Meiryo'

# データのエンコーディングを特定
with open('./night_light_gdp.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# データを読み込む
data = pd.read_csv('./night_light_gdp.csv', encoding=encoding)

# 夜間光を正規化
data['mean_light_ratio'] = (data['mean_light'] - data['mean_light'].min()) / (data['mean_light'].max() - data['mean_light'].min())

# 指定した都道府県のGDPを基準に比率化
reference_prefecture = '東京都'
reference_gdp = data.loc[data['region_name'] == reference_prefecture, 'GDP'].values[0]
data['gdp_ratio'] = data['GDP'] / reference_gdp

# 特徴量とターゲットを設定
X = data[['mean_light_ratio']]
y = data['gdp_ratio']

# 特徴量を標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 交差検証の結果を格納
results_kfold = []
results_loyo = []

# モデルを定義
model = RandomForestRegressor(random_state=42)

# --- K-Fold法 ---
print("K-Fold Cross-Validation")
kf = KFold(n_splits=8, shuffle=False)
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results_kfold.append(mse)
    print(f"Fold {fold + 1}: MSE = {mse:.4f}")

print(f"Average MSE (K-Fold): {np.mean(results_kfold):.4f}\n")

# --- Leave-One-Year-Out法 ---
print("Leave-One-Year-Out Cross-Validation")
unique_years = data['年'].unique()
for year in unique_years:
    train_data = data[data['年'] != year]
    test_data = data[data['年'] == year]
    X_train = scaler.transform(train_data[['mean_light_ratio']])
    y_train = train_data['gdp_ratio']
    X_test = scaler.transform(test_data[['mean_light_ratio']])
    y_test = test_data['gdp_ratio']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results_loyo.append(mse)
    print(f"Year {year}: MSE = {mse:.4f}")

print(f"Average MSE (Leave-One-Year-Out): {np.mean(results_loyo):.4f}")

# --- 未来のGDP予測 ---
print("\nPredicting GDP for 2022 and beyond...")
future_years = [2022, 2023, 2024]
future_data_list = []
for year in future_years:
    future_file = f"./2022-2024nightlight/{year}_average.csv"
    future_data = pd.read_csv(future_file, encoding=encoding)
    future_data['mean_light_ratio'] = (
        (future_data['mean_light'] - data['mean_light'].min())
        / (data['mean_light'].max() - data['mean_light'].min())
    )
    X_future = scaler.transform(future_data[['mean_light_ratio']])
    future_data['predicted_gdp_ratio'] = model.predict(X_future)
    future_data['predicted_gdp'] = future_data['predicted_gdp_ratio'] * reference_gdp
    future_data['Year'] = year
    future_data_list.append(future_data)

future_combined = pd.concat(future_data_list, ignore_index=True)

print("Future predictions:")
print(future_combined[['region_name', 'Year', 'predicted_gdp']])

# --- GDP予測値の多い順に棒グラフをプロット (2022年の東京を1とする) ---
# 2022年の東京のGDPを基準に設定
tokyo_2022_gdp = future_combined[
    (future_combined['region_name'] == '東京都') & (future_combined['Year'] == 2022)
]['predicted_gdp'].values[0]

# 比率を計算
future_combined['gdp_ratio_to_tokyo_2022'] = future_combined['predicted_gdp'] / tokyo_2022_gdp

# 各地域の3年間の平均で並べ替え
future_combined['mean_gdp_ratio'] = future_combined.groupby('region_name')['gdp_ratio_to_tokyo_2022'].transform('mean')

# グラフの描画
plt.figure(figsize=(12, 6))
bar_width = 0.25
x_positions = np.arange(future_combined['region_name'].nunique())
for i, year in enumerate(future_years):
    subset = future_combined[future_combined['Year'] == year].sort_values(by='mean_gdp_ratio', ascending=False)
    region_names = subset['region_name'].unique()
    plt.bar(x_positions + i * bar_width, subset['gdp_ratio_to_tokyo_2022'], width=bar_width, label=f'Year {year}')

plt.title('Predicted GDP Ratio (2022 Tokyo = 1) for 2022, 2023, and 2024')
plt.xlabel('Region')
plt.ylabel('Predicted GDP Ratio (2022 Tokyo = 1)')
plt.xticks(x_positions + bar_width, region_names, rotation=45, fontsize=8)
plt.legend()
plt.tight_layout()
plt.show()
