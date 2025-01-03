import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import os

year = '2010'

# 夜間光データの読み込み
nightlight_data = pd.read_csv('./yearly_csv/' + str(year) + '_average.csv')

# 経済指標データの読み込み
economic_data = pd.read_csv('./gdp_by_prefecture' + str(year) + '.csv')

# 入力ファイルの年数を取得
input_year = os.path.basename('./yearly_csv/' + str(year) + '_average.csv').split('_')[0]

# 列名の統一
nightlight_data.rename(columns={'region_name': '都道府県'}, inplace=True)
economic_data.rename(columns={'都道府県': '都道府県', 'GDP': 'gdp'}, inplace=True)

# データの結合
data = pd.merge(nightlight_data, economic_data, on='都道府県')

# 基準都道府県を東京都とする
baseline_region = '東京都'
baseline_values = data[data['都道府県'] == baseline_region]
if baseline_values.empty:
    raise ValueError(f"基準となる都道府県 {baseline_region} がデータに見つかりません。")

data['mean_light_ratio'] = data['mean_light'] / baseline_values['mean_light'].values[0]
data['sum_light_ratio'] = data['sum_light'] / baseline_values['sum_light'].values[0]
data['gdp_ratio'] = data['gdp'] / baseline_values['gdp'].values[0]

# 特徴量とターゲットの設定
X = data[['mean_light_ratio', 'sum_light_ratio']]
y = data['gdp_ratio']

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの定義
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# 結果を保存するリスト
results = []

# モデルごとの学習と評価
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 評価指標を計算
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'Mean Squared Error': mse,
        'R^2 Score': r2
    })

    # 結果の出力
    print(f"\n--- {model_name} ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

# 結果をデータフレームとして保存
results_df = pd.DataFrame(results)
print("\n--- Overall Results ---")
print(results_df)

# 結果をCSVに保存
output_csv_path = f'./model/model_results_{input_year}.csv'
results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"Results saved to {output_csv_path}")
