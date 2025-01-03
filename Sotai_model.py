import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import os

# 年の範囲を指定
start_year = 2006
end_year = 2021

# 範囲内の年を処理
for year in range(start_year, end_year + 1):
    print(f"Processing year: {year}")

    try:
        # 夜間光データの読み込み
        nightlight_file = f'./GEE_csv_data/{year}_relative.csv'
        if not os.path.exists(nightlight_file):
            print(f"File not found: {nightlight_file}. Skipping year {year}.")
            continue
        nightlight_data = pd.read_csv(nightlight_file)

        # 経済指標データの読み込み
        gdp_file = f'./gdp_by_prefecture{year}.csv'
        if not os.path.exists(gdp_file):
            print(f"File not found: {gdp_file}. Skipping year {year}.")
            continue
        economic_data = pd.read_csv(gdp_file)

        # 列名の統一
        nightlight_data.rename(columns={'region_name': '都道府県'}, inplace=True)
        economic_data.rename(columns={'都道府県': '都道府県', 'GDP': 'gdp'}, inplace=True)

        # データの結合
        data = pd.merge(nightlight_data, economic_data, on='都道府県')

        # 特徴量とターゲットの設定
        X = data[['relative_mean_light', 'relative_sum_light']]
        y = data['gdp']

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
            print(f"Training {model_name} for year {year}...")
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
            print(f"\n--- {model_name} for year {year} ---")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R^2 Score: {r2:.4f}")

        # 結果をデータフレームとして保存
        results_df = pd.DataFrame(results)
        print(f"\n--- Overall Results for year {year} ---")
        print(results_df)

        # 結果をCSVに保存
        output_csv_path = f'./Soutai_model/model_results_{year}.csv'
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_csv_path}")

    except Exception as e:
        print(f"An error occurred for year {year}: {e}")
