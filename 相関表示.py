import pandas as pd
import os

# 年の範囲を指定
start_year = 2006
end_year = 2021

# 範囲内の年を処理
for year in range(start_year, end_year + 1):
    print(f"Processing year: {year}")

    try:
        # 夜間光データの読み込み
        nightlight_data = pd.read_csv(f'./Chowa_CSV/{year}_harmonized.csv')

        # 経済指標データの読み込み
        economic_data = pd.read_csv(f'./gdp_by_prefecture{year}.csv')

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

        # 相関係数の計算
        correlation_mean_light = data['mean_light_ratio'].corr(data['gdp_ratio'])
        correlation_sum_light = data['sum_light_ratio'].corr(data['gdp_ratio'])

        # 結果の出力
        print(f"\n--- Correlation for year {year} ---")
        print(f"Correlation (Mean Light Ratio vs GDP Ratio): {correlation_mean_light:.4f}")
        print(f"Correlation (Sum Light Ratio vs GDP Ratio): {correlation_sum_light:.4f}")

        # 結果を保存
        output_path = f'./Chowa_model/correlation_results_{year}.csv'
        results_df = pd.DataFrame({
            'Metric': ['Mean Light Ratio vs GDP Ratio', 'Sum Light Ratio vs GDP Ratio'],
            'Correlation': [correlation_mean_light, correlation_sum_light]
        })
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Correlation results saved to {output_path}")

    except Exception as e:
        print(f"An error occurred for year {year}: {e}")
