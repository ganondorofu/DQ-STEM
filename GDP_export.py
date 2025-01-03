import pandas as pd
import re

# Excelファイルのパス
file_path = './huhyo2020.xlsx'

# シート名を指定
sheet_name = '合計'

# Excelファイルを読み込み
df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

# データ範囲を指定（B列: 都道府県、D列: GDP）
prefectures = df.iloc[8:55, 1].reset_index(drop=True)  # B列: 都道府県名
gdp = df.iloc[8:55, 3].reset_index(drop=True)         # D列: GDP

# データフレームを作成
gdp_data = pd.DataFrame({'都道府県': prefectures, 'GDP': gdp})

# データ型を調整（GDPを数値型に変換）
gdp_data['GDP'] = gdp_data['GDP'].replace(',', '', regex=True).astype(float)

# 入力ファイル名から年を抽出
year = re.search(r'\d{4}', file_path).group()

# CSVファイルに保存
output_csv_path = f'./gdp_by_prefecture{year}.csv'
gdp_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"都道府県別GDPデータを {output_csv_path} に保存しました。")
