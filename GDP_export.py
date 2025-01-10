import pandas as pd
import re

start_year = 2014
end_year = 2021

for y in range(start_year, end_year + 1):
    file_path = f'./huhyo{y}.xlsx'
    sheet_name = '合計'
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    prefectures = df.iloc[8:55, 1].reset_index(drop=True)
    gdp = df.iloc[8:55, 3].reset_index(drop=True)
    gdp_data = pd.DataFrame({'都道府県': prefectures, 'GDP': gdp})
    gdp_data['GDP'] = gdp_data['GDP'].replace(',', '', regex=True).astype(float)
    output_csv_path = f'./gdp_csv/gdp_by_prefecture{y}.csv'
    gdp_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"{y}年のデータを {output_csv_path} に保存しました。")
