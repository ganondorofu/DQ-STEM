import pandas as pd
import os

# フォルダパス
gdp_folder_path = './gdp_csv/'  # 年ごとのGDPデータが入っているフォルダ
night_light_folder_path = './night_light_csv/'  # 年ごとの夜間光データが入っているフォルダ
output_path = './night_light_gdp.csv'  # 結合結果の保存先

# 年の範囲
years = range(2014, 2022)

# 結合データを保存するリスト
merged_data_list = []

# 年ごとにデータを結合
for year in years:
    print(f"Processing year: {year}")
    
    # ファイルパスの生成
    gdp_file_path = os.path.join(gdp_folder_path, f'gdp_by_prefecture{year}.csv')
    night_light_file_path = os.path.join(night_light_folder_path, f'{year}_average.csv')
    
    # GDPデータの読み込み
    try:
        gdp_data = pd.read_csv(gdp_file_path)
        gdp_data.rename(columns=lambda x: x.strip(), inplace=True)  # 列名の余分な空白を削除
        gdp_data.rename(columns={'都道府県': 'region_name'}, inplace=True)  # 列名を統一
        gdp_data['年'] = year  # 年列がない場合に追加
        gdp_data['region_name'] = gdp_data['region_name'].str.strip()  # 都道府県名の整形
    except FileNotFoundError:
        print(f"GDP file not found: {gdp_file_path}")
        continue
    
    # 夜間光データの読み込み
    try:
        night_light_data = pd.read_csv(night_light_file_path)
        night_light_data.rename(columns=lambda x: x.strip(), inplace=True)  # 列名の余分な空白を削除
        night_light_data['年'] = year  # 年列がない場合に追加
        night_light_data['region_name'] = night_light_data['region_name'].str.strip()  # 都道府県名の整形
    except FileNotFoundError:
        print(f"Night light file not found: {night_light_file_path}")
        continue
    
    # 都道府県名と年をキーに結合
    try:
        merged_data = pd.merge(night_light_data, gdp_data, on=['region_name', '年'], how='inner')
        merged_data_list.append(merged_data)  # 結合結果をリストに追加
    except KeyError as e:
        print(f"KeyError: {e}. Check column names in files.")
        continue

# 全年分のデータを結合
if merged_data_list:
    final_merged_data = pd.concat(merged_data_list, ignore_index=True)

    # 結果の確認
    print("\n結合後のデータサンプル:")
    print(final_merged_data.head())

    # CSVとして保存
    final_merged_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n結合結果を保存しました: {output_path}")
else:
    print("No data was merged. Check the input files.")
