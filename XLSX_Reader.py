import pandas as pd
import openpyxl

# Excelファイルのパスを指定
file_path = './000972643.xlsx'

# openpyxlを使用してメインのExcelファイルを読み込む
workbook = openpyxl.load_workbook(file_path, data_only=True)
sheet = workbook.active

# マージされたセルを含むデータを取得
data = []
for row in sheet.iter_rows(min_row=13, values_only=True):
    data.append([cell if cell is not None else '' for cell in row])

# マージされたセルの値を上に引き上げる
for col in range(len(data[0])):
    last_value = ''
    for row in range(len(data)):
        if data[row][col] == '':
            data[row][col] = last_value
        else:
            last_value = data[row][col]

# DataFrameに変換
df = pd.DataFrame(data[1:], columns=data[0])  # 1行目を列名として設定

# 列インデックスで指定して列を抽出 (A=0, B=1, C=2, E=4)
df = df.iloc[:, [0, 1, 2, 4]]  # 0始まりの列インデックス

# データの先頭5行を表示
print(df.head())

# CSVファイルにエクスポート
output_file_path = './data/exported_data.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"データを {output_file_path} にエクスポートしました。")
