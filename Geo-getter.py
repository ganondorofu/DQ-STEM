import os
import requests
import zipfile
from pathlib import Path

# 保存先ディレクトリ
save_dir = "./N03-20220101_GML"
zip_url = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2022/N03-20220101_GML.zip"
zip_file_path = "./N03-20220101_GML.zip"

# フォルダが存在しない場合のみダウンロード
if not os.path.exists(save_dir):
    print("指定されたディレクトリが存在しません。ZIPファイルをダウンロードしています...")
    
    # ZIPファイルをダウンロード
    response = requests.get(zip_url, stream=True)
    if response.status_code == 200:
        with open(zip_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"ZIPファイルをダウンロードしました: {zip_file_path}")
    else:
        print(f"ダウンロードに失敗しました: ステータスコード {response.status_code}")
        exit()

    # ZIPファイルを解凍
    print("ZIPファイルを解凍しています...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"解凍完了: {save_dir}")
    
    # ZIPファイルを削除
    os.remove(zip_file_path)
    print("ZIPファイルを削除しました。")
else:
    print(f"指定されたディレクトリは既に存在します: {save_dir}")

# 解凍後のファイル確認
files = list(Path(save_dir).rglob("*"))
print("解凍されたファイル一覧:")
for file in files:
    print(file)
