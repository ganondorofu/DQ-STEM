import rasterio
from rasterio.merge import merge
import os
import glob

def merge_tiffs(tiff_folder, output_folder, year):
    """
    指定した年の分割GeoTIFFファイルを結合し、1つのファイルとして保存する。
    """
    # 指定年のGeoTIFFファイルを取得
    tiff_files = sorted(glob.glob(f"{tiff_folder}/{year}_harmonized_region*.tif"))
    if not tiff_files:
        print(f"No TIFF files found for year {year}.")
        return

    # 各TIFFファイルを開く
    src_files_to_mosaic = []
    for file in tiff_files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)

    # TIFFファイルを結合
    mosaic, out_transform = merge(src_files_to_mosaic)

    # 出力プロファイルを作成
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": src.crs
    })

    # 結合したTIFFを保存
    output_file = os.path.join(output_folder, f"{year}_harmonized_combined.tif")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    print(f"Combined TIFF saved at {output_file}")

def main():
    tiff_folder = './USA'
    output_folder = './USA_Combined'
    years = range(2006, 2022)

    for year in years:
        print(f"Processing year {year} for merging...")
        merge_tiffs(tiff_folder, output_folder, year)

if __name__ == "__main__":
    main()
