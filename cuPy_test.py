import cupy as cp

# GPUが利用可能かをテストする関数
def test_gpu_with_cupy():
    try:
        # 簡単な配列を生成して計算する
        a = cp.array([1, 2, 3, 4, 5])  # GPU上に配列を作成
        b = cp.array([5, 4, 3, 2, 1])  # GPU上に配列を作成
        result = a + b  # GPU上で配列の加算を実行

        # 結果を確認（GPU上のデータを取得して表示）
        print("GPUでの計算が成功しました！")
        print("計算結果:", cp.asnumpy(result))  # GPU上のデータをCPUに移動して表示
        print("GPUの情報:", cp.cuda.Device(0).name)  # 利用しているGPUデバイスの名前を取得
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("GPUが利用できません。エラー内容:")
        print(e)
    except Exception as ex:
        print("予期しないエラーが発生しました。エラー内容:")
        print(ex)

# テストを実行
if __name__ == "__main__":
    test_gpu_with_cupy()
