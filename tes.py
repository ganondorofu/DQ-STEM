import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# TensorFlowとKerasを使用
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'

# データの読み込み
df = pd.read_csv("DQ_人口データセット.csv", encoding="cp932", index_col=[0])

# 年次データを整数型に変換
df['year'] = df['year'].astype(int)

# 65歳以上人口の算出
df['65歳以上人口'] = df[['65歳～69歳', '70歳～74歳', '75歳～79歳', '80歳～84歳', '85歳～89歳', '90歳～94歳', '95歳～99歳', '100歳以上']].sum(axis=1)

# 高齢化率の計算
df['高齢化率'] = df['65歳以上人口'] / df['総数'] * 100

# 都道府県ごとのデータを取得
prefectures = df['都道府県名'].unique()

# 各都道府県の2030年までの高齢化率を予測し、結果を保存するリスト
results = []

for prefecture in prefectures:
    # 各都道府県のデータを取得
    df_pref = df[df['都道府県名'] == prefecture]
    
    # 年と高齢化率のデータを準備
    data = df_pref.groupby('year')['高齢化率'].mean().reset_index()
    
    # データの欠損値を確認
    if data['高齢化率'].isnull().any() or len(data) < 5:
        continue  # 欠損値がある場合やデータが少ない場合はスキップ
    
    # データの正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['高齢化率'].values.reshape(-1, 1))
    
    # 時系列データの作成
    sequence_length = 3  # 過去3年分のデータを使用
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # データの形状をLSTMモデルに適合させる
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # LSTMモデルの構築
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    # モデルのコンパイル
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # モデルの学習
    model.fit(X, y, epochs=50, batch_size=1, verbose=0)
    
    # 未来の高齢化率を予測（2030年まで）
    years_to_predict = 2030 - data['year'].max()
    predictions = []
    
    # 直近のデータを使用して予測を開始
    last_sequence = scaled_data[-sequence_length:]
    
    for _ in range(years_to_predict):
        X_pred = last_sequence.reshape(1, sequence_length, 1)
        pred = model.predict(X_pred)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred[0][0])
    
    # 予測結果を元のスケールに戻す
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # 予測年のリストを作成
    future_years = np.arange(data['year'].max() + 1, 2031)
    
    # 結果を保存
    for year, pred in zip(future_years, predictions.flatten()):
        results.append({
            '都道府県名': prefecture,
            'year': year,
            '予測高齢化率': pred
        })

# 結果をデータフレームに変換
df_results = pd.DataFrame(results)

# 2030年の高齢化率を抽出
df_2030 = df_results[df_results['year'] == 2030]

# 2030年に最も高齢化が進む都道府県を特定
highest_aging_prefecture = df_2030.loc[df_2030['予測高齢化率'].idxmax()]
print(f"2030年に最も高齢化が進むと予測される都道府県は: {highest_aging_prefecture['都道府県名']} で、高齢化率は {highest_aging_prefecture['予測高齢化率']:.2f}% です。")

# 結果の視覚化
# 都道府県ごとの2030年高齢化率予測を降順にソート
df_2030_sorted = df_2030.sort_values('予測高齢化率', ascending=False)

# 棒グラフの描画
plt.figure(figsize=(12, 8))
plt.barh(df_2030_sorted['都道府県名'], df_2030_sorted['予測高齢化率'], color='skyblue')
plt.xlabel('2030年高齢化率予測 (%)')
plt.title('都道府県別2030年高齢化率予測（LSTMモデル使用）')
plt.gca().invert_yaxis()  # 都道府県名を上から順に表示
plt.tight_layout()
plt.show()
