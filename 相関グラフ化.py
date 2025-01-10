import pandas as pd
import matplotlib.pyplot as plt
import os

# 年の範囲を指定
start_year = 2006
end_year = 2021

# 結果を格納するリスト
years = []
mean_light_correlations = []
sum_light_correlations = []

# データを収集
for year in range(start_year, end_year + 1):
    try:
        file_path = f'./Chowa_model/correlation_results_{year}.csv'
        if os.path.exists(file_path):
            result = pd.read_csv(file_path)
            years.append(year)
            mean_light_correlations.append(
                result.loc[
                    result['Metric'] == 'Mean Light Ratio vs GDP Ratio', 
                    'Correlation'
                ].values[0]
            )
            sum_light_correlations.append(
                result.loc[
                    result['Metric'] == 'Sum Light Ratio vs GDP Ratio', 
                    'Correlation'
                ].values[0]
            )
        else:
            print(f"File not found for year {year}: {file_path}")
    except Exception as e:
        print(f"An error occurred for year {year}: {e}")

data = pd.DataFrame({
    'Year': years,
    'Mean Light Correlation': mean_light_correlations,
    'Sum Light Correlation': sum_light_correlations
})

# 平均夜間光とGDPの相関の平均を計算
mean_correlation = data['Mean Light Correlation'].mean()
print("平均夜間光とGDPの相関の平均:", mean_correlation)

plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Mean Light Correlation'], label='Mean Light Ratio vs GDP Ratio', marker='o')
plt.plot(data['Year'], data['Sum Light Correlation'], label='Sum Light Ratio vs GDP Ratio', marker='o')
plt.title('Correlation between Nightlight Ratios and GDP Ratios (2006-2021)')
plt.xlabel('Year')
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.legend()
plt.grid()
plt.show()
