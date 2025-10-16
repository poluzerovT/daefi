import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv("Annual 2005-2011.csv")
print("Размер данных:", df.shape)

sts = pd.DataFrame({
    'Минимум': df.min(),
    'Максимум': df.max(),
    'Размах': df.max() - df.min(),
    'Среднее': df.mean(),
    'Дисперсия': df.var(),
    'Стандартное отклонение': df.std(),
    'Медиана': df.median(),
    'Квантиль 0.01': df.quantile(0.01),
    'Квантиль 0.05': df.quantile(0.05),
    'Квантиль 0.95': df.quantile(0.95),
    'Квантиль 0.99': df.quantile(0.99)
})

print("\nСтатистики:")
print(sts)

cols = df.select_dtypes(include=[np.number]).columns

for col in cols:
    plt.figure(figsize=(4, 4))
    plt.hist(df[col].dropna(), bins=100, color='yellow', edgecolor='black')
    plt.title(f'Гистограмма распределения: {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.show()
    var = df[col].var()
    skwn = df[col].skew()

    print(f"{col}: дисперсия = {var:.4f}, асимметрия = {skwn:.4f}")

    if abs(skwn) < 0.2:
        print("→ Распределение симметричное.\n")
    elif skwn > 0.2:
        print("→ Распределение скошено вправо.\n")
    else:
        print("→ Распределение скошено влево.\n")

corr_ma = df.corr(method='pearson')
print("\nМатрица корреляций Пирсона")
print(corr_ma)

# Проверка
p_val = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(len(df.columns))
p_val_df = pd.DataFrame(p_val, columns=df.columns, index=df.columns)

print("\nP-значения для корреляций")
print(p_val_df)

a = 0.05
signf = p_val_df < a
print(signf)

edge = 0.7  # порог сильной корреляции
s_pairs = np.where((np.abs(corr_ma) > edge) & signf)

p_pairs = set()
for i, j in zip(*s_pairs):
    if i != j and (j, i) not in p_pairs:
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=df.columns[i], y=df.columns[j], color='green', data=df)
        plt.title(f'Сильная корреляция: {df.columns[i]} vs {df.columns[j]} (r={corr_ma.iloc[i, j]:.2f})')
        plt.show()
        p_pairs.add((i, j))


