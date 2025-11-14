import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def c_bool(val):
    c = 'white' if val else 'red'
    return f'background-color: {c}; text-align: center;'


pd.set_option('display.max_columns', None)
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

cols_for_hist = cols[:-1]

fig, axes = plt.subplots(3, 7, figsize=(25, 8))

for idx, col in enumerate(cols_for_hist):
    row = idx // 7
    col_idx = idx % 7

    if row < 3:
        axes[row, col_idx].hist(df[col].dropna(), bins=100, color='yellow', edgecolor='black')
        axes[row, col_idx].set_title(f'{col}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Частота')

plt.tight_layout()
plt.show()

corr_ma = df.corr(method='pearson')
print("\nМатрица корреляций Пирсона")
print(corr_ma)

p_val = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(len(df.columns))
p_val_df = pd.DataFrame(p_val, columns=df.columns, index=df.columns)

print("\np-значения для корреляций")
print(p_val_df)

a = 0.05
signf = p_val_df < a
print("\nСтатистически значимые корреляции (p < 0.05):")
print(signf)

edge = 0.7
s_pairs = np.where((np.abs(corr_ma) > edge) & signf)

p_pairs = []
for i, j in zip(*s_pairs):
    if i != j and (j, i) not in p_pairs and (i, j) not in p_pairs:
        p_pairs.append((i, j))

if p_pairs:
    n_pairs = len(p_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))

    if n_pairs == 1:
        axes = [axes]

    for idx, (i, j) in enumerate(p_pairs):
        axes[idx].scatter(df[df.columns[i]], df[df.columns[j]], color='green', alpha=0.7)
        axes[idx].set_title(f'{df.columns[i]} vs {df.columns[j]}\n(r={corr_ma.iloc[i, j]:.2f})')
        axes[idx].set_xlabel(df.columns[i])
        axes[idx].set_ylabel(df.columns[j])

    plt.tight_layout()
    plt.show()