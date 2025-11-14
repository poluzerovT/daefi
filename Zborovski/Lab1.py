import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("Annual 2005-2011.csv")
print("Размер данных:", df.shape)

# 1. ПРЕДВАРИТЕЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ
print("\n" + "=" * 80)
print("1. ПРЕДВАРИТЕЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ")
print("=" * 80)

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

print("\nОписательные статистики (соответствует табл.1 методички):")
print(sts.round(4))

# 2. ГРАФИЧЕСКИЙ АНАЛИЗ РАСПРЕДЕЛЕНИЙ
print("\n" + "=" * 80)
print("2. ГРАФИЧЕСКИЙ АНАЛИЗ РАСПРЕДЕЛЕНИЙ")
print("=" * 80)

cols = df.select_dtypes(include=[np.number]).columns

print(f"\nАнализ распределений для {len(cols)} переменных")
print("Используйте стрелки ← → для навигации, ESC для выхода")


class HistogramNavigator:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns
        self.current_index = 0
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        col = self.columns[self.current_index]
        col_data = self.data[col].dropna()

        n, bins, patches = self.ax.hist(col_data, bins=50, color='lightblue',
                                        edgecolor='navy', alpha=0.7)

        stats_text = (f'Минимум: {col_data.min():.3f}\n'
                      f'Максимум: {col_data.max():.3f}\n'
                      f'Среднее: {col_data.mean():.3f}\n'
                      f'Медиана: {col_data.median():.3f}\n'
                      f'Стд: {col_data.std():.3f}\n'
                      f'Асимметрия: {col_data.skew():.3f}')


        skewness = col_data.skew()
        if abs(skewness) < 0.2:
            skew_type = "симметричное"
        elif skewness > 0.2:
            skew_type = "скошено вправо"
        else:
            skew_type = "скошено влево"

        self.ax.set_title(f'Гистограмма распределения: {col}\n({self.current_index + 1}/{len(self.columns)})',
                          fontsize=14, pad=20)
        self.ax.set_xlabel('Значения', fontsize=12)
        self.ax.set_ylabel('Частота', fontsize=12)
        self.ax.grid(True, alpha=0.3)


        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.ax.text(0.02, 0.75, f'Тип распределения: {skew_type}',
                     transform=self.ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        self.ax.text(0.02, 0.02, '← предыдущий | следующий → | ESC - закрыть',
                     transform=self.ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.columns)
            self.update_plot()
        elif event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.columns)
            self.update_plot()
        elif event.key == 'escape':
            plt.close()


hist_navigator = HistogramNavigator(df, cols)
plt.show()

# 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
print("\n" + "=" * 80)
print("3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 80)

corr_matrix = df.corr(method='pearson')
print("\nМатрица парных коэффициентов корреляции (аналог табл.2 методички):")
print(corr_matrix.round(3))


p_values = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(len(df.columns))
p_values_df = pd.DataFrame(p_values, columns=df.columns, index=df.columns)

print("\nМатрица p-значений:")
print(p_values_df.round(4))

alpha = 0.05
significant = p_values_df < alpha

strong_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if (abs(corr_matrix.iloc[i, j]) > 0.7 and significant.iloc[i, j]):
            strong_pairs.append((i, j, corr_matrix.iloc[i, j]))

if strong_pairs:
    print(f"\nНайдено {len(strong_pairs)} сильных корреляций (|r| > 0.7):")
    for i, j, corr_val in strong_pairs:
        print(f"  {df.columns[i]} ↔ {df.columns[j]}: r = {corr_val:.3f}")

    print(f"\nНавигация по scatter plots (используйте стрелки ← → для листания):")


    class ScatterNavigator:
        def __init__(self, data, columns, pairs):
            self.data = data
            self.columns = columns
            self.pairs = pairs
            self.current_index = 0
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.update_plot()

        def update_plot(self):
            self.ax.clear()
            i, j, corr_val = self.pairs[self.current_index]
            col_x = self.columns[i]
            col_y = self.columns[j]


            self.ax.scatter(self.data[col_x], self.data[col_y],
                            alpha=0.6, color='green', s=30)

            try:
                x_clean = self.data[col_x].dropna()
                y_clean = self.data[col_y].dropna()
                common_idx = x_clean.index.intersection(y_clean.index)
                if len(common_idx) > 1:
                    x_common = x_clean.loc[common_idx]
                    y_common = y_clean.loc[common_idx]
                    z = np.polyfit(x_common, y_common, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(x_common.min(), x_common.max(), 100)
                    self.ax.plot(x_range, p(x_range), "r-", linewidth=2, alpha=0.8,
                                 label=f'Линия тренда (r={corr_val:.3f})')
                    self.ax.legend()
            except:
                pass

            self.ax.set_xlabel(col_x, fontsize=12)
            self.ax.set_ylabel(col_y, fontsize=12)
            self.ax.set_title(f'Сильная корреляция: {col_x} ↔ {col_y}\n'
                              f'r = {corr_val:.3f}, p = {p_values_df.iloc[i, j]:.4f}\n'
                              f'({self.current_index + 1}/{len(self.pairs)})',
                              fontsize=14, pad=20)
            self.ax.grid(True, alpha=0.3)

            self.ax.text(0.02, 0.02, '← предыдущий | следующий → | ESC - закрыть',
                         transform=self.ax.transAxes, fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            self.fig.tight_layout()
            self.fig.canvas.draw()
            print(f"  {col_x} ↔ {col_y}: r = {corr_val:.3f}")

        def on_key_press(self, event):
            if event.key == 'right':
                self.current_index = (self.current_index + 1) % len(self.pairs)
                self.update_plot()
            elif event.key == 'left':
                self.current_index = (self.current_index - 1) % len(self.pairs)
                self.update_plot()
            elif event.key == 'escape':
                plt.close()

    scatter_navigator = ScatterNavigator(df, df.columns, strong_pairs)
    plt.show()

else:
    print("\nСильных корреляций не обнаружено")

# 4. ВИЗУАЛИЗАЦИЯ МАТРИЦЫ КОРРЕЛЯЦИЙ
print("\n" + "=" * 80)
print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхнего треугольника
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths= 0.5, cbar_kws={'shrink': 0.8})
plt.title('Матрица корреляций финансовых коэффициентов\n(аналог табл.2 методички)', fontsize=15, pad=20)
plt.tight_layout()
plt.show()

# 5. ДЕТАЛЬНЫЙ АНАЛИЗ СИЛЬНЫХ КОРРЕЛЯЦИЙ
if strong_pairs:
    print("\n" + "-" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ СИЛЬНЫХ КОРРЕЛЯЦИЙ")
    print("-" * 80)

    print("\nКоэффициенты с сильными корреляциями (|r| > 0.7):")
    correlation_summary = []
    for i, j, corr_val in strong_pairs:
        var1, var2 = df.columns[i], df.columns[j]
        p_val = p_values_df.iloc[i, j]
        correlation_summary.append({
            'Переменная 1': var1,
            'Переменная 2': var2,
            'Коэффициент корреляции': f"{corr_val:.3f}",
            'P-значение': f"{p_val:.4f}",
            'Сила связи': 'Очень сильная' if abs(corr_val) > 0.8 else 'Сильная'
        })

    summary_df = pd.DataFrame(correlation_summary)
    print(summary_df.to_string(index=False))
