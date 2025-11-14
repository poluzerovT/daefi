import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.linalg import eigh


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

try:
    df = pd.read_csv("Annual 2005-2011.csv")
    print(f"Данные успешно загружены. Размерность: {df.shape}")


    print("\nИНФОРМАЦИЯ О ДАННЫХ:")
    print("=" * 100)
    print(df.info())

    print("\nПЕРВЫЕ 5 СТРОК ДАННЫХ:")
    print("=" * 100)
    print(df.head())

    print("\nОСНОВНАЯ СТАТИСТИКА:")
    print("=" * 100)
    print(df.describe())

    print("\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
    print("=" * 100)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

except FileNotFoundError:
    print("Ошибка: Файл 'Annual 2005-2011.csv' не найден.")
    print("Проверьте путь к файлу и попробуйте снова.")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    exit()


print("\nВЫБОР ПЕРЕМЕННЫХ ДЛЯ АНАЛИЗА:")
print("=" * 100)


numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Найдено {len(numeric_columns)} числовых переменных:")
for i, col in enumerate(numeric_columns, 1):
    print(f"{i:2d}. {col}")


if len(numeric_columns) > 12:
    print("\nСлишком много переменных. Выбираем 12 наиболее информативных...")

    variances = df[numeric_columns].var().sort_values(ascending=False)
    selected_columns = variances.head(12).index.tolist()
else:
    selected_columns = numeric_columns

print(f"\nВыбрано {len(selected_columns)} переменных для анализа:")
for i, col in enumerate(selected_columns, 1):
    print(f"{i:2d}. {col}")


df_analysis = df[selected_columns].copy()


print("\nОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
print("=" * 100)
initial_shape = df_analysis.shape
df_analysis = df_analysis.dropna()
print(f"Удалено строк с пропущенными значениями: {initial_shape[0] - df_analysis.shape[0]}")
print(f"Осталось строк: {df_analysis.shape[0]}")

if df_analysis.shape[0] < 10:
    print("Ошибка: Слишком мало данных после очистки. Необходимо как минимум 10 наблюдений.")
    exit()



def standardize_data(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds, means, stds


def bartlett_sphericity_test(data):
    n, p = data.shape
    corr_matrix = np.corrcoef(data, rowvar=False)
    det_corr = np.linalg.det(corr_matrix)

    chi_square = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
    df = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_square, df)

    return chi_square, p_value, df


def kmo_test(data):
    corr_matrix = np.corrcoef(data, rowvar=False)
    inv_corr_matrix = np.linalg.inv(corr_matrix)

    r_sq = corr_matrix ** 2
    p_sq = np.zeros_like(corr_matrix)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i != j:
                p_sq[i, j] = -inv_corr_matrix[i, j] / np.sqrt(inv_corr_matrix[i, i] * inv_corr_matrix[j, j])

    p_sq = p_sq ** 2
    kmo_per_var = np.sum(r_sq, axis=1) / (np.sum(r_sq, axis=1) + np.sum(p_sq, axis=1))
    kmo_total = np.sum(r_sq) / (np.sum(r_sq) + np.sum(p_sq))

    return kmo_total, kmo_per_var


def pca_manual(data):
    centered_data = data - np.mean(data, axis=0)
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    principal_components = centered_data @ eigenvectors

    return principal_components, eigenvalues, eigenvectors, explained_variance_ratio, cumulative_variance_ratio


# 1. Факторный анализ и проверка применимости
print("\n" + "=" * 100)
print("1. ФАКТОРНЫЙ АНАЛИЗ И ПРОВЕРКА ПРИМЕНИМОСТИ")
print("=" * 100)


df_scaled, means, stds = standardize_data(df_analysis.values)

chi_square, p_value, df_bartlett = bartlett_sphericity_test(df_scaled)
print(f"Тест сферичности Бартлетта:")
print(f"Хи-квадрат: {chi_square:.4f}")
print(f"Степени свободы: {df_bartlett}")
print(f"p-value: {p_value:.10f}")

if p_value < 0.05:
    print("✓ Тест значим: данные подходят для факторного анализа")
else:
    print("✗ Тест не значим: данные могут не подходить для факторного анализа")

kmo_total, kmo_per_var = kmo_test(df_scaled)
print(f"\nKMO критерий: {kmo_total:.4f}")

if kmo_total >= 0.8:
    print("✓ Отличная применимость факторного анализа")
elif kmo_total >= 0.7:
    print("✓ Хорошая применимость факторного анализа")
elif kmo_total >= 0.6:
    print("✓ Средняя применимость факторного анализа")
elif kmo_total >= 0.5:
    print("✓ Низкая применимость факторного анализа")
else:
    print("✗ Неприемлемая применимость факторного анализа")

print("\nKMO по переменным:")
for i, var in enumerate(selected_columns):
    print(f"  {var}: {kmo_per_var[i]:.4f}")


principal_components, eigenvalues, eigenvectors, explained_variance, cumulative_variance = pca_manual(df_scaled)


n_factors = sum(eigenvalues > 1)
print(f"\nКоличество факторов по критерию Кайзера: {n_factors}")

print(f"\nСОБСТВЕННЫЕ ЗНАЧЕНИЯ:")
for i, eigenval in enumerate(eigenvalues, 1):
    print(f"  PC{i}: {eigenval:.4f}")

print(f"\nОБЪЯСНЕННАЯ ДИСПЕРСИЯ ПО КОМПОНЕНТАМ:")
for i, (var_ratio, cum_ratio) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"  PC{i}: {var_ratio * 100:6.2f}% (накопленно: {cum_ratio * 100:6.2f}%)")

# 2. Определение наиболее значимых коэффициентов
print("\n" + "=" * 100)
print("2. ЗНАЧИМЫЕ КОЭФФИЦИЕНТЫ И ДИСПЕРСИЯ")
print("=" * 100)


loadings = pd.DataFrame(
    eigenvectors,
    columns=[f'PC{i + 1}' for i in range(eigenvectors.shape[1])],
    index=selected_columns
)

print("НАГРУЗКИ ФАКТОРОВ (первые 5 компонент):")
print(loadings.iloc[:, :5].round(4))

significant_variables = {}

print(f"\nАНАЛИЗ ЗНАЧИМЫХ ПЕРЕМЕННЫХ ДЛЯ {n_factors} ОСНОВНЫХ КОМПОНЕНТ:")
for i in range(n_factors):
    pc_name = f'PC{i + 1}'
    significant_vars = loadings[loadings[pc_name].abs() > 0.4][pc_name]
    significant_variables[pc_name] = significant_vars

    print(f"\n--- Компонента {i + 1} (объясняет {explained_variance[i] * 100:.2f}% дисперсии) ---")
    sorted_vars = significant_vars.sort_values(key=abs, ascending=False)
    for var, loading in sorted_vars.items():
        interpretation = "↑ положительная связь" if loading > 0 else "↓ отрицательная связь"
        print(f"  {var}: {loading:.4f} ({interpretation})")

variance_retained = cumulative_variance[n_factors - 1] if n_factors > 0 else 0
print(
    f"\nОбъясненная дисперсия первыми {n_factors} компонентами: {variance_retained:.4f} ({variance_retained * 100:.2f}%)")

# 3. Расчет интегрального показателя кредитоспособности
print("\n" + "=" * 100)
print("3. ИНТЕГРАЛЬНЫЙ ПОКАЗАТЕЛЬ КРЕДИТОСПОСОБНОСТИ")
print("=" * 100)

if n_factors == 0:
    print("Предупреждение: Не найдено значимых факторов. Используем первую компоненту.")
    n_factors = 1

selected_components = principal_components[:, :n_factors]
selected_variance = explained_variance[:n_factors]
weights = selected_variance / selected_variance.sum()

print(f"Используемые компоненты: PC1 - PC{n_factors}")
print("Веса компонент:")
for i, weight in enumerate(weights, 1):
    print(f"  PC{i}: {weight:.6f}")
print(f"Сумма весов: {weights.sum():.6f}")

integral_score = np.zeros(df_analysis.shape[0])
for i in range(n_factors):
    integral_score += weights[i] * principal_components[:, i]

min_score = integral_score.min()
max_score = integral_score.max()
integral_score_normalized = 1000 * (integral_score - min_score) / (max_score - min_score)

df_results = df.copy()
df_results = df_results.iloc[df_analysis.index]
df_results['Интегральный_показатель'] = integral_score_normalized
df_results['Ранг_кредитоспособности'] = pd.qcut(integral_score_normalized, 5,
                                                labels=['Очень низкий', 'Низкий', 'Средний', 'Высокий',
                                                        'Очень высокий'])

print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА ИНТЕГРАЛЬНОГО ПОКАЗАТЕЛЯ:")
print(df_results['Интегральный_показатель'].describe().round(2))

print("\nТОП-10 НАИБОЛЕЕ КРЕДИТОСПОСОБНЫХ ЗАЕМЩИКОВ:")
top_columns = ['Интегральный_показатель', 'Ранг_кредитоспособности'] + selected_columns[:3]
top_borrowers = df_results.nlargest(10, 'Интегральный_показатель')[top_columns]
print(top_borrowers.round(2))


print("\nСОЗДАНИЕ ВИЗУАЛИЗАЦИЙ...")
print("=" * 100)

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. График собственных значений
ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', markersize=6, linewidth=1.5)
ax1.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='Порог Кайзера (λ=1)')
ax1.set_title('Критерий каменистой осыпи', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Номер главной компоненты')
ax1.set_ylabel('Собственное значение (λ)')
ax1.legend()
ax1.grid(True, alpha=0.3)


for i, (x, y) in enumerate(zip(range(1, len(eigenvalues) + 1), eigenvalues)):
    if i < 5:
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)

# 2. Объясненная дисперсия
components_to_show = min(8, len(explained_variance))
x_pos = np.arange(components_to_show) + 1

bars = ax2.bar(x_pos, explained_variance[:components_to_show],
               alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)

ax2_step = ax2.twinx()
ax2_step.step(x_pos, cumulative_variance[:components_to_show],
              where='mid', color='red', linewidth=2, marker='o',
              markersize=4)

ax2.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5,
            label='80% дисперсии')

ax2.set_title('Объясненная дисперсия', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('Главные компоненты')
ax2.set_ylabel('Доля дисперсии', color='skyblue')
ax2_step.set_ylabel('Накопленная', color='red')

for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0.05:
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold')

ax2.legend(['Объясненная', '80% порог'], loc='upper left')
ax2_step.legend(['Накопленная'], loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Распределение интегрального показателя
n_bins = min(20, len(integral_score_normalized) // 15)
ax3.hist(integral_score_normalized, bins=n_bins, alpha=0.7,
         color='lightgreen', edgecolor='darkgreen', linewidth=0.8)
ax3.set_title('Распределение показателя', fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel('Интегральный показатель')
ax3.set_ylabel('Количество')
ax3.grid(True, alpha=0.3)


mean_score = integral_score_normalized.mean()
median_score = np.median(integral_score_normalized)
ax3.axvline(mean_score, color='red', linestyle='--', linewidth=1.5,
            label=f'Среднее: {mean_score:.0f}')
ax3.axvline(median_score, color='blue', linestyle='--', linewidth=1.5,
            label=f'Медиана: {median_score:.0f}')
ax3.legend()

# 4. Распределение по рангам
rank_counts = df_results['Ранг_кредитоспособности'].value_counts().sort_index()
colors = ['#ff6b6b', '#ffa726', '#ffee58', '#90ee90', '#4caf50']
wedges, texts, autotexts = ax4.pie(rank_counts.values,
                                   labels=rank_counts.index,
                                   colors=colors,
                                   autopct='%1.1f%%',
                                   startangle=90)

ax4.set_title('Ранги кредитоспособности', fontsize=11, fontweight='bold', pad=10)


for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

plt.tight_layout(pad=3.0)
plt.show()


fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

# 5. Корреляция переменных с интегральным показателем
correlations = []
for var in selected_columns:
    corr = np.corrcoef(df_analysis[var], integral_score_normalized)[0, 1]
    correlations.append(corr)

corr_data = pd.DataFrame({
    'Переменная': selected_columns,
    'Корреляция': correlations
}).sort_values('Корреляция', ascending=True)

colors = ['red' if x < 0 else 'green' for x in corr_data['Корреляция']]
bars = ax5.barh(range(len(corr_data)), corr_data['Корреляция'],
                color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

ax5.set_title('Корреляция с показателем', fontsize=11, fontweight='bold', pad=10)
ax5.set_xlabel('Коэффициент корреляции')
ax5.set_yticks(range(len(corr_data)))
ax5.set_yticklabels(corr_data['Переменная'])
ax5.axvline(x=0, color='black', linewidth=0.8)
ax5.grid(True, alpha=0.3, axis='x')


for i, bar in enumerate(bars):
    width = bar.get_width()
    ax5.text(width + (0.02 if width >= 0 else -0.02), bar.get_y() + bar.get_height() / 2.,
             f'{width:.2f}', ha='left' if width >= 0 else 'right', va='center',
             fontsize=8, fontweight='bold')

# 6. Вклад компонент в интегральный показатель
if n_factors > 0:
    component_contributions = weights * 100
    colors_contrib = plt.cm.Set3(np.linspace(0, 1, n_factors))

    wedges, texts, autotexts = ax6.pie(component_contributions,
                                       labels=[f'PC{i + 1}' for i in range(n_factors)],
                                       colors=colors_contrib,
                                       autopct='%1.1f%%',
                                       startangle=90)

    ax6.set_title('Вклад компонент', fontsize=11, fontweight='bold', pad=10)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

plt.tight_layout(pad=3.0)
plt.show()


print("\nСОЗДАНИЕ ТЕПЛОВОЙ КАРТЫ КОРРЕЛЯЦИЙ...")

fig3, ax7 = plt.subplots(figsize=(12, 10))

corr_matrix = np.corrcoef(df_scaled, rowvar=False)


im = ax7.imshow(corr_matrix, cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)


for i in range(len(selected_columns)):
    for j in range(len(selected_columns)):
        if abs(corr_matrix[i, j]) > 0.3:
            text = ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha="center", va="center",
                            color="black" if abs(corr_matrix[i, j]) < 0.7 else "white",
                            fontsize=7, fontweight='bold')

ax7.set_xticks(range(len(selected_columns)))
ax7.set_yticks(range(len(selected_columns)))
ax7.set_xticklabels(selected_columns, rotation=45, ha='right', fontsize=8)
ax7.set_yticklabels(selected_columns, fontsize=8)
ax7.set_title('Матрица корреляций', fontsize=11, fontweight='bold', pad=10)


cbar = plt.colorbar(im, ax=ax7, shrink=0.8)
cbar.set_label('Корреляция', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()


output_filename = 'результаты_кредитоспособности_annual.csv'
df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 80)
print(f"Результаты сохранены в: '{output_filename}'")
print(f"Наблюдения: {df_analysis.shape[0]}")
print(f"Компоненты: {n_factors}")
print(f"Объясненная дисперсия: {variance_retained * 100:.1f}%")