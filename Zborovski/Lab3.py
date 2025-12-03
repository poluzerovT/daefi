import pandas as pd
import seaborn as sns
import pyreadstat
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from factor_analyzer import FactorAnalyzer
from factor_analyzer.rotator import Rotator
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 1. Загрузка данных
df = pd.read_csv('Annual 2005-2011.csv')
df = df.reset_index(drop=True)
df.index = df.index + 1

print(f"Размер данных: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())

# 2. Выбор переменных
cols = [f'k{i}' for i in range(1, 21) if f'k{i}' in df.columns]  # k1-k20
X = df[cols]
print(f"\nПеременные для анализа: {cols}")
print(f"Размер данных: {X.shape}")
print(X.head())

# 3. Предобработка данных
X_pca = X.dropna()
print(f"\nРазмер данных после удаления пропущенных значений: {X_pca.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# 4. Факторный анализ
print("\nВыполнение факторного анализа...")
pca_model = PCA(n_components=5)
pca_model.fit(X_scaled)

eigenvalues_initial = pca_model.explained_variance_[:5]
variance_ratio_initial = pca_model.explained_variance_ratio_[:5]
variance_percent_initial = variance_ratio_initial * 100
cumulative_variance_initial = np.cumsum(variance_percent_initial)

total_variance = eigenvalues_initial.sum() / variance_ratio_initial.sum()

component_matrix = pca_model.components_
factor_loadings = component_matrix.T * np.sqrt(eigenvalues_initial)

rotation = Rotator(method='quartimax')
rotated_loadings = rotation.fit_transform(factor_loadings)

eigenvalues_rotated = np.sum(rotated_loadings ** 2, axis=0)
variance_percent_rotated = eigenvalues_rotated / total_variance * 100
cumulative_variance_rotated = np.cumsum(variance_percent_rotated)

factor_results = pd.DataFrame({
    'Фактор': range(1, 6),
    'Собственное значение': eigenvalues_rotated,
    'Доля дисперсии %': variance_percent_rotated,
    'Накопленная доля %': cumulative_variance_rotated
}, index=[f'F{i + 1}' for i in range(5)]).round(3)

print("Результаты факторного анализа после вращения:")
print(factor_results)
print(f"\nВсего объяснено дисперсии: {cumulative_variance_rotated[-1]:.1f}%")

# 5. Факторные значения
factor_values = pca_model.transform(X_scaled)[:, :5]
factor_scores_table = pd.DataFrame(
    factor_values,
    index=X_pca.index,
    columns=[f'Фактор{i + 1}' for i in range(5)]
)

print("\nФакторные значения для первых 10 наблюдений:")
print(factor_scores_table.head(10))

# 6. Расчет интегрального показателя I
weights = factor_results['Доля дисперсии %'] / 100
I = pd.Series(0.0, index=factor_scores_table.index)
for i, factor in enumerate(['Фактор1', 'Фактор2', 'Фактор3', 'Фактор4', 'Фактор5']):
    I += weights.iloc[i] * factor_scores_table[factor]

factor_scores_table.insert(0, "I", I)

# 7. Создание финальной таблицы
final_df = pd.DataFrame(index=factor_scores_table.index)
final_df['I'] = factor_scores_table['I']
final_df = pd.concat([final_df, X.loc[X_pca.index]], axis=1)
final_df = pd.concat([final_df, factor_scores_table[['Фактор1', 'Фактор2', 'Фактор3', 'Фактор4', 'Фактор5']]], axis=1)

print("\nФинальная таблица результатов (первые 10 строк):")
print(final_df.round(3).head(10))

# 8. Кластерный анализ (используем 4 кластера как выбрано)
print("\n" + "=" * 60)
print("КЛАСТЕРНЫЙ АНАЛИЗ С 4 КЛАСТЕРАМИ")
print("=" * 60)

X_cluster = X_scaled  # Используем стандартизованные данные
n_clusters = 4

kmeans = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    n_init=10,
    random_state=42
)
kmeans.fit(X_cluster)

print(f"\nКластеризация завершена:")
print(f"Количество итераций: {kmeans.n_iter_}")
print(f"Inertia: {kmeans.inertia_:.2f}")

# 9. Центроиды кластеров
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    index=range(1, n_clusters + 1),
    columns=cols
)
print("\nЦентроиды кластеров (в исходных единицах измерения):")
print(centers.round(4))

# 10. Добавление кластеров в финальную таблицу
final_df['Cluster'] = kmeans.labels_ + 1  # Кластеры 1-4

# 11. Анализ распределения по кластерам
cluster_distribution = final_df['Cluster'].value_counts().sort_index()
total = len(final_df)

result_df = pd.DataFrame({
    'Кол-во наблюдений': cluster_distribution,
    'Процент': (cluster_distribution / total * 100).round(1)
})

print("\nРаспределение по кластерам:")
print(result_df)

# 12. Описательная статистика по кластерам
print("\nОписательная статистика по кластерам:")
cluster_stats = final_df.groupby('Cluster')['I'].agg([
    ('Количество', 'count'),
    ('Среднее', 'mean'),
    ('Медиана', 'median'),
    ('Стд. отклонение', 'std'),
    ('Минимум', 'min'),
    ('Максимум', 'max'),
    ('Q1', lambda x: x.quantile(0.25)),
    ('Q3', lambda x: x.quantile(0.75))
]).round(4)

print(cluster_stats)

# 13. Визуализация результатов - СОЗДАЕМ ТОЛЬКО НЕОБХОДИМЫЕ ГРАФИКИ
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Только 2 графика вместо 6

# График 1: Метод локтя
k_range = range(2, 11)
inertia_values = []
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_cluster)
    inertia_values.append(kmeans_temp.inertia_)

axes[0].plot(k_range, inertia_values, 'bo-', markersize=8, linewidth=2)
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k=4')
axes[0].set_xlabel('Количество кластеров')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Метод локтя')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# График 2: Кластеры в пространстве факторов
scatter = axes[1].scatter(final_df['Фактор1'], final_df['Фактор2'],
                         c=final_df['Cluster'], cmap='viridis', alpha=0.6, s=30)
axes[1].set_xlabel('Фактор 1')
axes[1].set_ylabel('Фактор 2')
axes[1].set_title('Кластеры в пространстве факторов 1 и 2')
axes[1].grid(True, alpha=0.3)

# Добавляем цветовую шкалу для кластеров
plt.colorbar(scatter, ax=axes[1], label='Кластер')

plt.suptitle('Результаты факторного и кластерного анализа', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

