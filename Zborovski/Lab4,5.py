import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv(r"D:\python  task\Zborovski\Annual 2005-2011.csv")

coef_columns = [f"k{i}" for i in range(1, 21)]
features = data[coef_columns].copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

cluster_model = KMeans(n_clusters=4, random_state=12)
raw_clusters = cluster_model.fit_predict(scaled_features)

data["seg"] = raw_clusters


cluster_profile_strength = (
    data.groupby("seg")[coef_columns].mean().mean(axis=1)
)

sorted_segments = cluster_profile_strength.sort_values(ascending=False).index

segment_mapping = {old: new_id for new_id, old in enumerate(sorted_segments, start=1)}

data["rating"] = data["seg"].map(segment_mapping)


test_parts = []
for grp, block in data.groupby("rating"):
    part = block.sample(frac=0.10, random_state=12)
    test_parts.append(part)

test_split = pd.concat(test_parts, ignore_index=True)
train_split = data.drop(test_split.index)


X_train = train_split[coef_columns]
y_train = train_split["rating"]

X_test = test_split[coef_columns]
y_test = test_split["rating"]


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

train_predictions = lda.predict(X_train)
test_predictions = lda.predict(X_test)

print("Дискриминантный анализ\n")

print("Точность (train):", accuracy_score(y_train, train_predictions))
print("Точность (test): ", accuracy_score(y_test, test_predictions))

print("\nМатрица ошибок (test):")
print(confusion_matrix(y_test, test_predictions))

print("\nПодробный отчёт по классам:")
print(classification_report(y_test, test_predictions))


scaler2 = StandardScaler()
X_train_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)
logit_model = LogisticRegression(
    solver="lbfgs",
    max_iter=5000
)

logit_model.fit(X_train_scaled, y_train)
logit_predictions = logit_model.predict(X_test_scaled)


print("Мультиномиальная логит-модель\n")

print("Точность (test):", accuracy_score(y_test, logit_predictions))

print("\nМатрица ошибок логит-модели:")
print(confusion_matrix(y_test, logit_predictions))

print("\nКлассификационный отчёт:")
print(classification_report(y_test, logit_predictions))

print("\nКоэффициенты модели (для каждого рейтинга):")
coeffs = pd.DataFrame(logit_model.coef_, columns=coef_columns)
coeffs["Класс"] = logit_model.classes_
print(coeffs)
