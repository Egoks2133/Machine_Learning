import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Загрузка датасета
wholesale_customers = fetch_ucirepo(id=292)

# Данные
X = wholesale_customers.data.features
y = wholesale_customers.data.targets

# =========
# 1) Масштабирование признаков
# =========

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========
# 2) Опыты с алгоритмами кластеризации
# =========

# Функция для оценки кластеризации
def evaluate_clustering(model, data):
    labels = model.fit_predict(data)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
    else:
        silhouette = -1
        calinski_harabasz = -1
    return labels, silhouette, calinski_harabasz

# 2.1 KMeans
kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []
scores_kmeans = []

print("=== KMeans Clustering ===")
for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels, silhouette, calinski_harabasz = evaluate_clustering(kmeans, X_scaled)
    print(f'KMeans с k={k}, Силуэтный коэффициент: {silhouette:.3f}, Calinski-Harabasz: {calinski_harabasz:.1f}')
    labels_for_k.append(labels)
    scores_kmeans.append((silhouette, calinski_harabasz))
    if silhouette > best_score_kmeans:
        best_score_kmeans = silhouette
        best_kmeans = kmeans
        best_labels_kmeans = labels
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

# Визуализация KMeans для каждого k
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    silhouette, calinski = scores_kmeans[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    axes[i].set_title(f'KMeans k={k}\nSilhouette: {silhouette:.3f}\nCalinski-Harabasz: {calinski:.1f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

# Скрываем последний subplot если нужно
if len(kmeans_params) < 6:
    axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# 2.2 Agglomerative Clustering
agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None
labels_list_agg = []
scores_agg = []

print("\n=== Agglomerative Clustering ===")
for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, silhouette, calinski_harabasz = evaluate_clustering(agg, X_scaled)
    print(f'Agglomerative с n_clusters={n}, Силуэтный коэффициент: {silhouette:.3f}, Calinski-Harabasz: {calinski_harabasz:.1f}')
    labels_list_agg.append(labels)
    scores_agg.append((silhouette, calinski_harabasz))
    if silhouette > best_score_agg:
        best_score_agg = silhouette
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

# Визуализация Agglomerative для каждого n
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, n in enumerate(agg_params):
    labels = labels_list_agg[i]
    silhouette, calinski = scores_agg[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', s=30, alpha=0.7)
    axes[i].set_title(f'Agglomerative n={n}\nSilhouette: {silhouette:.3f}\nCalinski-Harabasz: {calinski:.1f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

if len(agg_params) < 6:
    axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# 2.3 Spectral Clustering
spectral_params = [2, 3, 4, 5, 6]
best_score_spectral = -1
best_labels_spectral = []
best_n_spectral = None
labels_list_spectral = []
scores_spectral = []

print("\n=== Spectral Clustering ===")
for n in spectral_params:
    spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
    labels, silhouette, calinski_harabasz = evaluate_clustering(spectral, X_scaled)
    print(f'SpectralClustering с n_clusters={n}, Силуэтный коэффициент: {silhouette:.3f}, Calinski-Harabasz: {calinski_harabasz:.1f}')
    labels_list_spectral.append(labels)
    scores_spectral.append((silhouette, calinski_harabasz))
    if silhouette > best_score_spectral:
        best_score_spectral = silhouette
        best_labels_spectral = labels
        best_n_spectral = n

print(f'Лучшее число кластеров для SpectralClustering: {best_n_spectral} с коэффициентом: {best_score_spectral:.3f}')

# Визуализация Spectral для каждого n
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, n in enumerate(spectral_params):
    labels = labels_list_spectral[i]
    silhouette, calinski = scores_spectral[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='cool', s=30, alpha=0.7)
    axes[i].set_title(f'Spectral n={n}\nSilhouette: {silhouette:.3f}\nCalinski-Harabasz: {calinski:.1f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

if len(spectral_params) < 6:
    axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# Сравнение лучших результатов каждого метода
plt.figure(figsize=(15, 5))

# Вычисляем Calinski-Harabasz для лучших конфигураций
best_calinski_kmeans = calinski_harabasz_score(X_scaled, best_labels_kmeans)
best_calinski_agg = calinski_harabasz_score(X_scaled, best_labels_agg)
best_calinski_spectral = calinski_harabasz_score(X_scaled, best_labels_spectral)

# Лучший KMeans
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='viridis', s=30, alpha=0.7)
plt.title(f'Лучший KMeans (k={best_k})\nSilhouette: {best_score_kmeans:.3f}\nCalinski-Harabasz: {best_calinski_kmeans:.1f}')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Лучший Agglomerative
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='plasma', s=30, alpha=0.7)
plt.title(f'Лучший Agglomerative (n={best_n_agg})\nSilhouette: {best_score_agg:.3f}\nCalinski-Harabasz: {best_calinski_agg:.1f}')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Лучший Spectral
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_spectral, cmap='cool', s=30, alpha=0.7)
plt.title(f'Лучший Spectral (n={best_n_spectral})\nSilhouette: {best_score_spectral:.3f}\nCalinski-Harabasz: {best_calinski_spectral:.1f}')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()

# Определение лучшего метода
scores = {
    'KMeans': best_score_kmeans,
    'Agglomerative': best_score_agg,
    'SpectralClustering': best_score_spectral
}

calinski_scores = {
    'KMeans': best_calinski_kmeans,
    'Agglomerative': best_calinski_agg,
    'SpectralClustering': best_calinski_spectral
}

print("\n=== Сравнение методов ===")
for method in scores.keys():
    print(f'{method}: Silhouette: {scores[method]:.3f}, Calinski-Harabasz: {calinski_scores[method]:.1f}')

best_method_silhouette = max(scores, key=scores.get)
best_method_calinski = max(calinski_scores, key=calinski_scores.get)

print(f'\nЛучший метод по Silhouette: {best_method_silhouette} с коэффициентом {scores[best_method_silhouette]:.3f}')
print(f'Лучший метод по Calinski-Harabasz: {best_method_calinski} с коэффициентом {calinski_scores[best_method_calinski]:.1f}')

# Дополнительная визуализация: сравнение метрик для всех k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# График Silhouette scores
ax1.plot(kmeans_params, [score[0] for score in scores_kmeans], 'o-', label='KMeans', linewidth=2)
ax1.plot(agg_params, [score[0] for score in scores_agg], 'o-', label='Agglomerative', linewidth=2)
ax1.plot(spectral_params, [score[0] for score in scores_spectral], 'o-', label='Spectral', linewidth=2)
ax1.set_title('Silhouette Score по количеству кластеров')
ax1.set_xlabel('Количество кластеров')
ax1.set_ylabel('Silhouette Score')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График Calinski-Harabasz scores
ax2.plot(kmeans_params, [score[1] for score in scores_kmeans], 'o-', label='KMeans', linewidth=2)
ax2.plot(agg_params, [score[1] for score in scores_agg], 'o-', label='Agglomerative', linewidth=2)
ax2.plot(spectral_params, [score[1] for score in scores_spectral], 'o-', label='Spectral', linewidth=2)
ax2.set_title('Calinski-Harabasz Score по количеству кластеров')
ax2.set_xlabel('Количество кластеров')
ax2.set_ylabel('Calinski-Harabasz Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()