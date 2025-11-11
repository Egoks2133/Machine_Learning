from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета Phishing Websites
phishing_websites = fetch_ucirepo(id=327)

# Данные
X = phishing_websites.data.features
y = phishing_websites.data.targets

# =========
# 1) Разделение на тестовую и обучающую выборки
# =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Преобразование в массив
y_train = np.array(y_train).ravel()
y_test = np.array(y_test).ravel()

# =========
# 2) Масштабирование признаков
# =========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Размер тренировочной выборки: {X_train_scaled.shape}")
print(f"Размер тестовой выборки: {X_test_scaled.shape}")
print("\n" + "="*100 + "\n")

# =========
# 3) Обучение Perceptron и MLPClassifier
# =========
#Perceptron
print("Обучение Perceptron...")
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)

# MLPClassifier с базовыми параметрами
print("Обучение MLPClassifier...")
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# =========
# 4) Проверка точности моделей
# =========
print(f"Точность Perceptron: {accuracy_perceptron:.4f}")
print(f"Точность MLPClassifier: {accuracy_mlp:.4f}")
print("\n" + "="*100 + "\n")

# =========
# 5) Подбор гиперпараметров
# =========
print("Запуск GridSearchCV для подбора гиперпараметров...")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
                           param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train_scaled, y_train)

print("\nЛучшие параметры:", grid_search.best_params_)
print("Лучшая точность на валидационной выборке: {:.4f}".format(grid_search.best_score_))

# Оценка лучшей модели на тестовых данных
best_mlp = grid_search.best_estimator_
y_pred_best = best_mlp.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Точность лучшей модели на тестовых данных: {accuracy_best:.4f}")

# Визуализация результатов
results = pd.DataFrame(grid_search.cv_results_)

# Создание фигуры с графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Результаты экспериментов
ax1.plot(results['mean_test_score'], marker='o', alpha=0.7)
ax1.set_xlabel('Номер эксперимента')
ax1.set_ylabel('Средняя точность')
ax1.set_title('Результаты перебора гиперпараметров')
ax1.grid(True, alpha=0.3)

# 2. Сравнение моделей
models = ['Perceptron', 'MLP (базовый)', 'MLP (лучший)']
accuracies = [accuracy_perceptron, accuracy_mlp, accuracy_best]
bars = ax2.bar(models, accuracies, color=['lightblue', 'lightcoral', 'lightgreen'])
ax2.set_ylabel('Точность')
ax2.set_title('Сравнение точности моделей')
ax2.set_ylim(0, 1)
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Сравнение с базовыми моделями
print("\n" + "="*100)
print("ИТОГОВОЕ СРАВНЕНИЕ:")
print(f"Perceptron: {accuracy_perceptron:.4f}")
print(f"MLPClassifier (базовый): {accuracy_mlp:.4f}")
print(f"MLPClassifier (оптимизированный): {accuracy_best:.4f}")