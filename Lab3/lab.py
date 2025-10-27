from ucimlrepo import fetch_ucirepo
import random
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def extract_float_value(value):
    if isinstance(value, str):
        if ':' in value:
            try:
                return float(value.split(':')[-1])
            except (ValueError, IndexError):
                return np.nan
        elif ';' in value:
            try:
                return float(value.split(';')[-1])
            except (ValueError, IndexError):
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan


def data_preparation():
    istanbul_stock_exchange = fetch_ucirepo(id=247)

    x_df = istanbul_stock_exchange.data.features.copy()

    if istanbul_stock_exchange.data.targets is not None:
        y_df = istanbul_stock_exchange.data.targets.copy()
    else:
        # Если targets нет, используем последний столбец features как целевую переменную
        y_df = x_df.iloc[:, -1].copy()
        x_df = x_df.iloc[:, :-1]  # Убираем последний столбец из признаков

    # Применяем функцию преобразования к признакам
    x_df = x_df.map(extract_float_value)
    # Заполнение NaN: для каждого столбца заполняем медианой, если столбец не полностью NaN, иначе 0.
    x_df = x_df.apply(lambda col: col.fillna(col.median() if not col.isnull().all() else 0))
    x_df = x_df.astype(float)

    # Обработка целевой переменной
    if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
        y_df = y_df.iloc[:, 0]
    elif isinstance(y_df, pd.DataFrame) and y_df.shape[1] > 1:
        # Используем первый столбец как целевую переменную
        y_df = y_df.iloc[:, 0]
    elif not isinstance(y_df, pd.Series):
        # Если это не Series и не DataFrame, преобразуем
        y_df = pd.Series(y_df)

    y_df = y_df.map(extract_float_value)
    y_df = y_df.fillna(y_df.median() if not y_df.isnull().all() else 0)
    y_df = y_df.astype(float)

    return x_df, y_df


x, y = data_preparation()

# =========
# 1) Разделение на тестовую и обучающие выборки
# =========

# Общее количество образцов
n_samples = x.shape[0]

# Массив индексов от 0 до n_samples-1
indices = np.arange(n_samples)

# Перемешиваем индексы случайным образом
np.random.shuffle(indices)

# Применяем перемешанные индексы к x и y (DataFrame/Series)
x_shuffled = x.iloc[indices]
y_shuffled = y.iloc[indices]

# Разделяем перемешанные данные из расчета 80% train, 20% test
train_size = int(n_samples * 0.8)

x_train = x_shuffled[:train_size]
x_test = x_shuffled[train_size:]

y_train = y_shuffled[:train_size]
y_test = y_shuffled[train_size:]

print(f"Размер обучающей выборки: {x_train.shape}")
print(f"Размер тестовой выборки: {x_test.shape}")

# =========
# 2) Обучение модели линейной регрессии
# =========

# Создаем объект класса LinearRegression и запускаем алгоритм градиентного спуска
regressor = LinearRegression().fit(x_train, y_train)

# Предсказания
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

# =========
# 3) Проверка модели на тестовой выборке
# =========
# В качестве метрики проверки будем использовать коэфициент детерминации -
# статистический показатель, который показывает долю изменчивости зависимой переменной


print(f"Коэффициэнт детерминации TRAIN: {r2_score(y_train, y_train_pred):.2f}")
print(f"Коэффициэнт детерминации TEST: {r2_score(y_test, y_test_pred):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="black", alpha=0.6, label="Фактические vs. Прогноз")

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle='--', linewidth=2,
         label="Идеальный прогноз (y=x)")

plt.xlabel("Истинные значения (y_test)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.title("Истинные значения vs. Предсказанные значения (Линейная регрессия)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# =========
# 4) Построение модели с использованием полиномиальной регрессии
# =========

degrees = range(1, 4)
r2_train_list = []
r2_test_list = []

#Поочередное обучение модели с разной степенью полинома
for degree in degrees:
    print(f"Обучение полиномиальной регрессии степени {degree}")
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_regression", LinearRegression())
    ])

    pipeline.fit(x_train, y_train)

    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    r2_train_list.append(r2_train)
    r2_test_list.append(r2_test)

    print(f"  R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(degrees, r2_train_list, marker='o', label="Train R2")
plt.plot(degrees, r2_test_list, marker='o', label="Test R2")
plt.xlabel("Degree of Polynomial Features")
plt.ylabel("R^2")
plt.title("Полиномиальная регрессия")
plt.legend()
plt.grid(True)
plt.show()

# Выбираем лучшую степень на основе тестовой R2
if len(r2_test_list) > 0:
    best_degree_index = np.argmax(r2_test_list)
    best_degree = degrees[best_degree_index]
    print(f"Наилучшая степень полинома: {best_degree}")
else:
    best_degree = 1
    print(f"Используется степень по умолчанию: {best_degree}")

# =========
# 5) Построение модели с использованием регуляризации
# =========

degree = best_degree  # Используем лучшую степень
alphas = np.logspace(-4, 3, 10)  # диапазон коэффициентов регуляризации

r2_train_list = []
r2_test_list = []

for alpha in alphas:
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, max_iter=10000))
    ])

    # Обучаем модель на train
    pipeline.fit(x_train, y_train)

    # Предсказания
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    # R^2
    r2_train_list.append(r2_score(y_train, y_train_pred))
    r2_test_list.append(r2_score(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.semilogx(alphas, r2_train_list, marker='o', label="Train R^2")
plt.semilogx(alphas, r2_test_list, marker='o', label="Test R^2")
plt.xlabel("Alpha (коэффициент регуляризации)")
plt.ylabel("R^2")
plt.title(f"Ridge Regression (Polynomial degree={degree})")
plt.grid(True)
plt.legend()
plt.show()

if len(r2_test_list) > 0:
    best_index = np.argmax(r2_test_list)
    best_alpha = alphas[best_index]
    print(f"Наилучший alpha: {best_alpha:.4f}")
    print(f"Наилучшее R2 на тесте с Ridge: {r2_test_list[best_index]:.4f}")
else:
    print("Не удалось вычислить R2 для Ridge регрессии")

# Сравнение всех моделей
print("\n===== СРАВНЕНИЕ МОДЕЛЕЙ =====")
linear_r2_test = r2_score(y_test, LinearRegression().fit(x_train, y_train).predict(x_test))
print(f"Линейная регрессия - R2 test: {linear_r2_test:.4f}")

if len(r2_test_list) > 0 and best_degree_index < len(r2_test_list):
    poly_r2_test = r2_test_list[best_degree_index]
    print(f"Полиномиальная регрессия (degree={best_degree}) - R2 test: {poly_r2_test:.4f}")

if len(r2_test_list) > 0:
    ridge_r2_test = r2_test_list[best_index]
    print(f"Ridge регрессия (alpha={best_alpha:.4f}) - R2 test: {ridge_r2_test:.4f}")