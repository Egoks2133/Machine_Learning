# Лабораторная работа 4. Основы нейронных сетей. Вариант №15
## Задание
Работа ведется в с датасетом "Phishing Websites"

1. Написать программу, которая разделяет исходную выборку на обучающую и тестовую (training set, test set).
2. Произвести масштабирование признаков (scaling).
3. С использованием библиотеки scikit-learn обучить 2 модели нейронной сети (Perceptron и MLPClassifier) по обучающей выборке. Перед обучением необходимо осуществить масштабирование признаков.
4. Проверить точность модели по тестовой выборке.
5. Провести эксперименты и определить наилучшие параметры коэффициента обучения, параметра регуляризации, функции оптимизации. Данные экспериментов необходимо представить в отчете (графики, ход проведения эксперимента, выводы).


## 1) Разделение на тестовую и обучающую выборки
В документации sklearn под тестовую выборку выделяется 20% данных. В программе поступим следующим образом: найдем общее количество данных и создадим массив индексов, после перемешаем индексы случайным образом и разделим их из расчета 80% train, 20% test. Так же дополнительно выведем количество данных в обеих выборках

```
# =========
# 1) Разделение на тестовую и обучающую выборки
# =========

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```


## 2) Масштабирование признаков
Масштабирование признаков — это процесс приведения всех числовых признаков к одинаковому масштабу. В данном коде используется стандартизация - один из самых популярных методов масштабирования.

scaler = StandardScaler() — Создание объекта StandardScaler
X_train_scaled = scaler.fit_transform(X_train) — Обучение scaler на тренировочных данных и преобразование тренировочных данных
X_test_scaled = scaler.transform(X_test) — Преобразование тестовых данных с использованием параметров, полученных на тренировочных данных

```
# =========
# 2) Масштабирование признаков
# =========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


## 3) Обучение Perceptron и MLPClassifier
Perceptron(Однослойный перцептрон) — модель, имеющая только входной и выходной слои, без скрытых слоев
MLPClassifier (Многослойный перцептрон) — модель, имеющая только входной, выходной и скрытые слои

```
# =========
# 3) Обучение Perceptron и MLPClassifier
# =========
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

print(f"Точность Perceptron: {accuracy_perceptron:.4f}")
print(f"Точность MLPClassifier: {accuracy_mlp:.4f}")
print("\n" + "="*100 + "\n")
```

Получаем уследующие значения как для обучающей, так и для тестовой выборки. Модель обучена.
<p align="center">
  <img src="Screen_2.png" />
</p>


## 4) Построение модели с использованием полиномиальной регрессии
Полиномиальная регрессия — это метод машинного обучения, используемый для моделирования нелинейных зависимостей между переменными путем аппроксимации данных полиномом степени (k). В отличие от линейной регрессии, которая предполагает прямую связь, этот метод позволяет учитывать более сложные криволинейные тренды в данных.

Создаём pipeline - объект, который автоматизирует процесс трансформации признаков в полиномиальные. Он будет применять шаги poly_features и linear_regression по порядку. Иначе нам пришлось бы вручную использовать методы.

```
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
```

По графику можно сказать, что самой подходящей степенью полинома будет 1:
<p align="center">
  <img src="Screen_3.png" />
</p>


## 5) Построение модели с использованием регуляризации
Ridge-регрессия (или гребневая регрессия) — это регуляризованная версия линейной регрессии, которая используется для борьбы с мультиколлинеарностью (линейной зависимостью между предикторами) и переобучением. Она добавляет к функции потерь штрафное слагаемое в виде квадрата коэффициентов, чтобы сделать веса модели меньше, но не обнулить их, в отличие от Lasso-регрессии. Алгоритм обучения пытается найти баланс между подгонкой под данные и поддержанием коэффициентов на низком уровне.

```
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
```

Для борьбы с переобучением и коррелированными признаками применим регуляризацию Ridge (L2). Она добавляет штраф к сумме квадратов коэффициентов модели, что стабилизирует обучение и уменьшает влияние сильно коррелированных признаков. В качестве признаков используем полиномиальные признаки 1-ой степени, как в предыдущем пункте (средняя точность и меньше затрат в вычислительной мощности). Обновим наш pipeline, который теперь выполняет следующие шаги:

PolynomialFeatures(degree=degree, include_bias=False) – создаёт новые признаки, учитывающие комбинации и степени всех признаков. StandardScaler() – нормализует признаки. Это важно, так как L2-регуляризация чувствительна к масштабу признаков. Ridge(alpha=alpha, max_iter=10000) – линейная регрессия с L2-регуляризацией. Параметр α контролирует силу штрафа: чем выше α, тем сильнее регуляризация и меньше риск переобучения.

Для подбора оптимального значения α выбираем диапазон от 10⁻⁴ до 10³ в логарифмической шкале.На графике можно увидеть, при каком αlpha модель лучше всего балансирует между переобучением и недообучением. Оптимальное значение α соответствует максимальному R² на тестовой выборке.


<p align="center">
  <img src="Screen_4.png" />
</p>

<p align="center">
  <img src="Screen_5.png" />
</p>
