import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def generate_data():
    """Генерация данных и создание CSV файла"""
    x1 = np.linspace(0.1, 1.4, 500)  # Избегаем значений близких к pi/2
    x2 = np.linspace(0.2, 3.0, 500)  # Избегаем значений, где ctg(x2) не определен

    # Вычисление y = tg(x1) * ctg(x2)
    y = np.tan(x1) * (np.cos(x2) / np.sin(x2))

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

    df.to_csv('function_data.csv', index=False)
    print("Файл 'function_data.csv' успешно создан!")
    return df


def analyze_data(df):
    """Анализ данных и построение графиков"""
    print("\nСтатистика по столбцам:")
    for column in df.columns:
        print(f"{column}:")
        print(f"  Среднее: {df[column].mean():.4f}")
        print(f"  Минимальное: {df[column].min():.4f}")
        print(f"  Максимальное: {df[column].max():.4f}")
        print()


def plot_2d_graphs(df):
    """Построение 2D графиков"""
    x2_const = 1.0
    x1_const = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(df['x1'], df['y'], alpha=0.6, color='blue', s=10)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('y')
    ax1.set_title(f'График y(x1) при x2 = {x2_const:.2f}')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(df['x2'], df['y'], alpha=0.6, color='red', s=10)
    ax2.set_xlabel('x2')
    ax2.set_ylabel('y')
    ax2.set_title(f'График y(x2) при x1 = {x1_const:.2f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def filter_and_save_data(df):
    """Фильтрация и сохранение данных"""
    x1_mean = df['x1'].mean()
    x2_mean = df['x2'].mean()
    filtered_df = df[(df['x1'] < x1_mean) | (df['x2'] < x2_mean)]

    filtered_df.to_csv('filtered_data.csv', index=False)
    print(f"Отфильтрованные данные сохранены в 'filtered_data.csv'")
    print(f"Исходных строк: {len(df)}, отфильтрованных строк: {len(filtered_df)}")
    print(f"Условие фильтрации: x1 < {x1_mean:.4f} ИЛИ x2 < {x2_mean:.4f}")

    return filtered_df


def plot_3d_graph(df):
    """Построение 3D графика"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 500)
    x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 500)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    Y = np.tan(X1) * (np.cos(X2) / np.sin(X2))

    surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)

    scatter = ax.scatter(df['x1'], df['x2'], df['y'], c=df['y'], cmap='viridis', alpha=0.8, s=30, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('3D график функции y = tg(x1) * ctg(x2)')

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='y')

    plt.tight_layout()
    plt.show()

def main():
    """Основная функция программы"""
    print("=== Генерация и анализ данных функции y = tan(x1) * ctg(x2) ===\n")

    # Шаг 1: Генерация данных
    if not os.path.exists('function_data.csv'):
        df = generate_data()
    else:
        df = pd.read_csv('function_data.csv')
        print("Файл 'function_data.csv' уже существует, загружаем данные...")

    # Шаг 2: Анализ данных
    analyze_data(df)

    # Шаг 3: Построение 2D графиков
    print("Построение 2D графиков...")
    plot_2d_graphs(df)

    # Шаг 4: Фильтрация и сохранение данных
    print("Фильтрация данных...")
    filtered_df = filter_and_save_data(df)

    # Шаг 5: Построение 3D графика
    print("Построение 3D графика...")
    plot_3d_graph(df)

    print("\nПрограмма завершена успешно!")


if __name__ == "__main__":
    main()