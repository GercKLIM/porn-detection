# ========================
# == ИМПОРТ БИБЛИОТЕК ====
# ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC


# =============================================
# === ЗАГРУЗКА ДАННЫХ ТРЕНИРОВОЧГО ДАТАСЕТА ===
# =============================================

print("ИМПОРТ ДАТАСЕТА")

# Путь к файлу тренировочного датасета
train_ds_path = "datasets/train.csv"

# Получение треничровочго датасета
train_ds = pd.read_csv(train_ds_path, encoding='utf-8')

# Замена NaN на пустые строки
train_ds = train_ds.fillna('')



# ========================
# === ОБРАБОТКА ДАННЫХ ===
# ========================



# Параметры векторизации
max_tokens = 50000           # Максимальное количество уникальных слов
output_sequence_length = 100  # Длина последовательности после векторизации

# Создаем слой векторизации для url
url_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)

# Создаем слой векторизации для title
title_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)

# Адаптируем векторизаторы на данных
url_vectorizer.adapt(train_ds['url'])
title_vectorizer.adapt(train_ds['title'])

# Функция для преобразования данных
def preprocess_data(url, title):
    url_vec = url_vectorizer(url)
    title_vec = title_vectorizer(title)
    return tf.concat([url_vec, title_vec], axis=1)

# Применяем препроцессинг к данным
X_train = preprocess_data(train_ds['url'], train_ds['title'])
y_train = train_ds['label']



# ======================
# === СОЗДАЕМ МОДЕЛЬ ===
# ======================



# Параметры модели
embedding_dim = 64  # Размерность эмбеддингов
num_classes = 1  # Бинарная классификация

# Создаем модель
# model = Sequential([
#     Embedding(input_dim=max_tokens + 1, output_dim=embedding_dim, input_length=output_sequence_length * 2),
#     GlobalAveragePooling1D(),
#     Dense(16, activation='relu'),
#     Dense(num_classes, activation='sigmoid')  # Сигмоидная функция активации для бинарной классификации
# ])

# Создаем модель
model = Sequential([
    # Слой эмбеддингов
    Embedding(input_dim=max_tokens + 1, output_dim=embedding_dim, input_length=output_sequence_length * 2),

    # Сверточный слой для захвата локальных паттернов
    Conv1D(filters=256, kernel_size=5, activation='relu'),

    # Слой пулинга для снижения размерности
    MaxPooling1D(pool_size=4),

    # Dropout для предотвращения переобучения
    Dropout(0.5),

    # Глобальный пулинг для получения фиксированного представления
    GlobalMaxPooling1D(),

    # Полносвязный слой для извлечения признаков
    Dense(128, activation='relu'),

    # Dropout перед выходным слоем
    Dropout(0.5),

    # Выходной слой для бинарной классификации
    Dense(num_classes, activation='sigmoid')
])

# Компилируем модель
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Бинарная кросс-энтропия для бинарной классификации
    metrics=['accuracy']
)

# Выводим архитектуру модели
# model.summary()



# ======================
# === ОБУЧАЕМ МОДЕЛЬ ===
# ======================


# Обучение модели
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1  # Используем 20% данных для валидации
)

model.save("model_6.keras")



# ===================================
# === ВЫВОДИМ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===
# ===================================



# Оценка модели на тренировочной выборке
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=2)
print(f"Потери на тренировочной выборке: {train_loss:.4f}")
print(f"Точность на тренировочной выборке: {train_accuracy:.4f}")


# Извлекаем данные из history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Определяем количество эпох
epochs = range(1, len(train_loss) + 1)

# Создаем график для потерь
plt.figure(figsize=(12, 4))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Тренировочные потери')
plt.plot(epochs, val_loss, 'ro-', label='Валидационные потери')
plt.title('Потери на тренировке и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

# График точности
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'bo-', label='Тренировочная точность')
plt.plot(epochs, val_accuracy, 'ro-', label='Валидационная точность')
plt.title('Точность на тренировке и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

# Показываем графики
plt.tight_layout()
plt.show()

