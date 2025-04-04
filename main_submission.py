# ========================
# == ИМПОРТ БИБЛИОТЕК ====
# ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model  # Импортируем load_model

# =============================================
# === ЗАГРУЗКА ДАННЫХ ТЕСТОВОГО ДАТАСЕТА ===
# =============================================

print("ИМПОРТ ДАТАСЕТА")

# Путь к файлу тестового датасета
test_ds_path = "datasets/test.csv"

# Получение тестового датасета
test_ds = pd.read_csv(test_ds_path, encoding='utf-8')

# Замена NaN на пустые строки
test_ds = test_ds.fillna('')

# ========================
# === ОБРАБОТКА ДАННЫХ ===
# ========================

# Параметры векторизации (должны совпадать с параметрами при обучении)
max_tokens = 20000  # Максимальное количество уникальных слов
output_sequence_length = 50  # Длина последовательности после векторизации

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

# Путь к тренировочному датасету
train_ds_path = "datasets/train.csv"
train_ds = pd.read_csv(train_ds_path, encoding='utf-8').fillna('')

# Адаптируем векторизаторы на данных
url_vectorizer.adapt(train_ds['url'])
title_vectorizer.adapt(train_ds['title'])


# Функция для преобразования данных
def preprocess_data(url, title):
    url_vec = url_vectorizer(url)
    title_vec = title_vectorizer(title)
    return tf.concat([url_vec, title_vec], axis=1)


# Применяем препроцессинг к тестовым данным
X_test = preprocess_data(test_ds['url'].values.astype(str), test_ds['title'].values.astype(str))

# Явное преобразование в тензор (если необходимо)
if not isinstance(X_test, tf.Tensor):
    X_test = tf.convert_to_tensor(X_test, dtype=tf.int64)

# =======================
# === ЗАГРУЗКА МОДЕЛИ ===
# =======================

print("ЗАГРУЗКА МОДЕЛИ")

# Загрузка модели
model = load_model('model_6.keras')
print("Модель успешно загружена.")

# ==============================
# === ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЯ ===
# ==============================

print("ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ")

# Предсказание вероятностей на тестовой выборке
y_pred_proba_test = model.predict(X_test, batch_size=32, verbose=1)

# Преобразуем вероятности в метки (0 или 1)
y_pred_test = (y_pred_proba_test > 0.5).astype(int).flatten()

# Создаем DataFrame для submission
submission = pd.DataFrame({
    'ID': test_ds['ID'],  # Идентификаторы записей из тестового набора
    'label': y_pred_test  # Предсказанные метки
})

# Сохраняем результаты в файл
submission.to_csv('submission_6.csv', index=False)

print("Файл submission сохранен!")