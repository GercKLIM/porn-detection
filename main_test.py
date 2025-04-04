# ========================
# == ИМПОРТ БИБЛИОТЕК ====
# ========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.metrics import Precision, Recall, AUC, F1Score
from tensorflow.keras.models import load_model




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
max_tokens = 20000           # Максимальное количество уникальных слов
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


# =======================
# === ЗАГРУЗКА МОДЕЛИ ===
# =======================

# Загрузка модели
model = load_model('model_5.keras')


# ==============================
# === ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЯ ===
# ==============================


# Предсказание вероятностей
y_pred_proba = model.predict(X_train, batch_size=32, verbose=1)

# Преобразуем вероятности в метки (0 или 1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Преобразуем y_train и y_pred в двумерный формат
y_train_reshaped = tf.expand_dims(y_train, axis=-1)  # Добавляем новую ось
y_pred_reshaped = tf.expand_dims(y_pred, axis=-1)    # Добавляем новую ось

# Проверяем размерности
print("Форма y_train_reshaped:", y_train_reshaped.shape)
print("Форма y_pred_reshaped:", y_pred_reshaped.shape)

precision_metric = Precision()
recall_metric = Recall()
f1_metric = F1Score()

# Обновляем метрики
precision_metric.update_state(y_train_reshaped, y_pred_reshaped)
recall_metric.update_state(y_train_reshaped, y_pred_reshaped)
f1_metric.update_state(y_train_reshaped, y_pred_reshaped)

# Выводим результаты
print(f"Precision (точность): {precision_metric.result().numpy():.4f}")
print(f"Recall (полнота): {recall_metric.result().numpy():.4f}")
f1_score = f1_metric.result().numpy().item()  # Преобразуем в скалярное значение
print(f"F1-Score: {f1_score:.4f}")