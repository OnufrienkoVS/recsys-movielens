# MovieLens Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-✓-blue.svg)](https://www.docker.com/)
[![API](https://img.shields.io/badge/API-Documentation-brightgreen.svg)](http://localhost:8000/docs)

Система рекомендаций фильмов на основе коллаборативной фильтрации с использованием классических и графовых подходов. 

Проект включает обучение моделей в Jupyter Notebooks и REST API для получения рекомендаций.

**Используемый датасет:** [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) (версия ml-latest-small, 100.000 оценок)

## Модели  

| Модель         | Описание                                      | RMSE       | MAE         | Recall@50  | NDCG@k     |
|----------------|-----------------------------------------------|------------|-------------|------------|------------|
| **Popularity** | Бейзлайн (самые популярные фильмы)            | 0.9918     | 0.7627      | 0.0175     | 0.0095     |
| **KNN**        | Коллаборативная фильтрация на основе схожести | 0.9539     | 0.7267      | 0.0287     | 0.0217     |
| **SVD**        | Матричная факторизация (Funk SVD)             | **0.9022** | **0.6922**  | 0.0518     | 0.0470     |
| **LightGCN**   | Графовая нейронная сеть                       | 1.6717*    | 1.2097      | **0.1465** | **0.0719** |

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/OnufrienkoVS/recsys-movielens.git
cd recsys-movielens
```

### 2. Установка зависимостей

```bash
# Для API (оптимизированный набор)
pip install -r requirements-api.txt

# Для полного проекта
pip install -r requirements.txt
```

### 3. Запуск API

```bash
python -m app.main
```

### 4. Документация API

После запуска сервера документация доступна по адресам:

- Swagger UI: http://localhost:8000/docs

- ReDoc: http://localhost:8000/redoc

## Структура проекта

```
recsys-movielens/
├── app/                        # FastAPI приложение
│
├── models/                     # Сохраненные модели
│   └── svd_model.pkl           # SVD модель (включена в репозиторий)
│
├── notebooks/                  # Jupyter ноутбуки с обучением
│
├── tests/                      # Тесты API
│   └── test_api.py
│
├── src/                        # Вспомогательные скрипты
│   └── data_loader.py
│
├── data/                       # Датасет MovieLens
│   └── ml-latest-small/
│
├── requirements.txt            # Полные зависимости
├── requirements-api.txt        # Только для API
├── Dockerfile                  # Контейнеризация
├── docker-compose.yml
├── .gitignore
└── README.md
```

## API Эндпоинты

### GET `/health`

Проверка работоспособности сервиса.

### POST `/predict`

Предсказание оценки для пары (пользователь, фильм).

### POST `/recommend`

Получение топ-N рекомендаций для пользователя.

### GET `/models`

Список доступных моделей и текущая загруженная модель.

### POST `/models/switch`

Переключение на другую модель.

### GET `/models/current`

Информация о текущей загруженной модели.

## Тестирование

```bash
python tests/test_api.py
```

Ожидаемый результат:

```
✅ health: УСПЕШНО
✅ root: УСПЕШНО
✅ models: УСПЕШНО
✅ predict: УСПЕШНО
✅ predict_unknown: УСПЕШНО
✅ recommend: УСПЕШНО
✅ recommend_large: УСПЕШНО
✅ recommend_unknown: УСПЕШНО
✅ switch_model: УСПЕШНО
✅ switch_invalid: УСПЕШНО
✅ current_model: УСПЕШНО

Итого: 11 пройдено, 0 провалено

✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!
```

## Docker

``` bash
# Сборка и запуск
docker-compose up -d

# Остановка
docker-compose down
```

