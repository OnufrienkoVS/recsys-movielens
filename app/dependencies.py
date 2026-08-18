import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import HTTPException

from app.config import settings
from app.recommender import MovieRecommender


# Глобальные переменные
_recommender: Optional[MovieRecommender] = None
_movies_df: Optional[pd.DataFrame] = None


def load_movies_data() -> pd.DataFrame:
    """
    Загружает данные о фильмах из CSV файла
    """
    global _movies_df
    
    if _movies_df is None:
        movies_path = settings.DATA_DIR / "movies.csv"
        _movies_df = pd.read_csv(movies_path)
        print(f"✅ Загружено {len(_movies_df)} фильмов из {movies_path}")
    
    return _movies_df


def get_recommender() -> Optional[MovieRecommender]:
    global _recommender
    
    if _recommender is not None and _recommender.is_loaded():
        return _recommender
    
    # Загружаем данные о фильмах
    movies_df = load_movies_data()
    
    # Ищем модели в папке models/
    models_dir = settings.MODELS_DIR
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pt"))
    
    if not model_files:
        print("⚠️ Модели не найдены в папке models/")
        return None
    
    # Выбираем модель по приоритету
    chosen_file = None
    chosen_type = None
    
    for model_type in settings.AVAILABLE_MODELS:
        for file in model_files:
            if model_type in file.stem:
                chosen_file = file
                chosen_type = model_type
                break
        if chosen_file:
            break
    
    try:
        _recommender = MovieRecommender(chosen_file, chosen_type)
        _recommender.load().set_data(movies_df)
        
        print(f"✅ Рекомендательная система готова! (модель: {chosen_type})")
        return _recommender
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None


def get_recommender_or_raise():
    """Возвращает recommender"""
    recommender = get_recommender()
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Рекомендательная система недоступна. Модель не загружена."
        )
    return recommender