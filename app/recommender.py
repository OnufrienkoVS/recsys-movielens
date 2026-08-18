import pickle
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, Dataset, Reader

from app.config import settings


class PopularityRecommender:
    """Рекомендует самые популярные фильмы"""
    
    def __init__(self):
        self.popularity = None
        
    def fit(self, ratings):
        self.popularity = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False)
        self.mean_ratings = ratings.groupby('movieId')['rating'].mean()
        
    def recommend(self, n=10):
        return self.popularity.head(n).index.tolist()
    
    def predict(self, user_id, movie_id):
        # Предсказывает оценку (средняя оценка фильма)
        return self.mean_ratings.get(movie_id, 3.5)

class MovieRecommender:
    """
    Универсальный класс для работы с сохраненными моделями
    """
    
    def __init__(self, model_path: Optional[Path] = None, model_type: str = "unknown"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.movies_df = None
        self._loaded = False
        
        # Для LightGCN
        self.user_embeddings = None
        self.item_embeddings = None
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_id2idx = {}
        self.user_idx2id = {}
        self.item_id2idx = {}
        self.item_idx2id = {}
        
    def load(self) -> "MovieRecommender":
        """Загружает модель из файла"""
        if self.model_path is None:
            raise ValueError("Путь к модели не указан")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        try:
            # Определяем тип модели по расширению
            if self.model_path.suffix == '.pt':
                self._load_lightgcn()
            else:
                self._load_surprise_model()
            
            self._loaded = True
            print(f"✅ Модель {self.model_type} загружена из {self.model_path}")
            return self
            
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")
    
    def _load_surprise_model(self):
        """Загружает Surprise модель (SVD, KNN)"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Поддерживаем разные форматы сохранения
        if isinstance(data, dict):
            self.model = data.get('model')
            self.metrics = data.get('metrics', {})
        else:
            self.model = data
        
        # Проверяем, что модель имеет метод predict
        if not hasattr(self.model, 'predict'):
            raise ValueError("Загруженный объект не является моделью Surprise")
    
    def _load_lightgcn(self):
        """Загружает LightGCN модель из .pt файла"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.metrics = checkpoint.get('metrics', {})
        
        state_dict = checkpoint.get('model_state_dict')
        if state_dict is None:
            raise ValueError("В .pt файле нет 'model_state_dict'")
        
        required_keys = ['user_embedding.weight', 'item_embedding.weight']
        for key in required_keys:
            if key not in state_dict:
                raise ValueError(f"В state_dict нет ключа '{key}'")
        
        # Извлекаем эмбеддинги
        self.user_embeddings = state_dict['user_embedding.weight'].cpu().numpy()
        self.item_embeddings = state_dict['item_embedding.weight'].cpu().numpy()
        
        # Загружаем маппинг, если он есть
        if 'user_id2idx' in checkpoint:
            self.user_id2idx = checkpoint['user_id2idx']
            self.user_idx2id = checkpoint['user_idx2id']
            self.item_id2idx = checkpoint['item_id2idx']
            self.item_idx2id = checkpoint['item_idx2id']
    
    def _get_lightgcn_indices(self, user_id: int, movie_id: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Преобразует оригинальные ID в индексы для LightGCN
        """
        if not self.user_id2idx:
            if user_id < len(self.user_embeddings) and movie_id < len(self.item_embeddings):
                return user_id, movie_id
            return None, None
        
        # Используем маппинг
        user_idx = self.user_id2idx.get(user_id)
        item_idx = self.item_id2idx.get(movie_id)
        
        if user_idx is None or item_idx is None:
            return None, None
        
        return user_idx, item_idx
    
    def is_loaded(self) -> bool:
        """Проверяет, загружена ли модель"""
        if self.model is not None:
            return True
        if self.user_embeddings is not None and self.item_embeddings is not None:
            return True
        return False
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Предсказывает оценку"""
        if not self.is_loaded():
            raise ValueError("Модель не загружена")
        
        try:
            # Surprise модели (работают с оригинальными ID)
            if self.model is not None and hasattr(self.model, 'predict'):
                return self.model.predict(user_id, movie_id).est
            
            # LightGCN (нужна переиндексация)
            elif self.user_embeddings is not None and self.item_embeddings is not None:
                user_idx, item_idx = self._get_lightgcn_indices(user_id, movie_id)
                
                if user_idx is None or item_idx is None:
                    return 3.5
                
                pred = np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])
                # Масштабируем в диапазон 0.5-5.0 через sigmoid
                pred = 1 / (1 + np.exp(-pred))  # sigmoid
                pred = pred * 4.5 + 0.5
                return float(np.clip(pred, 0.5, 5.0))
            
            return 3.5
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return 3.5
    
    def recommend(self, user_id: int, n: int = 10) -> List[Dict]:
        """Рекомендует топ-N фильмов"""
        if not self.is_loaded():
            raise ValueError("Модель не загружена")
        
        if self.movies_df is None:
            raise ValueError("Данные о фильмах не загружены")
        
        # Получаем все фильмы
        all_movies = self.movies_df['movieId'].values
        
        # Проверяем валидность пользователя для LightGCN
        if self.model_type == "lightgcn" and self.user_embeddings is not None:
            if self.user_id2idx:
                if user_id not in self.user_id2idx:
                    return self._get_popular_movies(n)
            else:
                # Если маппинга нет, пробуем использовать ID как индекс
                if user_id >= len(self.user_embeddings):
                    return self._get_popular_movies(n)
        
        # Предсказываем оценки для всех фильмов
        predictions = []
        for movie_id in all_movies:
            try:
                pred = self.predict(user_id, movie_id)
                predictions.append((movie_id, pred))
            except Exception:
                continue
        
        # Сортируем по убыванию
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:n]
        
        # Добавляем названия
        results = []
        for movie_id, score in top_movies:
            title_row = self.movies_df[self.movies_df['movieId'] == movie_id]['title']
            title = title_row.values[0] if len(title_row) > 0 else f"Movie {movie_id}"
            results.append({
                'movie_id': int(movie_id),
                'title': title,
                'predicted_rating': round(score, 2)
            })
        
        return results
    
    def _get_popular_movies(self, n: int) -> List[Dict]:
        """Возвращает популярные фильмы"""
        top_movies = self.movies_df.head(n)
        return [
            {
                'movie_id': int(row['movieId']),
                'title': row['title'],
                'predicted_rating': 4.0
            }
            for _, row in top_movies.iterrows()
        ]
    
    def set_data(self, movies_df: pd.DataFrame) -> "MovieRecommender":
        """Устанавливает данные о фильмах"""
        self.movies_df = movies_df
        return self
    
    def set_mapping(self, user_id2idx: dict, user_idx2id: dict, 
                    item_id2idx: dict, item_idx2id: dict) -> "MovieRecommender":
        """Устанавливает маппинг для LightGCN"""
        self.user_id2idx = user_id2idx
        self.user_idx2id = user_idx2id
        self.item_id2idx = item_id2idx
        self.item_idx2id = item_idx2id
        return self

    @property
    def model_name(self) -> str:
        """Возвращает имя модели"""
        return self.model_type