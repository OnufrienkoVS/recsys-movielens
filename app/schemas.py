from pydantic import BaseModel
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Запрос на предсказание оценки"""
    user_id: int
    movie_id: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "movie_id": 1
            }
        }

class RecommendationRequest(BaseModel):
    """Запрос на рекомендации"""
    user_id: int
    n: Optional[int] = 10
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "n": 10
            }
        }

class ModelSwitchRequest(BaseModel):
    """Запрос на переключение модели"""
    model_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "svd_model"
            }
        }


class PredictionResponse(BaseModel):
    """Ответ с предсказанной оценкой"""
    user_id: int
    movie_id: int
    predicted_rating: float

class MovieRecommendation(BaseModel):
    """Один рекомендованный фильм"""
    movie_id: int
    title: str
    predicted_rating: float

class RecommendationResponse(BaseModel):
    """Ответ с рекомендациями"""
    user_id: int
    recommendations: List[MovieRecommendation]

class ModelInfo(BaseModel):
    """Информация о модели"""
    model_id: str
    name: str
    description: Optional[str] = None
    is_loaded: bool = False

class ModelsResponse(BaseModel):
    """Список доступных моделей"""
    available_models: List[ModelInfo]
    current_model: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    movies_loaded: bool
    n_movies: Optional[int] = None
    n_ratings: Optional[int] = None
    model_type: Optional[str] = None
    n_users: Optional[int] = None

class ModelSwitchResponse(BaseModel):
    """Ответ на переключение модели"""
    status: str
    message: str
    previous_model: str
    current_model: str