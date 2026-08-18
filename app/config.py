import os
from pathlib import Path
from typing import Optional

class Settings:
    
    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "ml-latest-small"
    MODELS_DIR = BASE_DIR / "models"
    
    # Настройки API
    API_TITLE = "MovieLens Recommendation API"
    API_DESCRIPTION = "API для рекомендаций фильмов на основе коллаборативной фильтрации"
    API_VERSION = "1.0.0"
    
    # Модель по умолчанию
    DEFAULT_MODEL = "svd"
    
    # Доступные модели (порядок = приоритет)
    AVAILABLE_MODELS = ["svd_model", "knn_model", "lightgcn"]
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Optional[Path]:
        """Возвращает путь к модели по имени"""
        model_file = cls.MODELS_DIR / f"{model_name}.pkl"
        if model_file.exists():
            return model_file
        
        # Проверяем .pt для LightGCN
        model_file = cls.MODELS_DIR / f"{model_name}.pt"
        if model_file.exists():
            return model_file
        
        return None
    
    @classmethod
    def get_available_models(cls) -> list:
        """Возвращает список доступных моделей в папке"""
        models = []
        for file in cls.MODELS_DIR.glob("*.pkl"):
            models.append(file.stem)
        for file in cls.MODELS_DIR.glob("*.pt"):
            models.append(file.stem)
        return models

settings = Settings()

if __name__ == '__main__':
    print(f"BASE_DIR: {settings.BASE_DIR}")
    print(f"MODELS_DIR: {settings.MODELS_DIR}")
    print(f"DATA_DIR: {settings.DATA_DIR}")
    print(f"Available models: {settings.get_available_models()}")
    for model in settings.get_available_models():
        print(f"Путь к {model}: {settings.get_model_path(model)}")