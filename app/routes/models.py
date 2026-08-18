from fastapi import APIRouter, Depends

from app.schemas import ModelsResponse, ModelInfo
from app.dependencies import get_recommender
from app.recommender import MovieRecommender

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/", response_model=ModelsResponse)
async def list_models(
    recommender: MovieRecommender = Depends(get_recommender)
):
    model_definitions = {
        "svd_model": {
            "name": "SVD",
            "description": "Матричная факторизация (funk)."
        },
        "knn_model": {
            "name": "KNN",
            "description": "Коллаборативная фильтрация на основе схожести пользователей (фильмов)."
        },
        "lightgcn": {
            "name": "LightGCN",
            "description": "Графовая нейронная сеть для коллаборативной фильтрации."
        }
    }

    available_models = []
    current_model = None

    for model_id, info in model_definitions.items():
        is_loaded = (
            recommender is not None and
            recommender.is_loaded() and
            recommender.model_type == model_id
        )

        available_models.append(ModelInfo(
            model_id=model_id,
            name=info["name"],
            description=info["description"],
            is_loaded=is_loaded
        ))

        if is_loaded:
            current_model = model_id

    if recommender and recommender.is_loaded() and current_model is None:
        current_model = recommender.model_type
        available_models.append(ModelInfo(
            model_id=recommender.model_type,
            name=recommender.model_type,
            description="Загруженная модель",
            is_loaded=True
        ))

    return ModelsResponse(
        available_models=available_models,
        current_model=current_model
    )