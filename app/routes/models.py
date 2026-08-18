from fastapi import APIRouter, Depends, HTTPException

from app.schemas import ModelsResponse, ModelInfo, ModelSwitchRequest, ModelSwitchResponse
from app.dependencies import get_recommender, get_recommender_or_raise
from app.recommender import MovieRecommender
from app.config import settings

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

@router.post("/switch", response_model=ModelSwitchResponse)
async def switch_model(
    request: ModelSwitchRequest,
    recommender: MovieRecommender = Depends(get_recommender_or_raise)
):
    """
    Переключает текущую модель
    """
    model_path = settings.get_model_path(request.model_id)
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{request.model_id}' не найдена"
        )

    try:
        previous_model = recommender.model_type
        recommender.switch_model(model_path, request.model_id)

        return ModelSwitchResponse(
            status="success",
            message=f"Модель переключена на {request.model_id}",
            previous_model=previous_model,
            current_model=request.model_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current", response_model=dict)
async def get_current_model(
    recommender: MovieRecommender = Depends(get_recommender)
):
    """Возвращает информацию о текущей модели"""
    if recommender is None or not recommender.is_loaded():
        return {
            "status": "no_model_loaded",
            "message": "Модель не загружена"
        }

    return {
        "status": "loaded",
        "model_type": recommender.model_type,
        "model_class": type(recommender.model).__name__ if recommender.model else None
    }