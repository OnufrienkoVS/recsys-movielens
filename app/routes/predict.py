from fastapi import APIRouter, Depends, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.dependencies import get_recommender_or_raise
from app.recommender import MovieRecommender

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("/", response_model=PredictionResponse)
async def predict_rating(
    request: PredictionRequest,
    recommender: MovieRecommender = Depends(get_recommender_or_raise)
):
    try:
        pred = recommender.predict(request.user_id, request.movie_id)
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=round(pred, 2)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")