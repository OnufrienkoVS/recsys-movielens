from fastapi import APIRouter, Depends, HTTPException

from app.schemas import RecommendationRequest, RecommendationResponse, MovieRecommendation
from app.dependencies import get_recommender_or_raise
from app.recommender import MovieRecommender

router = APIRouter(prefix="/recommend", tags=["recommendations"])

@router.post("/", response_model=RecommendationResponse)
async def recommend_movies(
    request: RecommendationRequest,
    recommender: MovieRecommender = Depends(get_recommender_or_raise)
):
    try:
        recommendations = recommender.recommend(request.user_id, request.n)

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[MovieRecommendation(**rec) for rec in recommendations]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения рекомендаций: {str(e)}")