from fastapi import APIRouter

from app.schemas import HealthResponse
from app.dependencies import get_recommender, load_movies_data
from app.recommender import MovieRecommender

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    recommender = get_recommender()
    movies_df = load_movies_data()

    return HealthResponse(
        status="healthy",
        model_loaded=recommender is not None and recommender.is_loaded(),
        movies_loaded=movies_df is not None,
        n_movies=len(movies_df) if movies_df is not None else None,
        n_ratings=None,
        model_type=recommender.model_type if recommender else None,
        n_users=len(recommender.user_embeddings) if (
            recommender and recommender.user_embeddings is not None
        ) else None
    )