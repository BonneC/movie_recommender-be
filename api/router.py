from fastapi import APIRouter, File, UploadFile, Depends
from datetime import datetime as dt
import crud
from typing import List
from schemas import User, Token, TokenData, Rating, RatingList
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()


@router.get("/")
async def read_root():
    return crud.get_user('jas')


@router.get("/movie/{item_id}")
async def show_movie(item_id: int):
    return crud.get_movie(item_id)


@router.get("/movies/")
async def get_movies(current_user: User = Depends(crud.get_current_user)):
    return crud.get_ratings_for_user(current_user)


@router.get("/movies/recommended")
async def get_recommendations(user_ratings: List[Rating] = Depends(get_movies)):
    return crud.get_recommended_movies(user_ratings)


@router.put("/movies/{movie_id}")
async def update_rating(movie_id: int, rating: float, current_user: User = Depends(crud.get_current_user)):
    return crud.update_rating(movie_id, rating, current_user)


@router.post("/movies/{movie_id}")
async def add_rating(movie_id: int, rating: float, current_user: User = Depends(crud.get_current_user)):
    return crud.add_rating(movie_id, rating, current_user)


@router.delete("/movies/{movie_id}")
async def delete_rating(movie_id: int, current_user: User = Depends(crud.get_current_user)):
    return crud.delete_rating(movie_id, current_user)


@router.post("/token", response_model=Token)
async def create_token(form_data: OAuth2PasswordRequestForm = Depends()):
    access_token = crud.login_for_access_token(form_data)
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=Token)
async def create_token(form_data: OAuth2PasswordRequestForm = Depends()):
    access_token = crud.login_for_access_token(form_data)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me")
async def read_users_me(current_user: User = Depends(crud.get_current_user)):
    return current_user
