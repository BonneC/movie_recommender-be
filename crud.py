import datetime
from fastapi.encoders import jsonable_encoder
from starlette.responses import FileResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import datetime as dt
import json
from typing import Optional
from config import DATABASE_URI, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from db import Session, engine
import models
import schemas
import services.recommender_functions as rec
import services.predictors as pred
import pandas as pd

# engine = create_engine(DATABASE_URI)
# Session = sessionmaker(bind=engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(name):
    with session_scope() as s:
        user = s.query(models.User).filter_by(username=name).first()
        return schemas.UserInDB(**jsonable_encoder(user))


def get_movie(id):
    with session_scope() as s:
        movie = s.query(models.Movie).filter_by(id=id).first()
        return jsonable_encoder(movie)


# get a list of all movies that contain the given keyword/s
def search_movies(keyword):
    """

    :param keyword:
    :return:
    """
    with session_scope() as s:
        print(keyword)
        #look_for = '%{0}%'.format(keyword)
        look_for = f'%{keyword}%'
        movies = s.query(models.Movie).filter(models.Movie.title.ilike(look_for)).all()
        print(movies)
        return jsonable_encoder(movies[:5])


# get a list of all the movies that the user has rated
def get_ratings_for_user(user: schemas.User):
    with session_scope() as s:
        user = s.query(models.User).filter_by(username=user.username).first()
        # ids = [r.movie_id for r in s.query(models.Rating.movie_id).filter_by(user_id=user.id)]
        # movies = s.query(models.Movie).filter(models.Movie.id.in_(ids)).all()
        ratings = s.query(models.Rating).filter_by(user_id=user.id).all()
        tmp = pd.DataFrame.from_dict(jsonable_encoder(ratings))
        print(tmp)
        return jsonable_encoder(ratings)


def get_movie_ids(user: schemas.User):
    with session_scope() as s:
        user = s.query(models.User).filter_by(username=user.username).first()
        ids = [r.movie_id for r in s.query(models.Rating.movie_id).filter_by(user_id=user.id)]
        return ids


# TODO fix dis
def get_recommended_movies(user: schemas.User):
    ids = get_movie_ids(user)
    # ids = [r['movie_id'] for r in ratings]
    # print(ids)
    with session_scope() as s:
        titles = s.query(models.Movie.title).filter(models.Movie.id.in_(ids)).all()
        user_titles = [title[0] for title in titles]
        ratings = s.query(models.Rating.rating).filter(models.Rating.movie_id.in_(ids)).all()
        ratings = [rating[0] for rating in ratings]
    print(user_titles)
    # # print(ratings)
    movies = pd.DataFrame()
    movies['id'] = ids
    movies['user_titles'] = user_titles
    movies['rating'] = ratings
    movies = movies.astype({"rating": float})
    print(movies.dtypes)
    # print(movies.head())
    predictions = pred.results(movies)
    print(predictions.head())
    final = predictions.head()
    print(final)
    # keyword_movies = rec.keyword_recommender(user_titles)
    # print(keyword_movies['title'].head())
    #
    # input_movies = pd.DataFrame(ratings)
    # input_movies['title'] = user_titles
    # print(input_movies.head())
    return jsonable_encoder(final.reset_index().to_dict('records'))
    # genre_recs = rec.genre_recommender(input_movies, keyword_movies)

    # print(genre_recs['title'].tolist())
    # return jsonable_encoder(genre_recs.to_dict(orient="index"))


def update_rating(movie_id, rating, current_user: schemas.User):
    with session_scope() as s:
        user = s.query(models.User).filter_by(username=current_user.username).first()
        rat = s.query(models.Rating).filter_by(user_id=user.id, movie_id=movie_id).first()
        rat.rating = rating
        s.commit()
    return 201, {"status": "sukses"}


def add_rating(movie_id, rating, current_user: schemas.User):
    with session_scope() as s:
        user = s.query(models.User).filter_by(username=current_user.username).first()
        rating = models.Rating(user_id=user.id, movie_id=movie_id, rating=rating, timestamp=dt.date.today())
        s.add(rating)
        s.commit()
    return 201, {"status": "sukses"}


def delete_rating(movie_id, current_user: schemas.User):
    with session_scope() as s:
        user = s.query(models.User).filter_by(username=current_user.username).first()
        s.query(models.Rating).filter_by(user_id=user.id, movie_id=movie_id).delete()
        s.commit()
    return 201, {"status": "sukses"}


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(name=token_data.username)
    if user is None:
        raise credentials_exception
    return user


def get_current_active_user(current_user: schemas.User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def login_for_access_token(form_data):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return access_token
