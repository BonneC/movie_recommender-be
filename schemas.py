from pydantic import BaseModel
from typing import Optional
import datetime
from typing import List


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

    class Config:
        orm_mode = True


class UserInDB(User):
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class Movie(BaseModel):
    id: int
    imdb_id: str
    title: str
    release_date: datetime.date
    vote_average: float
    vote_count: float

    class Config:
        orm_mode = True


class Rating(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime.date


class RatingList(BaseModel):
    ratings: List[Rating]
