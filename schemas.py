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
    # genres = Column(ARRAY(String))
    release_date: datetime.date
    vote_average: float
    vote_count: float

    class Config:
        orm_mode = True


class Rating(BaseModel):
    rating = float
    user_id = int
    timestamp = datetime.date
    movie_id = int

    class Config:
        orm_mode = True


class RatingList(BaseModel):
    ratings: List[Rating]
