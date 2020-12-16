from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, ForeignKey, Float, ARRAY
# from fastapi_users import models
from sqlalchemy.orm import relationship

Base = declarative_base()


class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    movie_id = Column(Integer)
    rating = Column(Float)
    timestamp = Column(Date)

    def __init__(self, user_id, movie_id, rating, timestamp):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating
        self.timestamp = timestamp

    def __repr__(self):
        return "<Rating(user_id='{}', movie_id='{}', rating='{}')>" \
            .format(self.user_id, self.movie_id, self.rating)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    password = Column(String)

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __repr__(self):
        return "<User(username='{}', password='{}')>" \
            .format(self.username, self.password)


class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True)
    imdb_id = Column(String)
    title = Column(String)
    release_date = Column(Date)
    vote_average = Column(Float)
    vote_count = Column(Float)

    def __init__(self, imdb_id, title, release_date, vote_average, vote_count):
        self.imdb_id = imdb_id
        self.title = title
        self.release_date = release_date
        self.vote_average = vote_average
        self.vote_count = vote_count

    def __repr__(self):
        return "<Movie(id='{}', imdb_id='{}', title='{}')>" \
            .format(self.id, self.imdb_id, self.title)


class Overview(Base):
    __tablename__ = 'overviews'
    id = Column(Integer, primary_key=True)
    overview = Column(String)
