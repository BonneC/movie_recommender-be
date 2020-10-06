from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, ForeignKey, Float, ARRAY
from fastapi_users import models
from sqlalchemy.orm import relationship

Base = declarative_base()


class Rating(Base):
    __tablename__ = 'ratings'
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    movie_id = Column(Integer, ForeignKey('movies.id'), primary_key=True)
    rating = Column(Float)
    timestamp = Column(Date)
    movie = relationship("Movie")

    def __init__(self, rating, timestamp):
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
    movies = relationship("Rating")

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
    # genres = Column(ARRAY(String))
    release_date = Column(Date)
    vote_average = Column(Float)
    vote_count = Column(Float)

    def __init__(self, imdb_id, title, release_date, vote_average, vote_count):
        self.imdb_id = imdb_id
        self.title = title
        # self.genres = genres
        self.release_date = release_date
        self.vote_average = vote_average
        self.vote_count = vote_count

    def __repr__(self):
        return "<Movie(id='{}', imdb_id='{}', title='{}')>" \
            .format(self.id, self.imdb_id, self.title)
