from config import DATABASE_URI, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
