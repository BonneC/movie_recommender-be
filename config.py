# Scheme: "postgres+psycopg2://<USERNAME>:<PASSWORD>@<IP_ADDRESS>:<PORT>/<DATABASE_NAME>"
#from dotenv import load_dotenv
import os

from pathlib import Path  # python3 only
#
# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=env_path)
#load_dotenv()

# DATABASE_URI = os.getenv('DATABASE_URI')


DATABASE_URI = 'postgres+psycopg2://postgres:fuk@localhost:5432/movie_base'
SECRET_KEY = "ec253568b58e9a1079e194181b90c11d7760e54e654e515785a13f4b5be55ff9"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30