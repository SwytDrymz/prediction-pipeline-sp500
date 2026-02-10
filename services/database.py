import os
import asyncio
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def create_table():
    engine = create_engine(re.sub(r'^postgresql:', 'postgresql+psycopg:', os.getenv('DATABASE_URL')))
    with engine.connect() as conn:
        sql_create_table = text(""" 
CREATE TABLE IF NOT EXISTS :table
  """)
