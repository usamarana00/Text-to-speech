from sqlalchemy import create_engine, text
import os

engine = create_engine("sqlite:///sample.db",echo=True)


with engine.connect() as connection:
     result= connection.execute(text('select "Hello World"'))

     print(result.all())
