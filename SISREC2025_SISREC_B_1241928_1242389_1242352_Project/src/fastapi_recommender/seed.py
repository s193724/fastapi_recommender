from models import SessionLocal, Base, engine
from generate_passwrpds import create_users_from_csv

Base.metadata.create_all(bind=engine)

db = SessionLocal()

create_users_from_csv("/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code/USER_DF.csv", db)
db.close()