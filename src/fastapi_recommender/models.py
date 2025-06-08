from sqlalchemy import Column, Integer, String, Float, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

database_url = "sqlite:///./test.db"
engine = create_engine(database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    username = Column(String)
    hashed_password = Column(String)
    mode = Column(String)  # "user" or "hotel"

class UserModeProfile(Base):
    __tablename__ = "user_mode_profiles"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    location_user = Column(String)
    num_cities = Column(Float)
    num_reviews_profile = Column(Float)
    num_helpful_votes_user = Column(Float)
    
class HotelModeProfile(Base):
    __tablename__ = "hotel_mode_profiles"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    service = Column(Float)
    cleanliness = Column(Float)
    overall = Column(Float)
    value = Column(Float)
    location_pref_score = Column(Float)
    sleep_quality = Column(Float)
    rooms = Column(Float)
    hotel_class = Column(Float)
    location_region = Column(String)

class RecommendationDB(Base):
    __tablename__ = "recommendations"
    hotel_id = Column(String, primary_key=True)
    score = Column(Float)

# Base.metadata.drop_all(bind=engine)

# # Create all tables fresh
# Base.metadata.create_all(bind=engine)

# print("Dropped existing tables and created new ones successfully.")


