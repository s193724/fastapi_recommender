from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.fastapi_recommender.models import SessionLocal, User
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from contextlib import asynccontextmanager
import json
import os
import csv
from scipy.sparse import load_npz
import uvicorn
import pandas as pd

# Importació de models i recomanacions
from src.fastapi_recommender.auth.user_model import UserRegister, LoginRequest, Recommendation
from src.fastapi_recommender.auth.auth_utils import create_access_token, get_current_user
from src.fastapi_recommender.models import UserModeProfile, HotelModeProfile, User as UserModel
from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_cold_start_def import (
    cold_start_recommendation_combined, apply_city_penalty, get_non_personalized_recommendations
)
from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_v3 import (
    hybrid_recommend
)
# from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_cold_start_def import (
#     cold_start_recommendation_combined,
#     apply_city_penalty
# )


#/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic&Code/recommender_v2.py
#app = FastAPI()

# Globals per les dades
user_item_matrix = None
user_similarity = None
hotel_similarity = None
user_id_to_idx = None
idx_to_user_id = None
hotel_id_to_idx = None
idx_to_hotel_id = None
hotel_meta_dict = None 
hotel_meta_df = None  

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""@app.on_event("startup")
def load_data():
    global user_item_matrix, user_similarity, hotel_similarity
    global user_id_to_idx, idx_to_user_id, hotel_id_to_idx, idx_to_hotel_id

    #base_dir = '/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'
    base_dir = '/Users/oliviapc/Documents/GitHub/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'
    
    #CSV_PATH = "/Users/filiporlikowski/Documents/fastapi_recommender/generated_passwords.csv"
    CSV_PATH = "/Users/oliviapc/Documents/GitHub/fastapi_recommender/generated_passwords.csv"
    
    print("Loading matrices...")
    user_item_matrix = load_npz(f'{base_dir}/user_hotel_matrix.npz')
    user_similarity = load_npz(f'{base_dir}/user_similarity_collab.npz')
    hotel_similarity = load_npz(f'{base_dir}/hotel_similarity_matrix.npz')
    hotel_features_sparse = load_npz(f'{base_dir}/hotel_features.npz')
    
    print("Loading mappings...")
    with open(f'{base_dir}/user_id_to_idx.json') as f:
        user_id_to_idx = json.load(f)
    with open(f'{base_dir}/idx_to_user_id.json') as f:
        idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
    with open(f'{base_dir}/hotel_id_to_idx.json') as f:
        hotel_id_to_idx = json.load(f)
    with open(f'{base_dir}/hotel_idx_to_id.json') as f:
        idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}
    
    print("All data loaded.")
    """
@asynccontextmanager
async def lifespan(app: FastAPI):
    global user_item_matrix, user_similarity, hotel_similarity
    global user_id_to_idx, idx_to_user_id, hotel_id_to_idx, idx_to_hotel_id
    global hotel_meta_df, hotel_meta_dict

    base_dir = '/Users/oliviapc/Documents/GitHub/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'

    print("Loading matrices...")
    user_item_matrix = load_npz(f'{base_dir}/user_hotel_matrix.npz')
    user_similarity = load_npz(f'{base_dir}/user_similarity_collab.npz')
    hotel_similarity = load_npz(f'{base_dir}/hotel_similarity_matrix.npz')

    print("Loading mappings...")
    with open(f'{base_dir}/user_id_to_idx.json') as f:
        user_id_to_idx = json.load(f)
    with open(f'{base_dir}/idx_to_user_id.json') as f:
        idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
    with open(f'{base_dir}/hotel_id_to_idx.json') as f:
        hotel_id_to_idx = json.load(f)
    with open(f'{base_dir}/hotel_idx_to_id.json') as f:
        idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

    
    hotel_meta_df = pd.read_csv(f'{base_dir}/hotel_df.csv')
    hotel_meta_dict = hotel_meta_df.set_index("offering_id").to_dict(orient="index")
    hotel_meta_df.set_index("offering_id", inplace=True)
    
    print(hotel_meta_df.columns)


    print("All data loaded.")
    yield

app = FastAPI(lifespan=lifespan)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#GENERATED_PASSWORDS_PATH = "/Users/filiporlikowski/Documents/fastapi_recommender/generated_passwords.csv"
GENERATED_PASSWORDS_PATH = "/Users/oliviapc/Documents/GitHub/fastapi_recommender/generated_passwords.csv"

#from models import User as UserModel, UserModeProfile, HotelModeProfile
#from fastapi import HTTPException, Depends
import os, csv

@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user. Follows a step-by-step flow:
    1. Provide basic info: user_id, username, password, mode
    2. Depending on mode, provide `user_mode_data` or `hotel_mode_data`
    """
    if not os.path.exists(GENERATED_PASSWORDS_PATH):
        with open(GENERATED_PASSWORDS_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "username", "password"])

    # Check if user already exists
    existing_user = db.query(UserModel).filter(UserModel.user_id == user.user_id).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already registered")

    # Hash password
    hashed_password = pwd_context.hash(user.password)

    # Create base user
    new_user = UserModel(
        user_id=user.user_id,
        username=user.username,
        hashed_password=hashed_password,
        mode = user.mode,
    )

    db.add(new_user)

    # Create corresponding profile depending on mode
    if user.mode == "user":
        if not user.user_mode_data:
            raise HTTPException(status_code=400, detail="Missing user_mode_data for 'user' mode")

        user_profile = UserModeProfile(
            user_id=user.user_id,
            location_user=user.user_mode_data.location_user,
            num_cities=user.user_mode_data.num_cities,
            num_reviews_profile=user.user_mode_data.num_reviews_profile,
            num_helpful_votes_user=user.user_mode_data.num_helpful_votes_user,
        )
        db.add(user_profile)

    elif user.mode == "hotel":
        if not user.hotel_mode_data:
            raise HTTPException(status_code=400, detail="Missing hotel_mode_data for 'hotel' mode")

        hotel_profile = HotelModeProfile(
            user_id=user.user_id,
            offering_id=user.hotel_mode_data.offering_id,  # Add this line
            service=user.hotel_mode_data.service,
            cleanliness=user.hotel_mode_data.cleanliness,
            overall=user.hotel_mode_data.overall,
            value=user.hotel_mode_data.value,
            location_pref_score=user.hotel_mode_data.location_pref_score,
            sleep_quality=user.hotel_mode_data.sleep_quality,
            rooms=user.hotel_mode_data.rooms,
            hotel_class=user.hotel_mode_data.hotel_class,
            location_region=user.hotel_mode_data.location_region,
)

        db.add(hotel_profile)

    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'user' or 'hotel'.")

    # Commit everything to DB
    db.commit()

    # Log password to CSV (useful only for dev/debugging — remove in prod)
    with open(GENERATED_PASSWORDS_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user.user_id, user.username, user.password])

    return {"message": "User registered successfully."}



# Dependency to get DB session

@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Wrong User ID")

    if not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Wrong Password")

    access_token = create_access_token(data={"sub": user.user_id})

    print(f"DEBUG: Logging in user {user.user_id}")
    print(f"DEBUG: Generated token: {access_token}")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "message": "Login successful"
    }

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def enrich_recommendations(recommendations, top_k=10):
    enriched = []

    # Traduïm els índexs interns a offering_id
    recommendations_with_ids = [
        (idx_to_hotel_id.get(idx, idx), score) for idx, score in recommendations[:top_k]
    ]

    for hotel_id, score in recommendations_with_ids:
        hotel_id_str = str(hotel_id)
        if hotel_id_str in hotel_meta_df.index:
            meta = hotel_meta_df.loc[hotel_id_str]
            enriched.append({
                "hotel_id": hotel_id_str,
                "score": round(score, 2),
                "hotel_name": meta.get("name", "N/A"),
                "hotel_class": meta.get("hotel_class", None),
                "location": meta.get("locality", "N/A"),
            })
        else:
            enriched.append({
                "hotel_id": hotel_id_str,
                "score": round(score, 2),
                "hotel_name": "sense metadades",
                "hotel_class": None,
                "location": None,
            })
    print("Raw recommendations (indices):", recommendations[:top_k])
    print("Mapped recommendations (offering_ids):", recommendations_with_ids)

    return enriched


# ---------------- USER ROUTES ---------------- #


# @app.get("/recommend/")
# def get_recommendations(user_id: str = Query(..., description="User ID to get recommendations for")):
#     try:
#         recommendations = hybrid_recommend(user_id, alpha=0.7, top_k=10)
#         result = [
#             {"hotel_id": hotel_id, "score": round(score, 4)}
#             for hotel_id, score in recommendations
#         ]
#         return {"user_id": user_id, "recommendations": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    


# ---------------- RECOMMENDATIONS ROUTE ---------------- #

# @app.get("/recommendations/")
# def get_recommendations(user_id: str = Query(...)):
#     try:
#         print(f"User ID received: {user_id}")

#         if user_id in user_id_to_idx:
#             print("Using hybrid recommendation...")
#             recommendations = hybrid_recommend(user_id, alpha=0.7, top_k=10)
#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail="User not found. Cold-start requires registration with profile metadata."
#             )

#         print("Applying city penalty...")
#         final_recommendations = apply_city_penalty(recommendations)

#         return {
#             "user_id": user_id,
#             "recommendations": [
#                 {"hotel_id": hotel_id, "score": round(score, 4)}
#                 for hotel_id, score in final_recommendations
#             ]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ───── Endpoint recomanacions personalitzades ──────────────────────────────
@app.get("/recommendations/")
def get_recommendations(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    print(f"DEBUG: In /recommendations/ endpoint, user_id={user_id}")
    try:
        if user_id in user_id_to_idx:
            recommendations = hybrid_recommend(user_id, alpha=0.7, top_k=10)
        else:
            user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
            if not user:
                raise HTTPException(status_code=400, detail="User not found. Please register first.")
            if user.mode == "user":
                profile = db.query(UserModeProfile).filter(UserModeProfile.user_id == user.user_id).first()
                if not profile:
                    raise HTTPException(status_code=400, detail="User profile not found.")
                recommendations = cold_start_recommendation_combined(
                    user_id=user.user_id,
                    mode="user",
                    location=profile.location_user,
                    cities=profile.num_cities,
                    reviews=profile.num_reviews_profile,
                    helpful=profile.num_helpful_votes_user,
                    top_k=10
                )
            elif user.mode == "hotel":
                profile = db.query(HotelModeProfile).filter(HotelModeProfile.user_id == user.user_id).first()
                if not profile:
                    raise HTTPException(status_code=400, detail="Hotel profile not found.")
                recommendations = cold_start_recommendation_combined(
                    user_id=user.user_id,
                    mode="hotel",
                    offering_id=profile.offering_id,
                    service=profile.service,
                    cleanliness=profile.cleanliness,
                    overall=profile.overall,
                    value=profile.value,
                    location_pref_score=profile.location_pref_score,
                    sleep_quality=profile.sleep_quality,
                    rooms=profile.rooms,
                    hotel_class=profile.hotel_class,
                    location_region=profile.location_region,
                    top_k=10
                )
            else:
                raise HTTPException(status_code=400, detail="User mode invalid.")

        final_recommendations = apply_city_penalty(recommendations)
        enriched = enrich_recommendations(final_recommendations)

        return {
            "user_id": user_id,
            "recommendations": enriched
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ───── Endpoint recomanacions no personalitzades ────────────────────────────
@app.get("/recommendations/non_personalized", response_model=list[Recommendation])
def non_personalized_recommendations(top_k: int = 10):
    recommendations = get_non_personalized_recommendations(top_k=top_k)
    adjusted_recommendations = apply_city_penalty(recommendations)
    return enrich_recommendations(adjusted_recommendations, top_k=top_k)



#mounting the static files directory

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# Mount the folder containing your frontend files
app.mount("/static", StaticFiles(directory="/Users/oliviapc/Documents/GitHub/fastapi_recommender/frontend"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/login_page")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("/favicon.ico")

# Route to serve the register page
@app.get("/register_page", response_class=FileResponse)
def serve_register_page():
    #return FileResponse("/Users/filiporlikowski/Documents/fastapi_recommender/frontend/register.html")
    return FileResponse("/Users/oliviapc/Documents/GitHub/fastapi_recommender/frontend/register.html")
# Optional: Route for login.html and recommendations.html if needed
@app.get("/login_page", response_class=FileResponse)
def serve_login_page():
    #return FileResponse("/Users/filiporlikowski/Documents/fastapi_recommender/frontend/login.html")
    return FileResponse("/Users/oliviapc/Documents/GitHub/fastapi_recommender/frontend/login.html")

@app.get("/recommendations_page", response_class=FileResponse)
def serve_recommendations_page():
    #return FileResponse("/Users/filiporlikowski/Documents/fastapi_recommender/frontend/recommendations.html")
    return FileResponse("/Users/oliviapc/Documents/GitHub/fastapi_recommender/frontend/recommendations.html")
