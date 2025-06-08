from pydantic import BaseModel, EmailStr

from typing import Optional

from pydantic import BaseModel, Field
from typing import Union, Literal

class UserRegisterBase(BaseModel):
    user_id: str
    username: str
    password: str
    mode: Literal["user", "hotel"]

# Extra data required if mode == "user"
class UserModeData(BaseModel):
    location_user: str
    num_cities: int
    num_reviews_profile: int
    num_helpful_votes_user: int

# Extra data required if mode == "hotel"
class HotelModeData(BaseModel):
    offering_id: int
    service: int
    cleanliness: int
    overall: int
    value: int
    location_pref_score: int
    sleep_quality: int
    rooms: int
    hotel_class: int
    location_region: str

# Combined wrapper model
class UserRegister(UserRegisterBase):
    user_mode_data: Optional[UserModeData] = None
    hotel_mode_data: Optional[HotelModeData] = None


    
class LoginRequest(BaseModel):
    user_id: str
    password: str

class Recommendation(BaseModel):
    hotel_id: str
    score: float