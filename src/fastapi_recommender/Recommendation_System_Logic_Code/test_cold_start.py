from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_cold_start_def import cold_start_recommendation_combined

# Test 1: Amb diversos camps
print("▶ Test 1: Amb location + helpful + reviews")
result_1 = cold_start_recommendation_combined(
    user_id="test_user_001",
    mode="user",
    helpful=12,
    reviews=5,
    location="New York City"
)
print(result_1)

# Test 2: Només amb location + cities
print("\n▶ Test 2: Amb location + cities")
result_2 = cold_start_recommendation_combined(
    user_id="test_user_002",
    mode="user",
    cities=2,
    location="Houston"
)
print(result_2)

# Test 3: Amb location desconeguda
print("\n▶ Test 3: Location desconeguda")
result_3 = cold_start_recommendation_combined(
    user_id="test_user_003",
    mode="user",
    reviews=10,
    location="NoExisteixCiutat"
)
print(result_3)

# Test 4: Mode hotel (opcional)
print("\n▶ Test 4: Mode hotel")
result_4 = cold_start_recommendation_combined(
    user_id="test_user_004",
    mode="hotel",
    service=4.0,
    cleanliness=4.2,
    overall=4.5,
    value=3.8,
    location_pref_score=4.0,
    sleep_quality=4.1,
    rooms=4.0,
    hotel_class=4.0,
    location_region="California"
)
print(result_4)
