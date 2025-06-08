import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz, hstack, csr_matrix, vstack
import joblib
import json
import pandas as pd
from collections import defaultdict

base_dir = '/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'

# --- Load persistent data --- /src/fastapi_recommender/Recommendation_System_Logic_Code/
user_features_sparse = load_npz(f'{base_dir}/user_features_sparse.npz')
user_item_matrix = load_npz(f'{base_dir}/user_hotel_matrix.npz')
user_features_collab = load_npz(f'{base_dir}/user_features_collab.npz')
hotel_meta_df = pd.read_csv(f'{base_dir}/hotel_df.csv')
hotel_meta_df.set_index('offering_id', inplace=True)
hotel_features_sparse = load_npz(f'{base_dir}/hotel_features.npz')

with open(f'{base_dir}/user_id_to_idx.json') as f:
    user_id_to_idx = json.load(f)
with open(f'{base_dir}/idx_to_user_id.json') as f:
    idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
with open(f'{base_dir}/hotel_idx_to_id.json') as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

scaler = joblib.load(f'{base_dir}/scaler.pkl')
vectorizer_location = joblib.load(f'{base_dir}/vectorizer_location.pkl')


def cold_start_recommendation(user_id, top_k=10):
    global user_features_sparse
    global user_item_matrix
    print("Cold start: please answer a few questions.")
    try:
        # Step 1: Input
        location = input("Where do you want to go (location)? ")
        cities = float(input("How many cities do you travel to per year? "))
        reviews = float(input("How many hotel reviews have you written? "))
        helpful = float(input("How many helpful votes have you received? "))

        # Step 2: Build cold user vector
        numeric_vector = scaler.transform([[helpful, cities, reviews]])
        location_vector = vectorizer_location.transform([location])
        cold_user_vector = hstack([csr_matrix(numeric_vector), location_vector])  # shape (1, 503)

        # Step 3: Compute similarity to existing users
        similarities = cosine_similarity(cold_user_vector, user_features_sparse)[0]

        # Step 4: Weighted collaborative recommendation
        scores = similarities @ user_item_matrix
        scores = np.array(scores).flatten()

        # Step 5: Top-K hotels
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_hotels = [(idx_to_hotel_id[i], scores[i]) for i in top_indices]

        # Step 6: Append new user data to persistent structures (optional)
        new_idx = len(user_id_to_idx)
        user_id_to_idx[user_id] = new_idx
        idx_to_user_id[new_idx] = user_id


        user_features_sparse = vstack([user_features_sparse, cold_user_vector])



        user_item_matrix = vstack([user_item_matrix, csr_matrix((1, user_item_matrix.shape[1]))])

        # Save updates (optional - persist later)
        with open('user_id_to_idx.json', 'w') as f:
            json.dump(user_id_to_idx, f)
        with open('idx_to_user_id.json', 'w') as f:
            json.dump({k: v for k, v in idx_to_user_id.items()}, f)

        # Return recommendations
        return top_hotels

    except Exception as e:
        print(f"Error: {e}")
        return []



city_penalty = {
    "New York City": 0.6,
    "Houston": 0.85,
    "San Antonio": 0.9,
    # Add others or default = 1.0
}

def apply_city_penalty(recommendations):
    global hotel_meta_df
    adjusted = []
    for hotel_id, score in recommendations:
        if hotel_id in hotel_meta_df.index:
            city = hotel_meta_df.loc[hotel_id]['locality']
            penalty = city_penalty.get(city, 1.0)
            adjusted.append((hotel_id, score * penalty))
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted[:10]

def cold_start_recommendation_combined(
    user_id: str,
    mode: str = "user",
    top_k: int = 10,
    # USER mode fields
    location: str = None,
    cities: float = None,
    reviews: float = None,
    helpful: float = None,
    # HOTEL mode fields
    service: float = None,
    cleanliness: float = None,
    overall: float = None,
    value: float = None,
    location_pref_score: float = None,
    sleep_quality: float = None,
    rooms: float = None,
    hotel_class: float = None,
    location_region: str = None
):
    global user_features_sparse
    global user_item_matrix
    global hotel_features_sparse

    try:
        if mode == "user":
            if None in (helpful, cities, reviews, location):
                raise ValueError("Missing user profile fields for user-mode cold start.")

            numeric_vector = scaler.transform([[helpful, cities, reviews]])
            location_vector = vectorizer_location.transform([location])
            cold_user_vector = hstack([csr_matrix(numeric_vector), location_vector])

            similarities = cosine_similarity(cold_user_vector, user_features_sparse)[0]
            scores = similarities @ user_item_matrix
            scores = np.array(scores).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_hotels = [(idx_to_hotel_id[i], scores[i]) for i in top_indices]

            # Update persistent data
            new_idx = len(user_id_to_idx)
            user_id_to_idx[user_id] = new_idx
            idx_to_user_id[new_idx] = user_id

            user_features_sparse = vstack([user_features_sparse, cold_user_vector])
            user_item_matrix = vstack([user_item_matrix, csr_matrix((1, user_item_matrix.shape[1]))])

            with open('user_id_to_idx.json', 'w') as f:
                json.dump(user_id_to_idx, f)
            with open('idx_to_user_id.json', 'w') as f:
                json.dump({k: v for k, v in idx_to_user_id.items()}, f)

            return top_hotels

        elif mode == "hotel":
            if None in (service, cleanliness, overall, value, location_pref_score,
                        sleep_quality, rooms, hotel_class, location_region):
                raise ValueError("Missing hotel preference fields for hotel-mode cold start.")

            hotel_features_sparse = load_npz(f"{base_dir}/hotel_features.npz")
            ohe = joblib.load(f'{base_dir}/hotel_region_ohe.pkl')

            if location_region not in ohe.categories_[0]:
                location_region = 'Unknown' if 'Unknown' in ohe.categories_[0] else ohe.categories_[0][0]

            location_encoded = ohe.transform([[location_region]]).toarray().flatten()

            user_categories = [service, cleanliness, overall, value,
                               location_pref_score, sleep_quality, rooms]
            avg_score = np.mean(user_categories)
            median_hotel_class = 3.0
            weighted_score_pref = avg_score * median_hotel_class * 10

            user_pref_vector = np.hstack((user_categories, [weighted_score_pref], location_encoded, hotel_class))
            cold_user_vector = csr_matrix(user_pref_vector).reshape(1, -1)

            hotel_class_idx = 25
            region_start_idx = 8

            hotel_class_col = hotel_features_sparse[:, hotel_class_idx].toarray().flatten()
            region_cols = hotel_features_sparse[:, region_start_idx:].toarray()

            hotel_class_mask = (hotel_class_col == hotel_class)
            region_idx = list(ohe.categories_[0]).index(location_region)
            region_mask = (region_cols[:, region_idx] == 1)

            combined_mask = region_mask & hotel_class_mask
            hotel_features_filtered = hotel_features_sparse[combined_mask]

            if hotel_features_filtered.shape[0] == 0:
                return []

            similarities = cosine_similarity(cold_user_vector, hotel_features_filtered)[0]
            top_indices_filtered = np.argsort(similarities)[::-1][:top_k]
            original_indices = np.where(combined_mask)[0]
            top_indices_original = original_indices[top_indices_filtered]

            top_hotels = [(idx_to_hotel_id[i], similarities[j]) for j, i in enumerate(top_indices_original)]
            return top_hotels

        else:
            raise ValueError("Invalid mode. Must be 'user' or 'hotel'.")

    except Exception as e:
        print(f"Error in cold start: {e}")
        print("No hotels matched for:")
        print("  hotel_class =", hotel_class)
        print("  location_region =", location_region)
        #print("  hotel_class options in dataset:", np.unique(hotel_class_col))
        #print("  available regions:", ohe.categories_[0])
        return []


def get_non_personalized_recommendations(top_k: int = 10, diversify: bool = False):
    global hotel_meta_df

    # Load necessary matrices
    hotel_features_sparse = load_npz(f'{base_dir}/hotel_features.npz')
    hotel_similarity = load_npz(f'{base_dir}/hotel_similarity_matrix.npz')

    # Load metadata
    df = hotel_meta_df.copy()
    
    # Load ID mappings
    with open(f"{base_dir}/hotel_idx_to_id.json") as f:
        idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}
    
    with open(f"{base_dir}/hotel_id_to_idx.json") as f:
        hotel_id_to_idx = {k: int(v) for k, v in json.load(f).items()}

    # Rebuild reverse mapping
    idx_to_hotel_meta = {hotel_id_to_idx[str(hid)]: (hid, row) for hid, row in df.iterrows() if str(hid) in hotel_id_to_idx}

    # Extract review-based scores from hotel_features_sparse
    #num_hotels = hotel_features_sparse.shape[0]

    # Columns used during construction
    # [service, cleanliness, overall, value, location, sleep_quality, rooms, weighted_score, region_ohe..., hotel_class]
    weighted_score_col = 7
    hotel_class_col = -1  # last column

    # Convert to dense arrays
    weighted_scores = hotel_features_sparse[:, weighted_score_col].toarray().flatten()
    hotel_classes = hotel_features_sparse[:, hotel_class_col].toarray().flatten()

    # Normalize both
    norm_scores = (weighted_scores - weighted_scores.min()) / (weighted_scores.max() - weighted_scores.min())
    norm_class = (hotel_classes - hotel_classes.min()) / (hotel_classes.max() - hotel_classes.min())

    # Final scoring
    combined_scores = 0.6 * norm_scores + 0.4 * norm_class

    # Diversify by boosting hotels similar to top ones
    if diversify:
        top_indices = np.argsort(combined_scores)[::-1][:20]
        sim_scores = hotel_similarity[top_indices].mean(axis=0).A1
        combined_scores += 0.1 * sim_scores  # Add slight boost for similar hotels

    # Final Top-K selection
    final_top_indices = np.argsort(combined_scores)[::-1][:top_k]

    recommendations = []
    for idx in final_top_indices:
        if idx in idx_to_hotel_meta:
            hotel_id, meta_row = idx_to_hotel_meta[idx]
            recommendations.append((hotel_id, combined_scores[idx]))

    return recommendations