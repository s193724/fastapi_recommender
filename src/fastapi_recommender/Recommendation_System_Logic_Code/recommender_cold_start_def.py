import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz, hstack, csr_matrix, vstack
import joblib
import json
import pandas as pd
from collections import defaultdict

#base_dir = '/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'
base_dir = '/Users/oliviapc/Documents/GitHub/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'

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

def add_cold_user(
    new_user_id: str,
    cold_user_vector,           # csr_matrix shape (1, feature_dim)
    hotel_features_sparse,      # csr_matrix shape (num_hotels, feature_dim)
    user_item_matrix_path: str = "user_hotel_matrix.npz",
    user_features_matrix_path: str = "user_features.npz",
    user_id_to_idx_path: str = "user_id_to_idx.json",
    idx_to_user_id_path: str = "idx_to_user_id.json",
):
    """
    Adds a cold start user to the user-item and user features matrices,
    updates user ID mappings, saves the updated data, and returns them.

    Returns:
    - user_item_matrix_updated: csr_matrix
    - user_features_sparse_updated: csr_matrix
    - user_id_to_idx: dict
    - idx_to_user_id: dict
    """

    # Load existing data
    user_item_matrix = load_npz(user_item_matrix_path)
    user_features_sparse = load_npz(user_features_matrix_path)

    with open(user_id_to_idx_path, "r") as f:
        user_id_to_idx = json.load(f)

    with open(idx_to_user_id_path, "r") as f:
        idx_to_user_id = {int(k): v for k, v in json.load(f).items()}

    # Check if user already exists
    if new_user_id in user_id_to_idx:
        raise ValueError(f"User ID '{new_user_id}' already exists.")

    # Compute similarity of cold user vector to all hotels
    similarities = cosine_similarity(cold_user_vector, hotel_features_sparse).flatten()

    # Convert similarities to sparse row vector (1 x num_hotels)
    cold_user_interactions = csr_matrix(similarities)

    # Append cold user interaction row to user-item matrix
    user_item_matrix_updated = vstack([user_item_matrix, cold_user_interactions])

    # Append cold user vector row to user features matrix
    user_features_sparse_updated = vstack([user_features_sparse, cold_user_vector])

    # Update mappings with new user index
    new_idx = user_item_matrix.shape[0]  # next index (0-based)
    user_id_to_idx[new_user_id] = new_idx
    idx_to_user_id[new_idx] = new_user_id

    # Save updated matrices and mappings
    save_npz(user_item_matrix_path, user_item_matrix_updated)
    save_npz(user_features_matrix_path, user_features_sparse_updated)

    with open(user_id_to_idx_path, "w") as f:
        json.dump(user_id_to_idx, f)

    with open(idx_to_user_id_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_user_id.items()}, f)

    print(f"Added cold user '{new_user_id}' as index {new_idx}.")

    return user_item_matrix_updated, user_features_sparse_updated, user_id_to_idx, idx_to_user_id

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
        #()numeric_vector = scaler.transform([[helpful, cities, reviews]])
        #()location_vector = vectorizer_location.transform([location])
        df_numeric = pd.DataFrame([[helpful, cities, reviews]], columns=scaler.feature_names_in_)
        numeric_vector = scaler.transform(df_numeric)
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
    offering_id: float = None,
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
            helpful = 0.0 if helpful is None else helpful
            cities = 0.0 if cities is None else cities
            reviews = 0.0 if reviews is None else reviews
            location = location if location is not None else "Unknown"

            # Protegeix contra ubicacions desconegudes
            if location not in vectorizer_location.vocabulary_:
                location = "Unknown"
                
            #()numeric_vector = scaler.transform([[helpful, cities, reviews]])
            #()location_vector = vectorizer_location.transform([location])
            df_numeric = pd.DataFrame([[helpful, cities, reviews]], columns=scaler.feature_names_in_)
            numeric_vector = scaler.transform(df_numeric)
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
            # 1. Validació tova (acepta np.nan i None)
            def _safe(x):    # helper
                return 0 if (x is None or (isinstance(x, float) and np.isnan(x))) else x

            service, cleanliness, overall, value = map(_safe, (service, cleanliness, overall, value))
            location_pref_score, sleep_quality, rooms = map(_safe, (location_pref_score, sleep_quality, rooms))
            hotel_class = _safe(hotel_class)

            # 2. One-hot encoder i crea vector usuari
            ohe = joblib.load(f'{base_dir}/hotel_region_ohe.pkl')
            if location_region not in ohe.categories_[0]:
                location_region = 'Unknown' if 'Unknown' in ohe.categories_[0] else ohe.categories_[0][0]
            #location_encoded = ohe.transform([[location_region]]).A1
            #()location_encoded = ohe.transform([[location_region]]).toarray().ravel()
            df_region = pd.DataFrame([location_region], columns=ohe.feature_names_in_)
            location_encoded = ohe.transform(df_region).toarray().ravel()

            user_categories = [service, cleanliness, overall, value,
                            location_pref_score, sleep_quality, rooms]
            avg_score = np.mean(user_categories)
            weighted_score_pref = avg_score * 3.0 * 10      # 3.0 = classe mitjana

            user_pref_vector = np.hstack((user_categories, [weighted_score_pref], location_encoded, [hotel_class]))
            cold_user_vector = csr_matrix(user_pref_vector)

            # 3. Índex dinàmic de columnes
            region_start_idx = 8
            hotel_class_idx = region_start_idx + len(ohe.categories_[0])

            #hotel_class_col = hotel_features_sparse[:, hotel_class_idx].A1
            hotel_class_col = hotel_features_sparse[:, hotel_class_idx].toarray().ravel()

            #region_cols = hotel_features_sparse[:, region_start_idx:].A
            region_cols = hotel_features_sparse[:, region_start_idx:].toarray()


            hotel_class_mask = (hotel_class_col == hotel_class)
            region_mask = (region_cols[:, list(ohe.categories_[0]).index(location_region)] == 1)
            combined_mask = region_mask & hotel_class_mask
            hotel_features_filtered = hotel_features_sparse[combined_mask]

            if hotel_features_filtered.shape[0] == 0:
                combined_mask = region_mask        # relaxem una mica
                hotel_features_filtered = hotel_features_sparse[combined_mask]

            similarities = cosine_similarity(cold_user_vector, hotel_features_filtered).ravel()
            top_idx_filtered = np.argsort(similarities)[::-1][:top_k]
            original_idx = np.where(combined_mask)[0][top_idx_filtered]

            top_hotels = [(idx_to_hotel_id[i], similarities[j])      # ús correcte de j
                        for j, i in zip(top_idx_filtered, original_idx)]
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
        #sim_scores = hotel_similarity[top_indices].mean(axis=0).A1
        sim_scores = hotel_similarity[top_indices].mean(axis=0).toarray().ravel()
        combined_scores += 0.1 * sim_scores  # Add slight boost for similar hotels

    # Final Top-K selection
    final_top_indices = np.argsort(combined_scores)[::-1][:top_k]

    recommendations = []
    for idx in final_top_indices:
        if idx in idx_to_hotel_meta:
            hotel_id, meta_row = idx_to_hotel_meta[idx]
            recommendations.append((hotel_id, combined_scores[idx]))

    return recommendations
"""
def print_recommendations_with_features(recommendations, hotel_meta_df):
    for hotel_id, score in recommendations:
        if hotel_id in hotel_meta_df.index:
            features = hotel_meta_df.loc[hotel_id].to_dict()
            print(f"Hotel ID: {hotel_id}, Score: {score:.4f}")
            print("Features:")
            for k, v in features.items():
                print(f"  {k}: {v}")
            print("-" * 40)
        else:
            print(f"Hotel ID: {hotel_id} not found in metadata.")
"""