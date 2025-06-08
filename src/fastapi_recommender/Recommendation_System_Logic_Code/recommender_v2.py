import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.preprocessing import normalize
import json

from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_cold_start_def import (
    cold_start_recommendation_combined, apply_city_penalty,get_non_personalized_recommendations
)

base_dir = '/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'
# --- Load data ---
user_item_matrix = load_npz(f'{base_dir}/user_hotel_matrix.npz')
user_similarity = load_npz(f'{base_dir}/user_similarity_collab.npz')
hotel_similarity = load_npz(f'{base_dir}/hotel_similarity_matrix.npz')

with open(f'{base_dir}/user_id_to_idx.json') as f:
    user_id_to_idx = json.load(f)
with open(f'{base_dir}/idx_to_user_id.json') as f:
    idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
with open(f'{base_dir}/hotel_id_to_idx.json') as f:
    hotel_id_to_idx = json.load(f)
with open(f'{base_dir}/hotel_idx_to_id.json') as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

# --- Load persistent data --- /src/fastapi_recommender/Recommendation_System_Logic_Code/
user_features_sparse = load_npz(f'{base_dir}/user_features_sparse.npz')
user_item_matrix = load_npz(f'{base_dir}/user_hotel_matrix.npz')
user_features_collab = load_npz(f'{base_dir}/user_features_collab.npz')
hotel_meta_df = pd.read_csv(f'{base_dir}/hotel_df.csv')
hotel_meta_df.set_index('offering_id', inplace=True)


# --- Recommend for a target user ---
def hybrid_recommend(user_id, alpha=0.7, top_k=10):
    """
    Generate hybrid recommendations for a given user.

    alpha: weight for collaborative filtering [0-1]. (1-alpha) is for content-based
    """
    if user_id not in user_id_to_idx:
       #return cold_start_recommendation(user_id, top_k=top_k)
       #return cold_start_recommendation_collab(user_id)
    #    print("Welcome new user. Would you like to receive reccomendations based on users with meatdata variables similar to yours(type: 'user' or ")
    #    mode = input("find the best hotel based on your own preferences(type: 'hotel'))? ")
       return cold_start_recommendation_combined(user_id, top_k=top_k)

    user_idx = user_id_to_idx[user_id]

    # Collaborative filtering: weighted sum of similar users
    user_sim_scores = user_similarity[user_idx].toarray().flatten()
    #collab_scores = user_sim_scores @ user_item_matrix
    if np.sum(user_sim_scores) > 0:
        collab_scores = (user_sim_scores @ user_item_matrix) / np.sum(user_sim_scores)

    # Content-based: similar items to user's already rated hotels
    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    item_sim_scores = user_ratings @ hotel_similarity

    # Normalize both scores
    collab_scores = normalize(collab_scores.reshape(1, -1))[0]
    item_sim_scores = normalize(item_sim_scores.reshape(1, -1))[0]

    # Hybrid score
    hybrid_scores = alpha * collab_scores + (1 - alpha) * item_sim_scores

    # Exclude already rated hotels
    rated_indices = np.where(user_ratings > 0)[0]
    hybrid_scores[rated_indices] = 0

    # Top-K recommendation indices
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    recommended_hotel_ids = [idx_to_hotel_id[i] for i in top_indices]
    recommended_scores = [hybrid_scores[i] for i in top_indices]

    return list(zip(recommended_hotel_ids, recommended_scores))


# --- Example usage ---
user_id = "3199EEEDB7088BA85AAE3F7DB9BC224"  # replace with real user
recommendations = hybrid_recommend(user_id, alpha=0.7, top_k=10)
recommendations = apply_city_penalty(recommendations)
print("Top recommendations:")
for hotel_id, score in recommendations:
    print(f"Hotel {hotel_id} — Score: {score:.4f}")
