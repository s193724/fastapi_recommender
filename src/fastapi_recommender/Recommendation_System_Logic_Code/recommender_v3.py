import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.preprocessing import normalize
import json

from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_cold_start_def import (
    cold_start_recommendation_combined, apply_city_penalty,get_non_personalized_recommendations
)
from src.fastapi_recommender.Recommendation_System_Logic_Code.multi_criteria_recommender import multi_criteria_recommender


#base_dir = '/Users/filiporlikowski/Documents/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'
base_dir = '/Users/oliviapc/Documents/GitHub/fastapi_recommender/src/fastapi_recommender/Recommendation_System_Logic_Code'

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
    If new user → cold start
    If user with multi-criteria reviews → use multi-criteria recommender
    If user with only simple ratings → classic hybrid CF+CBF
    """
    
    if user_id not in user_id_to_idx:
        return cold_start_recommendation_combined(user_id, top_k=top_k)

    if user_has_multi_reviews(user_id):
        # Usa el recommender multi-criteri
        return multi_criteria_recommender(user_id, top_k=top_k)
    else:
        # Mantenim l'híbrid clàssic
        user_idx = user_id_to_idx[user_id]

        user_sim_scores = user_similarity[user_idx].toarray().flatten()
        if np.sum(user_sim_scores) > 0:
            collab_scores = (user_sim_scores @ user_item_matrix) / np.sum(user_sim_scores)
        else:
            collab_scores = np.zeros(user_item_matrix.shape[1])

        user_ratings = user_item_matrix[user_idx].toarray().flatten()
        item_sim_scores = user_ratings @ hotel_similarity

        collab_scores = normalize(collab_scores.reshape(1, -1))[0]
        item_sim_scores = normalize(item_sim_scores.reshape(1, -1))[0]

        hybrid_scores = alpha * collab_scores + (1 - alpha) * item_sim_scores

        rated_indices = np.where(user_ratings > 0)[0]
        hybrid_scores[rated_indices] = 0

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        recommended_hotel_ids = [idx_to_hotel_id[i] for i in top_indices]
        recommended_scores = [hybrid_scores[i] for i in top_indices]

        return list(zip(recommended_hotel_ids, recommended_scores))


def user_has_multi_reviews(user_id):
    # Implementa la lògica real segons com guardis les reviews
    # Exemple: 
    return user_id in user_id_to_idx and np.sum(user_item_matrix[user_id_to_idx[user_id]].toarray()) > 0

# --- Example usage ---
user_id = "3199EEEDB7088BA85AAE3F7DB9BC224"  # replace with real user
recommendations = hybrid_recommend(user_id, alpha=0.7, top_k=10)
recommendations = apply_city_penalty(recommendations)
print("Top recommendations:")
for hotel_id, score in recommendations:
    print(f"Hotel {hotel_id} — Score: {score:.4f}")

