import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.preprocessing import normalize
import json
import pickle

from src.fastapi_recommender.Recommendation_System_Logic_Code.recommender_cold_start_def import (
    cold_start_recommendation_combined, apply_city_penalty,get_non_personalized_recommendations)
#from fastapi_recommender.Recommendation_System_Logic_Code.no_multi_criteria_recommender import recommend


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
    # Load KMeans cluster model and user latent factors
with open(f'{base_dir}/user_cluster_model.pkl', "rb") as f:
    kmeans = pickle.load(f)
U_factors = np.load(f'{base_dir}/U_factors.npy')
with open(f'{base_dir}/cluster_top_hotels.json') as f:
    cluster_top_hotels = json.load(f)

# --- Load persistent data --- /src/fastapi_recommender/Recommendation_System_Logic_Code/
user_features_sparse = load_npz(f'{base_dir}/user_features_sparse.npz')
user_item_matrix = load_npz(f'{base_dir}/user_hotel_matrix.npz')
user_features_collab = load_npz(f'{base_dir}/user_features_collab.npz')
hotel_meta_df = pd.read_csv(f'{base_dir}/hotel_df.csv')
hotel_meta_df.set_index('offering_id', inplace=True)

# Al principi del mòdul
global_min, global_max = np.inf, -np.inf

# --- Recommend for a target user ---
def hybrid_recommend(user_id, alpha=0.7, top_k=10):
    print(f"Called hybrid_recommend with user_id={user_id}")

    # -------- 0) cold-start --------
    if user_id not in user_id_to_idx:
        return cold_start_recommendation_combined(user_id, top_k=top_k)

    user_idx     = user_id_to_idx[user_id]
    user_ratings = user_item_matrix[user_idx].toarray().flatten()

    # -------- 1) historial buit => cluster --------
    if user_ratings.sum() == 0:
        cluster_id    = kmeans.predict(U_factors[user_idx].reshape(1, -1))[0]
        cluster_hotels = cluster_top_hotels.get(str(cluster_id), [])
        return [(hid, 1.0) for hid in cluster_hotels[:top_k]]

    # -------- 2) scores col·laboratius + contingut --------
    user_sim      = user_similarity[user_idx].toarray().flatten()
    collab_scores = (user_sim @ user_item_matrix) / user_sim.sum() if user_sim.sum() else np.zeros(user_item_matrix.shape[1])
    item_scores   = user_ratings @ hotel_similarity

    # normalitza cada vector per separat
    collab_scores = normalize(collab_scores.reshape(1, -1))[0]
    item_scores   = normalize(item_scores.reshape(1, -1))[0]

    # combinació i normalització global
    hybrid_scores = alpha * collab_scores + (1 - alpha) * item_scores
    hybrid_scores = normalize(hybrid_scores.reshape(1, -1))[0]

    # -------- 3) elimina hotels ja valorats ABANS de min/max --------
    rated_idx = np.where(user_ratings > 0)[0]
    hybrid_scores[rated_idx] = 0

    valid_mask   = hybrid_scores > 0
    valid_scores = hybrid_scores[valid_mask]

    if valid_scores.size == 0:                     # cap score positiu
        print("User has no positive hybrid scores → fallback cluster")
        cluster_id    = kmeans.predict(U_factors[user_idx].reshape(1, -1))[0]
        cluster_hotels = cluster_top_hotels.get(str(cluster_id), [])
        return [(hid, 1.0) for hid in cluster_hotels[:top_k]]

    min_s, max_s = valid_scores.min(), valid_scores.max()
    print(f"Hybrid scores (after zeroing rated) → min={min_s:.4f}, max={max_s:.4f}")

    # -------- 4) NORMALITZACIÓ 1-10 NOMÉS on valid_mask --------
    norm_scores = np.zeros_like(hybrid_scores)
    if max_s > min_s:
        norm_scores[valid_mask] = 1 + 9 * (hybrid_scores[valid_mask] - min_s) / (max_s - min_s)
    else:
        norm_scores[valid_mask] = 10.0             # tots iguals

    # rated_idx ja estan a 0, es mantenen

    # -------- 5) Top-k --------
    top_idx   = np.argsort(norm_scores)[::-1][:top_k]
    rec_ids   = [idx_to_hotel_id[i] for i in top_idx]
    rec_scores = norm_scores[top_idx]

    print(f"Normalized top scores: {rec_scores}")

    return list(zip(rec_ids, rec_scores))

"""# --- Example usage ---
user_id = "EEE0674F7271A66FACABBB1EE20A164E"
recommendations = hybrid_recommend(user_id, alpha=0.7, top_k=10)
recommendations = apply_city_penalty(recommendations)

print("Top recommendations (normalized 1–10):")
for hotel_id, score in recommendations:
    if hotel_id in hotel_meta_df.index:
        info = hotel_meta_df.loc[hotel_id]
        print(f"{info.get('name','N/A')} (ID {hotel_id}) — "
              f"{info.get('hotel_class','N/A')}★, {info.get('locality','N/A')} "
              f"— Score: {score:.2f}")
    else:
        print(f"Hotel {hotel_id} — Score: {score:.2f} (sense metadades)")"""



