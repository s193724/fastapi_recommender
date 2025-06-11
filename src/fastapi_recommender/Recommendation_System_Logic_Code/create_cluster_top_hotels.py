import numpy as np
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.sparse import load_npz
import pickle

DATA_DIR = Path("src/fastapi_recommender/Recommendation_System_Logic_Code")

# Charge data
U_factors = np.load(DATA_DIR / "U_factors.npy")
user_item_matrix = load_npz(DATA_DIR / "user_hotel_matrix.npz")
with open(DATA_DIR / "filtered_user_id_to_idx.json") as f:
    user_id_to_idx = json.load(f)
with open(DATA_DIR / "hotel_idx_to_id.json") as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}
with open(DATA_DIR / "user_cluster_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Assign cluster labels to users
cluster_assignments = kmeans.predict(U_factors)

# Prepare dictionary to hold hotel scores per cluster
cluster_hotel_scores = defaultdict(lambda: defaultdict(float))

# Iteration users and scores
for user_id, user_idx in user_id_to_idx.items():
    if user_idx >= len(cluster_assignments):
        continue  # o pots fer logging si vols rastrejar-ho
    cluster_id = int(cluster_assignments[int(user_idx)])
    user_ratings = user_item_matrix[int(user_idx)].toarray().flatten()
    for hotel_idx, score in enumerate(user_ratings):
        if score > 0:
            hotel_id = idx_to_hotel_id[hotel_idx]
            cluster_hotel_scores[cluster_id][hotel_id] += score

# Obtain top hotels per cluster
cluster_top_hotels = {
    str(cluster): sorted(hotels.items(), key=lambda x: -x[1])[:20]  # top 20
    for cluster, hotels in cluster_hotel_scores.items()
}

# Convert to JSON (only hotel IDs)
cluster_top_hotels_cleaned = {
    cluster: [hotel_id for hotel_id, _ in hotel_list]
    for cluster, hotel_list in cluster_top_hotels.items()
}

with open(DATA_DIR / "cluster_top_hotels.json", "w") as f:
    json.dump(cluster_top_hotels_cleaned, f)

print("✅ File 'cluster_top_hotels.json' created.")
