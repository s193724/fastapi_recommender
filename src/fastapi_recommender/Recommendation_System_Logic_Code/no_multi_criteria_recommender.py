import pickle
import numpy as np

ATTR_COLS = [
    "service", "cleanliness", "overall", "value",
    "location", "sleep_quality", "rooms",
]
DEFAULT_WEIGHTS = np.array([0.15, 0.15, 0.25, 0.15, 0.1, 0.1, 0.1])
MIN_RATINGS_FOR_PERSONAL = 5

# Load factors once on import
FACTORS_FILE = "tensor_factors.pkl"
with open(FACTORS_FILE, 'rb') as f:
    weights_cp, factors_cp, user_id_to_idx, hotel_id_to_idx = pickle.load(f)

U_f, H_f, A_f = factors_cp

# Precompute some helper structures
inv_hotel = {v: k for k, v in hotel_id_to_idx.items()}

# Assume you have user_attr_mean and user_attr_counts precomputed similarly
# Load from precomputed csv or json, or add code here to compute from REVIEWS_DF.csv

def compute_user_weights(user_id, user_attr_mean, user_attr_counts):
    if user_id not in user_attr_mean or user_attr_counts.get(user_id, 0) < MIN_RATINGS_FOR_PERSONAL:
        return DEFAULT_WEIGHTS
    pref = user_attr_mean[user_id]
    exp_p = np.exp(pref)
    soft = exp_p / exp_p.sum()
    weights = 0.7 * soft + 0.3 * DEFAULT_WEIGHTS
    return weights / weights.sum()

_attr_weight_cache = {}

def _cached_attr_weight_vector(weights_vec):
    key = tuple(np.round(weights_vec, 3))
    if key in _attr_weight_cache:
        return _attr_weight_cache[key]
    s_k = (A_f * weights_vec[:, None]).sum(axis=0)
    _attr_weight_cache[key] = s_k
    return s_k

def recommend(user_id, user_attr_mean, user_attr_counts, top_k=10):
    if user_id not in user_id_to_idx:
        raise ValueError("Unknown user")
    uid = user_id_to_idx[user_id]
    w = compute_user_weights(user_id, user_attr_mean, user_attr_counts)
    s_k = _cached_attr_weight_vector(w)
    user_lat = U_f[uid] * s_k
    scores = user_lat @ H_f.T
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(inv_hotel[i], float(scores[i])) for i in top_idx]
