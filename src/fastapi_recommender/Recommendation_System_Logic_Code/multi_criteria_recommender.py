# multi_criteria_recommender.py
"""
Multi‑criteria hotel recommender – **user‑adaptive weights**
===========================================================
Aquest mòdul construeix i factoritza un tensor de valoracions
**R[user, hotel, attribute]** a partir de `REVIEWS_DF.csv` i genera
recomanacions que combinen contingut i filtratge col·laboratiu.  Ara:
• **Aprèn factors latents** amb CP‑ALS (TensorLy).
• **Deriva pesos per atribut** Individuals per usuari – si l’usuari ha
  puntuat molts hotels i mostra preferència clara per algun criteri,
  aquests criteris es ponderen més en la recomanació.
• Proporciona `recommend(user_id, top_k)` que es pot cridar des del teu
  backend FastAPI (vegeu exemple a baix).

Dependències (instal·la un cop):
```
pip install pandas numpy scipy tensorly==0.8.1
```
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import tensorly as tl
from tensorly.decomposition import parafac
import sparse

# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "Recommendation_System_Logic_Code"
REVIEWS_CSV = DATA_DIR / "REVIEWS_DF.csv"
USER_MAP   = DATA_DIR / "user_id_to_idx.json"
HOTEL_MAP  = DATA_DIR / "hotel_id_to_idx.json"

ATTR_COLS = [
    "service", "cleanliness", "overall", "value",
    "location", "sleep_quality", "rooms",
]
N_ATTR = len(ATTR_COLS)
DEFAULT_WEIGHTS = np.array([0.15, 0.15, 0.25, 0.15, 0.1, 0.1, 0.1])
MIN_RATINGS_FOR_PERSONAL = 5   # requerim >=5 reviews d’usuari per personalitzar
RANK = 25  # factorització (es pot validar)
# ---------------------------------------------------------------------------
print("[MCR] Loading data …")
reviews_df = pd.read_csv(REVIEWS_CSV, usecols=["offering_id", "id_user", *ATTR_COLS])
with USER_MAP.open() as f: user_id_to_idx = {k: int(v) for k, v in json.load(f).items()}
with HOTEL_MAP.open() as f: hotel_id_to_idx = {k: int(v) for k, v in json.load(f).items()}

U, H = max(user_id_to_idx.values()) + 1, max(hotel_id_to_idx.values()) + 1
print(f"[MCR] Tensor dims ⇒ users:{U}, hotels:{H}, attrs:{N_ATTR}")

# ---------------- Sparse tensor ------------------------------------------------
coords, vals = [], []
for r in reviews_df.itertuples(index=False):
    u = user_id_to_idx.get(r.id_user); h = hotel_id_to_idx.get(str(int(r.offering_id)))
    if u is None or h is None: continue
    for a_idx, col in enumerate(ATTR_COLS):
        rating = getattr(r, col)
        if np.isnan(rating) or rating == 0: continue
        coords.append((u, h, a_idx)); vals.append(float(rating))
coords = np.array(coords).T; vals = np.array(vals, dtype=np.float32)
X_sparse = sparse.COO(coords, vals, shape=(U, H, N_ATTR))

# ---------------- Factorització CP‑ALS ----------------------------------------
print("[MCR] Factorising tensor (CP‑ALS)…")
weights, factors = parafac(X_sparse, rank=RANK, n_iter_max=150, init='random', random_state=42)
U_f, H_f, A_f = factors  # (U,R) (H,R) (A,R)

# ---------------- Helper: pes d’usuari per atribut ----------------------------
user_attr_mean = reviews_df.groupby('id_user')[ATTR_COLS].mean()
user_attr_counts = reviews_df.groupby('id_user')["offering_id"].count()

def compute_user_weights(user_id: str) -> np.ndarray:
    """Return normalized attribute‑weights for *user_id*.
    • Si l’usuari té poques ressenyes (<MIN_RATINGS_FOR_PERSONAL) → DEFAULT_WEIGHTS
    • Sinó: vector de preferència (=mitjana d’atributs), softmax suau + blend.
    """
    if user_id not in user_attr_mean.index or user_attr_counts[user_id] < MIN_RATINGS_FOR_PERSONAL:
        return DEFAULT_WEIGHTS
    pref = user_attr_mean.loc[user_id].values.astype(float)
    # softmax per obtenir distribució
    exp_p = np.exp(pref)
    soft = exp_p / exp_p.sum()
    # blend 70% personal + 30% default per evitar weights 0
    weights = 0.7 * soft + 0.3 * DEFAULT_WEIGHTS
    return weights / weights.sum()

# ---------------- Score matrix quick compute ----------------------------------
attr_weight_cache: dict[str, np.ndarray] = {}

def _cached_attr_weight_vector(weights_vec: np.ndarray):
    key = tuple(np.round(weights_vec, 3))
    if key in attr_weight_cache: return attr_weight_cache[key]
    # attr score for each latent dim: s_k = Σ_a w_a * A_f[a,k]
    s_k = (A_f * weights_vec[:, None]).sum(axis=0)  # shape (R,)
    attr_weight_cache[key] = s_k
    return s_k

# ---------------- Public API ---------------------------------------------------

inv_hotel = {v: int(k) for k, v in hotel_id_to_idx.items()}

def recommend(user_id: str, top_k: int = 10) -> list[tuple[int, float]]:
    """Return top‑k (hotel_id, score) for *user_id* using adaptive weights."""
    if user_id not in user_id_to_idx:
        raise ValueError("Unknown user_id")
    uid = user_id_to_idx[user_id]
    w = compute_user_weights(user_id)
    s_k = _cached_attr_weight_vector(w)  # (R,)
    # Score vector for user uid:   score_h = Σ_k (U_f[uid,k] * s_k[k]) * H_f[h,k]
    user_lat = U_f[uid] * s_k            # shape (R,)
    scores = user_lat @ H_f.T            # (H,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(inv_hotel[i], float(scores[i])) for i in top_idx]

# --------------- Example run ---------------------------------------------------
if __name__ == "__main__":
    some_user = list(user_id_to_idx.keys())[0]
    print(f"Top‑5 per a {some_user} (weights adaptatius):")
    for hid, sc in recommend(some_user, 5):
        print(f"  {hid}  →  {sc:.3f}")
