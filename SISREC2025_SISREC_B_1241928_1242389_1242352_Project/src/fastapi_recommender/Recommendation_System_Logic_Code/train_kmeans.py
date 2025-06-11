from sklearn.cluster import KMeans
import numpy as np
import pickle
from pathlib import Path

# Configuration
DATA_DIR = Path("src/fastapi_recommender/Recommendation_System_Logic_Code")
N_CLUSTERS = 5  # ajust as needed

# Charge les factors U
U_factors = np.load(DATA_DIR / "U_factors.npy")

# train KMeans
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans.fit(U_factors)

# 
with open(DATA_DIR / "user_cluster_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print(f"✅ KMeans trained with {N_CLUSTERS} clusters.")

print("Cluster centers:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: {center}")
    
# plot pca
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
U_factors_2d = pca.fit_transform(U_factors)
plt.figure(figsize=(10, 8))
plt.scatter(U_factors_2d[:, 0], U_factors_2d[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.title("KMeans Clustering of User Factors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster Label')
plt.savefig(DATA_DIR / "kmeans_clustering.png")
print("✅ KMeans clustering plot saved as 'kmeans_clustering.png'.")
# Show the plot
plt.show()

# plot 3d pca
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
U_factors_3d = PCA(n_components=3).fit_transform(U_factors)
ax.scatter(U_factors_3d[:, 0], U_factors_3d[:, 1], U_factors_3d[:, 2], c=kmeans.labels_, cmap='viridis', s=50)
ax.set_title("KMeans Clustering of User Factors (3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.savefig(DATA_DIR / "kmeans_clustering_3d.png")
print("✅ KMeans 3D clustering plot saved as 'kmeans_clustering_3d.png'.")
# Show the 3D plot
plt.show()
