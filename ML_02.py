import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
data = pd.read_csv("ML_02.csv")
numeric_columns = data.select_dtypes(include=[np.number]).columns
X = data[numeric_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.show()
optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print("Optimal number of clusters:", optimal_num_clusters)
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
data['Cluster'] = labels
plt.figure(figsize=(10, 6))
for cluster in range(optimal_num_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data.iloc[:, 2], cluster_data.iloc[:, 3], label=f'Cluster {cluster}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering of Customers (2D)')
plt.legend()
plt.show()
print(data[['CustomerID', 'Cluster']])
