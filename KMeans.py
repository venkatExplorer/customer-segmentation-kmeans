import pandas as pd 
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\venka\Downloads\cars-1.csv")

car_features = df[['mpg','disp','hp']]

kmeans_3 = KMeans(n_clusters= 3, random_state=42)
car_features['Cluster_3'] = kmeans_3.fit_predict(car_features)

cluster_3_groups = [car_features[car_features['Cluster_3'] == i] for i in range(3)]

kmeans_5 = KMeans(n_clusters=5, random_state=42)
car_features['Cluster_5'] = kmeans_5.fit_predict(car_features[['mpg', 'disp', 'hp']])


cluster_5_groups = [car_features[car_features['Cluster_5'] == i] for i in range(5)]

print("3 Clusters:")
for i, group in enumerate(cluster_3_groups):
    print(f"/nCluster {i}:\n", group)
print("\n5 Clusters:")
for i, group in enumerate(cluster_5_groups):
    print(f"\nCluster {i}:\n", group)