import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 

df = pd.read_csv(r"C:\Users\venka\Downloads\customers.csv")

print("First 5 rows of the dataset:\n", df.head())
print("\nNull values in each column:\n", df.isnull().sum())

plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['Spending Score (1-100)'], c = 'blue', edgecolors='k')
plt.xlabel("Age")
plt.ylabel("Spending Score(1-100)")
plt.title("Age vs Spending Score")
plt.grid(True)
plt.show()

X = df[['Age','Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Age', y = 'Spending Score (1-100)', hue='Cluster',palette='Set2', s= 100)
plt.title('K-Means Clustering: Age vs Spending Score')
plt.grid(True)
plt.show()