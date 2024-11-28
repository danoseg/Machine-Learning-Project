import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import mode

# Specify the file path
file_path = r'F:\M.Eng\Courses\Machine Learning\Iris Dataset\iris.csv'

# Specify the column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load the dataset without headers and assign the column names
iris_df = pd.read_csv(file_path, header=None, names=column_names)

# Separate features and label
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values  # Extract Features
y = iris_df['species'].values  # Extract Target labels

# Perform KMeans++ clustering (3 clusters for the 3 Iris species)
kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++')  # KMeans++ initialization
kmeans.fit(X)
labels = kmeans.labels_

# Perform PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame with the PCA components and the labels
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = labels  # KMeans cluster labels
df['True Labels'] = y  # True species names

# Convert True Labels (species) to numeric values for coloring in the plot
df['True Labels Numeric'] = pd.factorize(df['True Labels'])[0]

# Remap the KMeans cluster labels to the most common true label in each cluster
new_labels = pd.Series([0] * len(df))  # Initialize with default labels
for i in range(3):  # Adjust for the number of clusters
    mask = (df['Cluster'] == i)
    most_common = mode(df[mask]['True Labels Numeric'])[0][0]  # Find the most common true label in this cluster
    new_labels[mask] = most_common  # Remap the cluster label to match the true label

df['Remapped Cluster'] = new_labels  # Add the remapped labels to the dataframe

# Create a subplot with 1 row and 2 columns to display both plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot the KMeans clusters (after remapping) on the first axis
axes[0].scatter(df['PC1'], df['PC2'], c=df['Remapped Cluster'], cmap='viridis', marker='o', edgecolor='k', s=100)
axes[0].set_title('KMeans++ Clustering of Iris Dataset (PCA-reduced)')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')

# Plot the actual classes (True Labels) on the second axis
scatter = axes[1].scatter(df['PC1'], df['PC2'], c=df['True Labels Numeric'], cmap='viridis', marker='o', edgecolor='k', s=100)
axes[1].set_title('True Classes of Iris Dataset (PCA-reduced)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')

# Add a colorbar for the second plot
fig.colorbar(scatter, ax=axes[1], label='True Label')

# Adjust layout to prevent overlapping elements
plt.tight_layout()  # Adjust layout
plt.subplots_adjust(wspace=0.2)  # Adjust spacing between the plots

# Show both plots at the same time
plt.show()
