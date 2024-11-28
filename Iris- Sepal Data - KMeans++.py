import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mode

# Specify the file path
file_path = r'F:\M.Eng\Courses\Machine Learning\Iris Dataset\iris.csv'

# Specify the column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load the dataset without headers and assign the column names
iris_df = pd.read_csv(file_path, header=None, names=column_names)

# Separate the sepal features (sepal_length and sepal_width) for clustering
X_sepal = iris_df[['sepal_length', 'sepal_width']].values  # Sepal data only
y = iris_df['species'].values  # Extract Target labels (True Labels)

# Perform KMeans++ clustering (3 clusters for the 3 Iris species) on sepal data
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(X_sepal)  # Get cluster labels for sepal data

# Create a DataFrame with the original sepal features and the labels
df = pd.DataFrame(X_sepal, columns=['Sepal Length', 'Sepal Width'])
df['Cluster'] = kmeans_labels  # KMeans++ cluster labels based on sepal data
df['True Labels'] = y  # True species names

# Convert True Labels (species) to numeric values for metrics calculations
df['True Labels Numeric'] = pd.factorize(df['True Labels'])[0]

### --- Purity Calculation --- ###
def calculate_purity(true_labels, cluster_labels):
    # Create a contingency matrix
    contingency_matrix = pd.crosstab(cluster_labels, true_labels)

    # Calculate purity
    correct_assignments = contingency_matrix.max(axis=1).sum()
    purity = correct_assignments / len(true_labels)

    return purity

# Calculate purity for KMeans++ on sepal data
purity = calculate_purity(df['True Labels Numeric'], df['Cluster'])
print(f'Purity (Sepal Data, KMeans++): {purity:.4f}')

### --- NMI Calculation --- ###
nmi = normalized_mutual_info_score(df['True Labels Numeric'], df['Cluster'])
print(f'NMI (Sepal Data, KMeans++): {nmi:.4f}')

### --- Visualization --- ###
# Create a subplot with 1 row and 2 columns to display both plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot the KMeans++ clustering results on the first axis (Sepal Data)
axes[0].scatter(df['Sepal Length'], df['Sepal Width'], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=100)
axes[0].set_title('KMeans++ Clustering (Sepal Length vs Sepal Width)')
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')

# Plot the actual classes (True Labels) on the second axis (for comparison)
scatter = axes[1].scatter(df['Sepal Length'], df['Sepal Width'], c=df['True Labels Numeric'], cmap='viridis', marker='o', edgecolor='k', s=100)
axes[1].set_title('True Classes (Sepal Length vs Sepal Width)')
axes[1].set_xlabel('Sepal Length')
axes[1].set_ylabel('Sepal Width')

# Add a colorbar for the second plot
fig.colorbar(scatter, ax=axes[1], label='True Label')

# Adjust layout to prevent overlapping elements
plt.tight_layout()  # Adjust layout
plt.subplots_adjust(wspace=0.2)  # Adjust spacing between the plots

# Show both plots at the same time
plt.show()
