import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mode

# Specify the file path
file_path = r'F:\M.Eng\Courses\Machine Learning\Iris Dataset\iris.csv'

# Specify the column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load the dataset without headers and assign the column names
iris_df = pd.read_csv(file_path, header=None, names=column_names)

# Separate the petal features (petal_length and petal_width) for clustering
X_petal = iris_df[['petal_length', 'petal_width']].values  # Petal data only
y = iris_df['species'].values  # Extract Target labels (True Labels)

# Perform Spectral Clustering (3 clusters for the 3 Iris species) on petal data
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=25, random_state=42)
spectral_labels = spectral.fit_predict(X_petal)  # Get cluster labels for petal data

# Create a DataFrame with the original petal features and the labels
df = pd.DataFrame(X_petal, columns=['Petal Length', 'Petal Width'])
df['Cluster'] = spectral_labels  # Spectral clustering labels based on petal data
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

# Calculate purity for Spectral Clustering on petal data
purity = calculate_purity(df['True Labels Numeric'], df['Cluster'])
print(f'Purity (Petal Data, Spectral Clustering): {purity:.4f}')

### --- NMI Calculation --- ###
nmi = normalized_mutual_info_score(df['True Labels Numeric'], df['Cluster'])
print(f'NMI (Petal Data, Spectral Clustering): {nmi:.4f}')

### --- Visualization --- ###
# Create a subplot with 1 row and 2 columns to display both plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot the Spectral Clustering results on the first axis (Petal Data)
axes[0].scatter(df['Petal Length'], df['Petal Width'], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=100)
axes[0].set_title('Spectral Clustering (Petal Length vs Petal Width)')
axes[0].set_xlabel('Petal Length')
axes[0].set_ylabel('Petal Width')

# Plot the actual classes (True Labels) on the second axis (for comparison)
scatter = axes[1].scatter(df['Petal Length'], df['Petal Width'], c=df['True Labels Numeric'], cmap='viridis', marker='o', edgecolor='k', s=100)
axes[1].set_title('True Classes (Petal Length vs Petal Width)')
axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')

# Add a colorbar for the second plot
fig.colorbar(scatter, ax=axes[1], label='True Label')

# Adjust layout to prevent overlapping elements
plt.tight_layout()  # Adjust layout
plt.subplots_adjust(wspace=0.2)  # Adjust spacing between the plots

# Show both plots at the same time
plt.show()
