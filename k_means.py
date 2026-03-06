import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def inertia(self, X, labels):
        """Sum of squared distances from each point to its centroid."""
        return sum(
            np.sum((X[labels == i] - self.centroids[i]) ** 2)
            for i in range(self.n_clusters)
        )


def elbow_plot(X, k_range=range(2, 15)):
    """Run KMeans for each K and plot inertia to find the elbow."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k)
        km.fit(X)
        labels = km._assign_labels(X)
        inertias.append(km.inertia(X, labels))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, 'o-', color='royalblue', linewidth=2, markersize=6)
    ax.set_xlabel('K (number of clusters)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Plot — Choose K where the curve bends', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('elbow_plot.png', dpi=150)
    print("Saved elbow_plot.png")
    plt.show()


def scatter_plots(X, labels, centroids, feature_names):
    """Plot all three pairwise combinations of the three orbital elements."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

    for ax, (i, j) in zip(axes, pairs):
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(X[mask, i], X[mask, j],
                       s=8, alpha=0.5, color=colors[cluster_id],
                       label=f'Cluster {cluster_id}')
        # Plot centroids on top
        ax.scatter(centroids[:, i], centroids[:, j],
                   s=120, marker='*', color='black', zorder=5, label='Centroids')
        ax.set_xlabel(feature_names[i], fontsize=11)
        ax.set_ylabel(feature_names[j], fontsize=11)
        ax.set_title(f'{feature_names[i]} vs {feature_names[j]}', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Single shared legend outside the plots
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.12))
    plt.suptitle('K-Means Clusters in Proper Orbital Element Space', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('cluster_scatter.png', dpi=150, bbox_inches='tight')
    print("Saved cluster_scatter.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

feature_columns = ['a', 'e', 'sin_i']
proper = pd.read_csv('proper_asteriod_data_clean.csv')
families = pd.read_csv('family_membership.csv')

df = proper.merge(families[['name','family1']], on='name', how='left')
df['family1'] = df['family1'].fillna(0).astype(int)

print(df.head())
X = df[feature_columns].dropna().values

# 1. Elbow plot — run this first to decide on K
# elbow_plot(X, k_range=range(2, 15))

# 2. Fit with chosen K
K = 30
kmeans = KMeans(n_clusters=K)
kmeans.fit(X)
labels = kmeans._assign_labels(X)

print("Cluster Assignments:", labels)
print("Final Centroids:\n", kmeans.centroids)

# 3. Scatter plots
scatter_plots(X, labels, kmeans.centroids, feature_columns)

mask = df['family1'] > 0
score = adjusted_rand_score(df.loc[mask, 'family1'], labels[mask])
print(f"Adjusted Rand Score: {score:.3f}")

results = []
for family_id in df['family1'].unique():
    if family_id == 0:
        continue

    true_members = df[df['family1'] == family_id].index
    dominant_cluster = pd.Series(labels[true_members]).mode()[0]

    # What fraction of true members landed in dominant cluster (recall/completeness)
    completeness = (labels[true_members] == dominant_cluster).mean()

    # What fraction of that cluster actually belongs to this family (precision)
    cluster_members = np.where(labels == dominant_cluster)[0]
    precision = (df.loc[cluster_members, 'family1'] == family_id).mean()

    # F1 combines both
    if precision + completeness > 0:
        f1 = 2 * precision * completeness / (precision + completeness)
    else:
        f1 = 0

    results.append({
        'family_id': family_id,
        'n_members': len(true_members),
        'completeness': completeness,
        'precision': precision,
        'f1': f1
    })

results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
print(results_df.to_string())
print(f"\nFamilies above 95% completeness AND precision: {((results_df['completeness'] >= 0.35) & (results_df['precision'] >= 0.35)).sum()}")