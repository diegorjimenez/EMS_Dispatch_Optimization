import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'allegheny_county_911_EMS_dispatches.csv')

df = pd.read_csv(DATA_PATH)

# trying kmeans model
# scaling data
# dropping priority because thats the baseline we are trying to beat and replace
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X = df.drop('priority', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# doign pca for elbow curve
from sklearn.decomposition import PCA
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)

pca.explained_variance_ratio_

#elbow curve
pca_full = PCA().fit(X_scaled)
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Elbow Curve for PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# running 2D kmeans with 4 clusters

n_components = 2
n_clusters = 4

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans.fit(X_pca)
# dump(kmeans, save_file)
log_message("KMeans model trained and saved successfully.")

predicted_clusters = kmeans.labels_

# plotting
from matplotlib.colors import ListedColormap

predicted_clusters = kmeans.labels_

unique_clusters = np.unique(predicted_clusters)
cluster_colors = ListedColormap(plt.cm.tab10.colors[:len(unique_clusters)])

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predicted_clusters, cmap=cluster_colors, s=30, alpha=0.7)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors(i), markersize=10)
            for i in range(len(unique_clusters))]
plt.legend(handles, [f'Cluster {i}' for i in unique_clusters], title="Clusters", loc="upper right")

plt.title("KMeans Clusters (2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# running 3D kmeans with 4 clusters

n_components = 3
n_clusters = 4

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans.fit(X_pca)
log_message("KMeans model trained and saved successfully.")

predicted_clusters = kmeans.labels_

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=predicted_clusters, cmap='viridis', s=30, alpha=0.7)
fig.colorbar(scatter_3d, label='Cluster Label')
ax.set_title("KMeans Clusters (3D PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()

# plottign in 3d with plotly
import plotly.express as px

num_points_to_show = 10000

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Cluster'] = predicted_clusters

df_pca['Cluster'] = df_pca['Cluster'].astype(str)

fig = px.scatter_3d(
    df_pca.iloc[np.linspace(0, len(X_pca) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))
# fig.update_layout(
#     scene=dict(
#         xaxis_title="Principal Component 1",
#         yaxis_title="Principal Component 2",
#         zaxis_title="Principal Component 3"
#     )
# )


fig.update_layout(
    scene=dict(
        xaxis_title="",
        yaxis_title="",
        zaxis_title=""
    )
)

fig.show()

# plottign in 3d with plotly

# this is to save as hmtl and make gif which im doing on local
import plotly.express as px

num_points_to_show = 10000

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Cluster'] = predicted_clusters

df_pca['Cluster'] = df_pca['Cluster'].astype(str)

fig = px.scatter_3d(
    df_pca.iloc[np.linspace(0, len(X_pca) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))

fig.update_layout(
    title=None,  # Remove title
    showlegend=False,  # Remove legend
    scene=dict(
        xaxis=dict(title="", showticklabels=False),  # Remove x-axis label and tick numbers
        yaxis=dict(title="", showticklabels=False),  # Remove y-axis label and tick numbers
        zaxis=dict(title="", showticklabels=False),  # Remove z-axis label and tick numbers
    )
)

fig.show()

fig.write_html("3dplot.html", include_plotlyjs=True, full_html=True, div_id="plot")

# running 2D kmeans with 8 clusters

n_components = 2
n_clusters = 8

pca_8clust = PCA(n_components=n_components)
X_pca_8clust = pca.fit_transform(X_scaled)

kmeans_8clust = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans_8clust.fit(X_pca)
# dump(kmeans, save_file)
log_message("KMeans model trained and saved successfully.")

predicted_clusters_8clust = kmeans.labels_

# plotting
from matplotlib.colors import ListedColormap

unique_clusters_8clust = np.unique(predicted_clusters_8clust)
cluster_colors_8clust = ListedColormap(plt.cm.tab10.colors[:len(unique_clusters_8clust)])

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_8clust[:, 0], X_pca_8clust[:, 1], c=predicted_clusters_8clust, cmap=cluster_colors_8clust, s=30, alpha=0.7)

handles_8clust = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors(i), markersize=10)
            for i in range(len(unique_clusters_8clust))]
plt.legend(handles_8clust, [f'Cluster {i}' for i in unique_clusters_8clust], title="Clusters", loc="upper right")

plt.title("KMeans Clusters (2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# running 3D kmeans with 8 clusters

n_components = 3
n_clusters = 8

pca_8clust = PCA(n_components=n_components)
X_pca_8clust = pca.fit_transform(X_scaled)

kmeans_8clust = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans_8clust.fit(X_pca)
log_message("KMeans model trained and saved successfully.")

predicted_clusters_8clust = kmeans_8clust.labels_

# plots
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(X_pca_8clust[:, 0], X_pca_8clust[:, 1], X_pca_8clust[:, 2], c=predicted_clusters_8clust, cmap='viridis', s=30, alpha=0.7)
fig.colorbar(scatter_3d, label='Cluster Label')
ax.set_title("KMeans Clusters (3D PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()

# 3d plotting
num_points_to_show = 10000

df_pca_8clust = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca_8clust['Cluster (8)'] = predicted_clusters_8clust

df_pca_8clust['Cluster (8)'] = df_pca_8clust['Cluster (8)'].astype(str)

fig = px.scatter_3d(
    df_pca_8clust.iloc[np.linspace(0, len(X_pca_8clust) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster (8)',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster (8)': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3"
    )
)

fig.show()

fig.write_html("3dplot.html", include_plotlyjs=True, full_html=True, div_id="plot")

# ran GMM
# didn't provide any extra information
from sklearn.mixture import GaussianMixture
n_components = 3

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

gmm = GaussianMixture(n_components=n_components, random_state=1)
gmm.fit(X_scaled)

pred_lables = gmm.predict(X_scaled)

# adding the newfound cluster to the original data set
df["Cluster"] = df_pca['Cluster']

# df[df.Cluster == 1].head()



df_0.head(8)

df_1.head(8)

df_2.head(8)

df_3.head(8)

# gonna run new model without quarters...
X = df.drop(['priority', 'Q1', 'Q2', 'Q3', 'Q4', 'Cluster'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# running 3D kmeans with 4 clusters

n_components = 3
n_clusters = 4

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans.fit(X_pca)
log_message("KMeans model trained and saved successfully.")

predicted_clusters = kmeans.labels_

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=predicted_clusters, cmap='viridis', s=30, alpha=0.7)
fig.colorbar(scatter_3d, label='Cluster Label')
ax.set_title("KMeans Clusters (3D PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()

# 3d plotting with plotly

num_points_to_show = 10000

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Cluster'] = predicted_clusters

df_pca['Cluster'] = df_pca['Cluster'].astype(str)

fig = px.scatter_3d(
    df_pca.iloc[np.linspace(0, len(X_pca) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3"
    )
)

fig.show()

fig.write_html("3dplot.html", include_plotlyjs=True, full_html=True, div_id="plot")

# running 3D kmeans with 8 clusters and plotting in 3d

n_components = 3
n_clusters = 8

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans.fit(X_pca)
log_message("KMeans model trained and saved successfully.")

predicted_clusters = kmeans.labels_

num_points_to_show = 10000

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Cluster'] = predicted_clusters

df_pca['Cluster'] = df_pca['Cluster'].astype(str)

fig = px.scatter_3d(
    df_pca.iloc[np.linspace(0, len(X_pca) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3"
    )
)

fig.show()

# running 3D kmeans with 6 clusters and plotting in 3d

n_components = 3
n_clusters = 6

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans.fit(X_pca)
log_message("KMeans model trained and saved successfully.")

predicted_clusters = kmeans.labels_

num_points_to_show = 10000

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Cluster'] = predicted_clusters

df_pca['Cluster'] = df_pca['Cluster'].astype(str)

fig = px.scatter_3d(
    df_pca.iloc[np.linspace(0, len(X_pca) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3"
    )
)

fig.show()

## Tring with dif number of clusters.... we found 3 had the best results

# running 3D kmeans with 3 clusters and plotting in 3d

n_components = 3
n_clusters = 3

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
kmeans.fit(X_pca)
log_message("KMeans model trained and saved successfully.")

predicted_clusters = kmeans.labels_

num_points_to_show = 10000

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Cluster'] = predicted_clusters

df_pca['Cluster'] = df_pca['Cluster'].astype(str)

fig = px.scatter_3d(
    df_pca.iloc[np.linspace(0, len(X_pca) - 1, num_points_to_show, dtype=int)],
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    title="3D Visualization of KMeans Clusters (PCA)",
    labels={'Cluster': 'Cluster Label'},
    opacity=0.8
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3"
    )
)

fig.show()

# adding the cluster from the 3D, 3 cluster model to the original df
df["Cluster"] = df_pca['Cluster']

df.Cluster.nunique()

# grouping our data points by cluster
grouped = df.groupby('Cluster')

grouped.describe()

# plotting to try to see how each cluster is different
sns.boxplot(data=df, x='Cluster', y='call_year')
plt.title('Call Year Distribution by Cluster')
plt.show()

sns.countplot(data=df, x='priority', hue='Cluster')
plt.title('Priority Counts by Cluster')
plt.show()


# trying to find statistical significance

from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['priority'], df['Cluster'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test p-value for `priority`: {p}")

# trying to find statistical significance

from scipy.stats import f_oneway

anova_result = f_oneway(*[group['call_year'] for name, group in grouped])
print(f"ANOVA Test p-value for `call_year`: {anova_result.pvalue}")

# plotting to try to see how each cluster is different

sns.violinplot(data=df, x='Cluster', y='call_year', inner='quartile')
plt.title('Call Year Distribution by Cluster')
plt.show()

# plotting to try to see how each cluster is different

g = sns.FacetGrid(df, col='Cluster', sharex=True, sharey=True, height=4)
g.map(plt.hist, 'call_year', bins=10, color='skyblue', edgecolor='black')
g.set_axis_labels('Call Year', 'Frequency')
g.set_titles('Cluster {col_name}')
plt.show()

