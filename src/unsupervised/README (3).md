# Unsupervised Models

This folder contains all unsupervised learning models used in the EMS Dispatch Optimization project.  
Each script loads the dataset automatically from the `data` folder using dynamic relative paths.

## Models & Functions Included

### 1. **K-Means Clustering**
- Clusters EMS dispatches into groups for pattern discovery.
- Uses scaling and PCA preprocessing.
- Provides cluster assignments and interpretation plots.

### 2. **Principal Component Analysis (PCA)**
- Reduces high-dimensional features into principal components.
- Visualizes data structure prior to clustering.

### 3. **Gaussian Mixture Models (GMM)**
- Fits probabilistic clusters to EMS data.
- Compares cluster quality against K-Means.

### 4. **Visualization Functions**
Includes:
- Cluster scatterplots
- PCA component heatmaps
- Cluster distribution charts

### 5. **Statistical Testing**
May include:
- Comparison of clusters across categorical features
- Distribution analysis across clusters

## How It Works

Every model file begins with dynamic dataset loading:
```python
file_id = "1un4xuLvenGq7lYbL2ASSa-pjG1FYsTO8"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

df = pd.read_csv(url)
```

This ensures the script can run from any machine or directory structure.

Run using:
```
python unsupervised_models.py
```
