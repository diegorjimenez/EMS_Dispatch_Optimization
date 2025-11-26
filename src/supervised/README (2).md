# Supervised Models

This folder contains all supervised machine learning models used in the EMS Dispatch Optimization project.  
Each script loads the dataset automatically from the `data` folder using dynamic relative paths.

## Models & Functions Included

### 1. **Train/Test Split**
- Splits the EMS dataset into training and testing subsets.
- Handles features and target encoding.

### 2. **K-Nearest Neighbors (KNN)**
- Fits a KNN classifier on the dataset.
- Predicts EMS priority level.
- Includes accuracy reporting and confusion matrix output.

### 3. **Random Forest Classifier**
- Trains a RandomForest model to classify EMS priority.
- Provides feature importance rankings.
- Outputs model performance metrics.

### 4. **XGBoost Classifier**
- Executes an XGBoost model for classification.
- Includes label encoding and parameter tuning.
- Saves and loads models using `joblib`.

### 5. **Evaluation Functions**
Includes:
- `accuracy_score`
- `classification_report`
- `confusion_matrix`
- Visualization utilities for results

## How It Works

Every model file begins by loading:
```python
file_id = "1un4xuLvenGq7lYbL2ASSa-pjG1FYsTO8"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

df = pd.read_csv(url)
```

This ensures the code works anywhere the project is cloned.

Run using:
```
python supervised_models.py
```
