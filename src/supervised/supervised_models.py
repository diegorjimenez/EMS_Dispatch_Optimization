import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'allegheny_county_911_EMS_dispatches.csv')

df = pd.read_csv(DATA_PATH)

# splitting data for regression
from sklearn.model_selection import train_test_split

# Trying without descriptions and city code first
X = df.drop(['priority'], axis=1)
y = df['priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# using knn
from sklearn.neighbors import KNeighborsClassifier

num_groups = df.priority.nunique()

knn = KNeighborsClassifier(n_neighbors = num_groups)
knn.fit(X_train, y_train)

# getting predictions
y_pred = knn.predict(X_test) # if perfect y_pred = y_test

# analyzing model results
from sklearn.metrics import classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# analyzing model results
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

log_file_path = "training_log.txt"
def log_message(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{message}\n")
    print(message)

# Lets us cache models so we don't have to rerun all the time
# note, none of the models took more than 2 min so I didn't end up using this
from joblib import load, dump
import traceback

def load_and_test_model(model_path, X_test, y_test, model_name="Model"):
    try:
        model = load(model_path)
        print(f"{model_name} loaded successfully.")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_name}: {accuracy:.2f}")

        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


    except Exception as e:
        print(f"Error loading/testing {model_name}: {e}")
        print(traceback.format_exc())

# running XGBoost model
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

model_name = 'xgboost'
save_file = f'{model_name}_model.joblib'

try:

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=1, stratify=y_encoded)

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=1
    )
    xgb_model.fit(X_train, y_train)
    dump(xgb_model, save_file)
    log_message("XGBoost model trained and saved successfully.")

    load_and_test_model(save_file, X_test, y_test, model_name=model_name)
except Exception as e:
    log_message(f"Error training XGBoost model: {e}")
    log_message(traceback.format_exc())



# checking feature importance for xgb
model = load('xgboost_model.joblib')
xgb.plot_importance(model)
plt.show()

# trying RF model ... I rerun the train test split because I like to see it here
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
model_name = 'Random Forest'
print(f"Accuracy for {model_name}: {accuracy:.2f}")
print(f"Classification Report for {model_name}:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


importances = model.feature_importances_
feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

