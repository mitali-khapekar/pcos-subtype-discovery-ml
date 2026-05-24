import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

# Load dataset
df = pd.read_csv("pcos_patient_records.csv")

print("Dataset shape:", df.shape)

# =========================
# CREATE FOLLICLE FEATURE
# =========================
df['Follicle_Count'] = df['Follicle No. (L)'] + df['Follicle No. (R)']

# =========================
# SELECT FEATURES
# =========================
features = [
    ' Age (yrs)',
    'BMI',
    'Weight (Kg)',
    'AMH(ng/mL)',
    'TSH (mIU/L)',
    'Cycle length(days)',
    'Follicle_Count'
]

X = df[features]

# =========================
# 🔥 FIX: CONVERT TO NUMERIC
# =========================
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# =========================
# REMOVE BAD VALUES
# =========================
X = X.replace([np.inf, -np.inf], np.nan)

# =========================
# HANDLE MISSING VALUES
# =========================
X = X.fillna(X.median())

print("Data cleaned successfully")

# =========================
# STANDARDIZATION
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# PCA
# =========================
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

print("PCA applied")

# =========================
# KMEANS
# =========================
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

print("Clustering done")

# =========================
# CLASSIFIER
# =========================
clf = RandomForestClassifier(random_state=42)
clf.fit(X_pca, clusters)

print("Model trained")

# =========================
# SAVE MODEL
# =========================
model_data = {
    "scaler": scaler,
    "pca": pca,
    "model": clf
}

pickle.dump(model_data, open("model.pkl", "wb"))

print("✅ Model saved successfully!")

print("\nCluster Summary:")
print(pd.DataFrame(X).groupby(clusters).mean())
