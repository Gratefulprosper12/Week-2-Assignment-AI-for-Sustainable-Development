# 🌍 SDG 3: Good Health and Well-Being
# 🦠 Disease Outbreak Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 📥 Load Dataset
# Replace with the actual path to your dataset
df = pd.read_csv('disease_outbreak_data.csv')

# 📊 Basic Data Exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 🧹 Preprocessing
df.dropna(inplace=True)
features = ['temperature', 'humidity', 'rainfall', 'population_density']
target = 'outbreak'
X = df[features]
y = df[target]

# ✂️ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔍 Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔢 Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
