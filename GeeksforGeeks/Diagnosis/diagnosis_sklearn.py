from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("wisconsin_breast_cancer_data.csv")

# 2. Clean data
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# 3. Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Build pipeline: scaling + logistic regression
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),                 # normalize features
    ("logreg", LogisticRegression(max_iter=10000))
])

# 6. Train
pipeline.fit(X_train, y_train)

# 7. Predict
y_pred = pipeline.predict(X_test)

# 8. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

#10 Probability Distribution Plot
results_df = X_test.copy()

# Add the actual outcomes, predicted outcomes, and predicted probabilities
results_df['y_actual'] = y_test
results_df['y_predicted'] = y_pred
results_df['y_probability_1'] = y_pred[:, 1]

# Using sns.kdeplot (Kernel Density Estimate)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=results_df, x='y_probability_1', hue='y_actual', fill=True)
plt.title('Distribution of Predicted Probabilities by Actual Class')
plt.xlabel('Predicted Probability of Class 1')
plt.ylabel('Density')
plt.legend(title='Actual Class', labels=['Class 1', 'Class 0']) # Adjust labels as needed
plt.show()

# Using sns.histplot (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=results_df, x='y_probability_1', hue='y_actual', multiple='layer', kde=True)
plt.title('Histogram of Predicted Probabilities by Actual Class')
plt.xlabel('Predicted Probability of Class 1')
plt.ylabel('Count')
plt.show()

# Pair Plot Comparison
# Pair plot for all features (can be heavy with 30+ features)
sns.pairplot(data, hue="diagnosis", 
             vars=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"],
             diag_kind="kde", palette=["#4daf4a", "#e41a1c"])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()
