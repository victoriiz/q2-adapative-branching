from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_path = './testing_set.csv' 
train_path = './training_set_resampled.csv'
train_set = pd.read_csv(train_path); test_set = pd.read_csv(test_path)
X_train = train_set.drop('Default', axis=1); y_train = train_set['Default']
X_test = test_set.drop('Default',axis=1); y_test = test_set['Default']

tree = RandomForestClassifier(n_estimators = 1000, random_state = 42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:,1])

# Print results
print(f"Final Model Performance with Optimized Parameters:")
print(f"Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()