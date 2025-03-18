from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\testing_set.csv' 
train_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\training_set.csv'
train_set = pd.read_csv(train_path); test_set = pd.read_csv(test_path)
X_train = train_set.drop('Default', axis=1); y_train = train_set['Default']
X_test = test_set.drop('Default',axis=1); y_test = test_set['Default']

#custom gain calculation with penalty (second one uses impurity directlym with penalty for low gain)
'''
def compute_custom_gain(parent_entropy, left_probs, right_probs, penalty_factor=0.1, min_gain_threshold=0.02):
    left_entropy = -np.sum(left_probs * np.log2(left_probs + 1e-9))
    right_entropy = -np.sum(right_probs * np.log2(right_probs + 1e-9))
    weighted_entropy = (len(left_probs) * left_entropy + len(right_probs) * right_entropy) / (len(left_probs) + len(right_probs))
    gain = parent_entropy - weighted_entropy
    return gain - max(0, penalty_factor * (min_gain_threshold - gain))
'''
def compute_custom_gain(parent_impurity, left_impurity, right_impurity, left_weight, right_weight, penalty_factor=0.1, min_gain_threshold=0.02):
    weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
    gain = parent_impurity - weighted_impurity
    penalty = max(0, penalty_factor * (min_gain_threshold - gain))
    return gain - penalty

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)

#pruning based on gain constraints
def prune_tree(tree, penalty_factor=0.1, min_gain_threshold=0.02):
    tree_ = tree.tree_
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] == _tree.TREE_LEAF:
            continue
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        
        gain = compute_custom_gain(tree_.impurity[node_id], 
                                   np.array(tree_.impurity[tree_.children_left[node_id]]), 
                                   np.array(tree_.impurity[tree_.children_right[node_id]]),
                                   penalty_factor, min_gain_threshold)
        if gain < 0:  #prune
            tree_.children_left[node_id] = _tree.TREE_LEAF
            tree_.children_right[node_id] = _tree.TREE_LEAF

prune_tree(tree, penalty_factor=0.1, min_gain_threshold=0.02)
print('done')

y_pred_pruned = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_pruned)
f1 = f1_score(y_test, y_pred_pruned)
precision = precision_score(y_test, y_pred_pruned)
recall = recall_score(y_test, y_pred_pruned)
roc_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:,1])

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

cm = confusion_matrix(y_test, y_pred_pruned)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
