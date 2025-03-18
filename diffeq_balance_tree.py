import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import _tree
from scipy.integrate import solve_ivp
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

test_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\testing_set.csv' 
train_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\training_set.csv'
train_set = pd.read_csv(train_path); test_set = pd.read_csv(test_path)
X_train = train_set.drop('Default', axis=1); y_train = train_set['Default']
X_test = test_set.drop('Default',axis=1); y_test = test_set['Default']

# Resample training set with SMOTE (IMPORTANT CHANGE)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# custom gain function with class weighting
def compute_custom_gain(parent_impurity, left_impurity, right_impurity, left_weight, right_weight, penalty_factor, min_gain_threshold, class_weights):
    weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
    gain = parent_impurity - weighted_impurity
    penalty = max(0, penalty_factor * (min_gain_threshold - gain))
    
    # apply class weights
    gain *= class_weights[1] / class_weights[0]
    return gain - penalty

# custom cost func.
def cost_function(params, X_train, y_train, X_test, y_test, lambda_):
    penalty_factor, min_gain_threshold = params
    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, class_weight='balanced')
    tree.fit(X_train, y_train)
    
    prune_with_custom_gain(tree, penalty_factor, min_gain_threshold)
    
    # compute cost with f1 score
    y_pred = tree.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    complexity_penalty = penalty_factor * tree.tree_.node_count
    return -f1 + lambda_ * complexity_penalty

def prune_with_custom_gain(tree, penalty_factor, min_gain_threshold):
    tree_ = tree.tree_
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] == _tree.TREE_LEAF:
            continue
        left, right = tree_.children_left[node_id], tree_.children_right[node_id]
        left_weight = tree_.n_node_samples[left] / tree_.n_node_samples[node_id]
        right_weight = tree_.n_node_samples[right] / tree_.n_node_samples[node_id]
        gain = compute_custom_gain(
            tree_.impurity[node_id], 
            tree_.impurity[left], 
            tree_.impurity[right], 
            left_weight, 
            right_weight, 
            penalty_factor, 
            min_gain_threshold, 
            class_weights={0: len(y_train_resampled) / sum(y_train_resampled == 0), 
                           1: len(y_train_resampled) / sum(y_train_resampled == 1)}
        )
        if gain < 0:
            tree_.children_left[node_id] = _tree.TREE_LEAF
            tree_.children_right[node_id] = _tree.TREE_LEAF

# optimize diffeq
def optimize_tree(t, params, X_train, y_train, X_test, y_test, lambda_):
    grad = gradient(params, X_train, y_train, X_test, y_test, lambda_)
    return -grad

def gradient(params, X_train, y_train, X_test, y_test, lambda_):
    epsilon = 1e-5
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_eps = params.copy()
        params_eps[i] += epsilon
        grad[i] = (cost_function(params_eps, X_train, y_train, X_test, y_test, lambda_) -
                   cost_function(params, X_train, y_train, X_test, y_test, lambda_)) / epsilon
    return grad

# set some initial params
initial_params = [0.1, 0.01]  # [penalty_factor, min_gain_threshold]
lambda_ = 0.1
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# solve and find optimal params
solution = solve_ivp(optimize_tree, t_span, initial_params, t_eval=t_eval, args=(X_train_resampled, y_train_resampled, X_test, y_test, lambda_))
optimized_params = solution.y[:, -1]
best_penalty_factor, best_min_gain_threshold = optimized_params

# FINAL EVAL
final_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, class_weight='balanced')
final_tree.fit(X_train_resampled, y_train_resampled)
prune_with_custom_gain(final_tree, best_penalty_factor, best_min_gain_threshold)
y_pred = final_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Balanced Tree/Diffeq Model')
plt.show()

