import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import _tree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

test_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\testing_set.csv' 
train_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\training_set.csv'
train_set = pd.read_csv(train_path); test_set = pd.read_csv(test_path)
X_train = train_set.drop('Default', axis=1); y_train = train_set['Default']
X_test = test_set.drop('Default', axis=1); y_test = test_set['Default']

def compute_custom_gain(parent_impurity, left_impurity, right_impurity, left_weight, right_weight, penalty_factor, min_gain_threshold):
    weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
    gain = parent_impurity - weighted_impurity
    penalty = max(0, penalty_factor * (min_gain_threshold - gain))
    return gain - penalty

def fitness_function(X_train, y_train, X_test, y_test, penalty_factor, min_gain_threshold, lam):
    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42)
    tree.fit(X_train, y_train)
    prune_with_gain(tree, penalty_factor, min_gain_threshold)
    #gain = acc
    accuracy = accuracy_score(y_test, tree.predict(X_test))
    complexity_penalty = penalty_factor * tree.tree_.node_count
    obj = accuracy-lam*complexity_penalty #lagrangian objective ohhhh
    return obj

def prune_with_gain(tree, penalty_factor, min_gain_threshold):
    tree_ = tree.tree_
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] == _tree.TREE_LEAF:
            continue
        left, right = tree_.children_left[node_id], tree_.children_right[node_id]
        left_weight = tree_.n_node_samples[left] / tree_.n_node_samples[node_id]
        right_weight = tree_.n_node_samples[right] / tree_.n_node_samples[node_id]
        gain = compute_custom_gain(tree_.impurity[node_id], tree_.impurity[left], tree_.impurity[right], left_weight, right_weight, penalty_factor, min_gain_threshold)
        if gain < 0:
            tree_.children_left[node_id] = _tree.TREE_LEAF
            tree_.children_right[node_id] = _tree.TREE_LEAF
            
def sim_annealing(X_train, y_train, X_test, y_test, initial_penalty_factor=0.1, initial_min_gain_threshold=0.01, lambda_=0.01, max_iterations=100, initial_temp=1.0, temp_decay=0.30):
    current_penalty_factor = initial_penalty_factor
    current_min_gain_threshold = initial_min_gain_threshold
    best_penalty_factor = current_penalty_factor
    best_min_gain_threshold = current_min_gain_threshold
    current_fitness = fitness_function(X_train, y_train, X_test, y_test, current_penalty_factor, current_min_gain_threshold, lambda_)
    best_fitness = current_fitness
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        #pertubation
        new_penalty_factor = current_penalty_factor + np.random.uniform(-0.05, 0.05)
        new_min_gain_threshold = current_min_gain_threshold + np.random.uniform(-0.005, 0.005)
        
        #enforce bounds + recalc fitness
        new_penalty_factor = np.clip(new_penalty_factor, 0.05, 0.3)
        new_min_gain_threshold = np.clip(new_min_gain_threshold, 0.005, 0.05)
        new_fitness = fitness_function(X_train, y_train, X_test, y_test, new_penalty_factor, new_min_gain_threshold, lambda_)
        
        #accept new solution based on simulated annealing acceptance prob
        if new_fitness > current_fitness or np.random.rand() < np.exp((new_fitness - current_fitness) / temperature):
            current_penalty_factor = new_penalty_factor
            current_min_gain_threshold = new_min_gain_threshold
            current_fitness = new_fitness
        #update
        if current_fitness > best_fitness:
            best_penalty_factor = current_penalty_factor
            best_min_gain_threshold = current_min_gain_threshold
            best_fitness = current_fitness
        #cool temp
        temperature *= temp_decay
    
    return best_penalty_factor, best_min_gain_threshold

#run SA with best parameters
best_penalty_factor, best_min_gain_threshold = sim_annealing(X_train, y_train, X_test, y_test)

#build final tree with optimal parameters
final_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5)
final_tree.fit(X_train, y_train)
prune_with_gain(final_tree, best_penalty_factor, best_min_gain_threshold)

y_pred = final_tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, final_tree.predict_proba(X_test)[:,1])

print(f"Final Model Performance with Optimized Parameters:")
print(f"Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Simulated Annealing/Tree Model')
plt.show()
