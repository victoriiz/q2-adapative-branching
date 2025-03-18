from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

file_path = 'C:\\Users\\15840\\OneDrive\\Documents\\ML\\Q2 Project\\credit_risk.csv' 
data = pd.read_csv(file_path)

print(f"Information about the dataset: {data.info()}")
print(data.head())

print(data.describe())

# 1. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Income Distribution
plt.figure(figsize=(10, 6))
sns.histplot((data['Income']), kde=True, bins=30)
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Default', data=data)
plt.title('Loan Default Status')
plt.xlabel('Default')
plt.ylabel('Count')
plt.show()

#PREPROCESSING STUFF BELOWW
#drop id variable as it is likely irrelevant
data.drop('Id', axis=1, inplace=True)

#removing missing val
data['Emp_length'].fillna(data['Emp_length'].median(), inplace=True)
data['Rate'].fillna(data['Rate'].median(), inplace=True)

#encode categorical var
data = pd.get_dummies(data, columns=['Home', 'Intent'], drop_first=True)
data['Default'] = data['Default'].map({'Y': 1, 'N': 0})

#normalize income, amount, percent_income
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['Income', 'Amount', 'Percent_income']] = scaler.fit_transform(data[['Income', 'Amount', 'Percent_income']])

X = data.drop('Default', axis=1)
y = data['Default']

#save data
output_path = 'processed_credit_risk.csv'  
data.to_csv(output_path, index=False)

print(f"Preprocessed dataset saved to {output_path}")

#splitting and saving split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_set = X_train.copy()
train_set['Default'] = y_train
train_set.to_csv("training_set.csv", index=False)
print("\nTraining set saved as 'training_set.csv'.")
test_set = X_test.copy()
test_set['Default'] = y_test
test_set.to_csv("testing_set.csv", index=False)
print("\nTesting set saved as 'testing_set.csv'.")