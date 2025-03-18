import pandas as pd 
from imblearn.over_sampling import SMOTE

train_path = './training_set_songs.csv'
train_set = pd.read_csv(train_path)
X_train = train_set.drop('2.1', axis=1); y_train = train_set['2.1']

# Resample training set with SMOTE (IMPORTANT CHANGE)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

train_set_resampled = X_train_resampled.copy()
train_set_resampled['2.1'] = y_train_resampled

old_distr = y_train.value_counts()
distribution = train_set_resampled['2.1'].value_counts()

print(old_distr)
print(distribution)

train_set_resampled.to_csv('training_set_resampled_songs.csv', index=False)
