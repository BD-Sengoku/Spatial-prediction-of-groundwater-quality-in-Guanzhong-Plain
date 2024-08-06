import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier
import hyperopt
from hyperopt import hp
import joblib
import logging

# Set the logging level for Hyperopt
logging.getLogger('hyperopt').setLevel(logging.WARNING)  # You can also set it to INFO or DEBUG

# Define test size and random seed for reproducibility
test_size = 0.4  # This means the data is split into (1-test_size:test_size) for training and validation
seed = 13254

# Load data from a CSV file
data01 = pd.read_csv(r'C:\Users\Sengoku\GuanzhongPlain\PointData.csv', encoding='gbk')
# Set paths for saving results and model
filename = r'C:\Users\Sengoku\GuanzhongPlain\06-09\seed{}_{}.txt'.format(seed, test_size)
model_path = r'C:\Users\Sengoku\GuanzhongPlain\model\seed{}_{}.model'.format(seed, test_size)
max_evals = 1000  # Maximum evaluations for hyperparameter optimization

# Get the column headers
header = data01.columns.tolist()

# Split data into two DataFrames based on the 'target' value
data_0 = data01.loc[data01['target'] == 0]
data_1 = data01.loc[data01['target'] == 1]

# Prepare data with target = 0
data_0_X = data_0.drop(columns=["target"], axis=1)
data_0_Y = data_0.target
# Split into train and validation sets
train_0_X, valid_0_X, train_0_y, valid_0_y = train_test_split(data_0_X, data_0_Y, test_size=test_size, random_state=seed)

# Save train and validation data for target = 0
save_TrainDate_0 = pd.DataFrame(np.column_stack([train_0_X, train_0_y]), columns=header)
save_ValidDate_0 = pd.DataFrame(np.column_stack([valid_0_X, valid_0_y]), columns=header)

# Prepare data with target = 1
data_1_X = data_1.drop(columns=["target"], axis=1)
data_1_Y = data_1.target
# Split into train and validation sets
train_1_X, valid_1_X, train_1_y, valid_1_y = train_test_split(data_1_X, data_1_Y, test_size=test_size, random_state=seed)

# Save train and validation data for target = 1
save_TrainDate_1 = pd.DataFrame(np.column_stack([train_1_X, train_1_y]), columns=header)
save_ValidDate_1 = pd.DataFrame(np.column_stack([valid_1_X, valid_1_y]), columns=header)

# Combine and shuffle the training sets
train_date = pd.concat([save_TrainDate_0, save_TrainDate_1])
train_date = train_date.sample(frac=1, random_state=42)

# Combine and shuffle the validation sets
valid_date = pd.concat([save_ValidDate_0, save_ValidDate_1])
valid_date = valid_date.sample(frac=1, random_state=42)

# Extract features and targets from training data
train_y = train_date.target
train_X = train_date.drop(columns=["target"], axis=1)

# Extract features and targets from validation data
valid_y = valid_date.target
valid_X = valid_date.drop(columns=["target"], axis=1)

# Define a cross-validation function
def cross_validation(model_params):
    gbm = LGBMClassifier(**model_params)
    gbm.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], verbose=0)
    best_score = gbm.best_score_['valid_1']['auc']
    return 1 - best_score

# Objective function for Hyperopt
def hyperopt_objective(params):
    cur_param = {
        'objective': 'binary',
        'boosting_type': params['boosting_type'],
        'metric': 'auc',
        'num_leaves': params['num_leaves'],
        'learning_rate': params['learning_rate'],
        'early_stopping_rounds': 10,
        'bagging_freq': params['bagging_freq'],
        'bagging_fraction': params['bagging_fraction'],
        'feature_fraction': params['feature_fraction']
    }
    print("*" * 30)
    res = cross_validation(cur_param)
    print(params)
    print("Current best 1-auc score is:  {}, auc score is: {}".format(res, 1 - res))
    return res  # Hyperopt minimizes this value

# Define the search space for hyperparameters
params_space = {
    'objective': 'binary',
    "boosting_type": hp.choice("boosting_type", ['gbdt', 'dart', 'rf']),
    'metric': 'auc',
    "num_leaves": hp.choice("num_leaves", range(15, 128)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'bagging_freq': hp.choice("bagging_freq", range(4, 7)),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 0.9),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.9)
}

trials = hyperopt.Trials()

import warnings
warnings.filterwarnings("ignore")

# Run the hyperparameter optimization
best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=max_evals,
    trials=trials)

print("Best parameters")
print(best)

# Evaluate the best parameters
best_params = hyperopt.space_eval(params_space, best)
# Add additional parameters
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['num_iterations'] = 500
best_params['early_stopping_rounds'] = 200

# Train the model using the best parameters
light_model = LGBMClassifier(**best_params)
light_model.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)])

# Save the model to a file
joblib.dump(light_model, model_path)
print("Model saved:", model_path)

# Predict probabilities
y_pred1 = light_model.predict_proba(valid_X)[:, 1]

# Calculate AUC score
auc1 = roc_auc_score(valid_y, y_pred1)
# Convert probabilities to binary predictions
y_pred1 = (y_pred1 >= 0.5) * 1

# Calculate confusion matrix
a = confusion_matrix(valid_y, y_pred1)
a = a.tolist()
a0 = str(a[0])
a1 = str(a[1])

# Calculate various evaluation metrics
Precesion = str('Precesion: %.4f' % metrics.precision_score(valid_y, y_pred1))
Recall = str('Recall: %.4f' % metrics.recall_score(valid_y, y_pred1))
F1_score = str('F1-score: %.4f' % metrics.f1_score(valid_y, y_pred1))
Accuracy = str('Accuracy: %.4f' % metrics.accuracy_score(valid_y, y_pred1))
AUC = str('AUC: %.4f' % auc1)
AP = str('AP: %.4f' % metrics.average_precision_score(valid_y, y_pred1))
Log_loss = str('Log_loss: %.4f' % metrics.log_loss(valid_y, y_pred1, eps=1e-15, normalize=True, sample_weight=None, labels=None))
kappa_score = str('Kappa_score: %.4f' % metrics.cohen_kappa_score(valid_y, y_pred1))
confusion_matrix = f'{a0}\n{a1}\n'
metrics = f'{AUC}\n{Precesion}\n{Recall}\n{F1_score}\n{Accuracy}\n{AP}\n{Log_loss}\n{kappa_score}\n'

# Feature importance analysis
my_dict = dict(zip(train_X.columns, light_model.feature_importances_))
# Sort the dictionary by value in descending order
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
# Calculate the total sum of values
total = sum(sorted_dict.values())
# Calculate the percentage for each value and save to a new dictionary
dict1 = {key: (value / total) * 100 for key, value in sorted_dict.items()}

# Write results to a file
with open(filename, 'w') as f:
    # Write the confusion matrix
    f.write('---------Confusion Matrix---------\n')
    f.write(confusion_matrix)

    # Write evaluation metrics
    f.write('--------Evaluation Metrics-----------\n')
    f.write(metrics)
    # Write feature importance
    f.write('-------Importance------------\n')

    for key, value in dict1.items():
        f.write(f'{key}: {value:.2f}\n')
    # Write the best parameters for future reference
    f.write('----------Best Parameters-----------\n')
    f.write(str(best_params))
    f.write('\n')
    seed_str = f'seed = {seed}'
    f.write('----------Seed-----------\n')
    f.write(seed_str)
