import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score,confusion_matrix
from lightgbm import LGBMClassifier
import joblib
import logging

# Set the logging level for Hyperopt
logging.getLogger('hyperopt').setLevel(logging.WARNING)  # You can also set it to INFO or DEBUG

# Define test size and random seed for reproducibility
test_size = 0.4  # This means the data is split into (1-test_size:test_size) for training and validation
seed = 13254

# Load data from a CSV file
data01 = pd.read_csv(r'C:\Users\Sengoku\GuanzhongPlain\PointData.csv', encoding='gbk')



filename_import  = r'C:\Users\Sengoku\GuanzhongPlain\PointDataForTheWholeRegion.csv'

data_all = pd.read_csv(filename_import)
data_all.columns = ['Nighttime Lights', 'GDP', 'Ten Years Change of NDVI', 'Population',
                 'Degree of Urbanization', 'Impact of the Vadose Zone', 'Depth to Groundwater', 'Slope',
                 'Aquifer Water Yield Capacity', 'Net Recharge', 'Conductivity', 'LULC', 'PPSD','X','Y']




model_path = r'C:\Users\Sengoku\GuanzhongPlain\model\seed{}_{}.model'.format(seed, test_size)

filename_export = r'C:\Users\Sengoku\GuanzhongPlain\TheWholeRegionPointResults.csv'


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


# Load the pre-trained LightGBM model
light_model = joblib.load(model_path)

# Print the best AUC score from the validation data
best_score = light_model.best_score_['valid_1']['auc']
print("*******best_score*******")
print(best_score)

# Prepare data for prediction
# Drop irrelevant columns and align with training feature columns
X_all = data_all.drop(columns=[ "pointed", "X", "Y"], axis=1)
header = X_all.columns.tolist()

# Extract latitude and longitude for output
latitude_and_longitude = data_all.drop(columns=header, axis=1)

# Reorder X_all columns to match the order of train_X columns
X_all = X_all.reindex(columns=train_X.columns)

# Predict probabilities for the target variable
Y_all = light_model.predict_proba(X_all)[:, 1]

# Convert predictions to DataFrame format for easier output handling
Y_all = pd.DataFrame(Y_all, columns=['target'])

# Concatenate latitude/longitude with predictions
merge_XY = pd.concat([latitude_and_longitude, X_all, Y_all], axis=1)

# Select only the X, Y, and target columns for final output
merge_XY = pd.concat([merge_XY.X, merge_XY.Y, merge_XY.target], axis=1)

# Export results to a CSV file without the index
merge_XY.to_csv(filename_export, index=False)

print('DOWN')
