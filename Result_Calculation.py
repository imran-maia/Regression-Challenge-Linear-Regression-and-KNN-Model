import pandas as pd
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
data = pd.read_csv('0075729_Hossain_Actual_Pred_Result.csv')

# Extract the actual regression result and prediction results
actual = data['Actual_Y']
lr_prediction = data['LR_Y']
knn_prediction = data['KNN_Y']

# Calculate the mean squared error for LR prediction
lr_mse = mean_squared_error(actual, lr_prediction)

# Calculate the mean squared error for KNN prediction
knn_mse = mean_squared_error(actual, knn_prediction)

# Print the results
print('Mean Squared Error (LR Prediction):', lr_mse)
print('Mean Squared Error (KNN Prediction):', knn_mse)
