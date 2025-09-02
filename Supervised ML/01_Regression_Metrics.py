from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    r2_score
)
import numpy as np

# True values (actual output)
y_true = [12, 50, 22, 27]

# Predicted values
y_pred = [10, 46, 19, 26]

# Evaluation Metrics
mae = mean_absolute_error(y_true, y_pred)  # Average absolute errors
mse = mean_squared_error(y_true, y_pred)   # Squared errors
rmse = np.sqrt(mse)                        # Root of MSE
mape = mean_absolute_percentage_error(y_true, y_pred)  # Percentage error
r2 = r2_score(y_true, y_pred)              # Goodness of fit

# Display results
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R² Score:", r2)
