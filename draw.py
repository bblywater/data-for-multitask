import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Assuming data.csv contains your data
# data = pd.read_csv('serial_345/v25/test_exet_x.csv', header=None)
data = pd.read_csv('serial_345/v25/exet.csv', header=None)
# Remove outliers
data = data.applymap(lambda x: 0 if x > 1 else x)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
horizontal_lines = [0.3, 0.4, 0.5]

# Linear regression model
model = LinearRegression()

# Plot each column
for i, ax in enumerate(axs):
    # Filter out NaN values
    valid_data = data[i].dropna()
    X = np.array(valid_data.index).reshape(-1, 1)
    y = valid_data.values
    
    # Plot data
    ax.plot(data[i].index, data[i], label=f'Column {i + 1}', linestyle='-')
  
    
    # Add horizontal line
    ax.axhline(y=horizontal_lines[i], color='r', linestyle='--', label=f'Horizontal line: {horizontal_lines[i]}')
    
    # Regression
    if len(valid_data) > 1:  # Need at least 2 points to fit a line
        model.fit(X, y)
        y_pred = model.predict(X)
        ax.plot(X, y_pred, color='green', label='Regression line')
    
    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('Ordinal Number')

plt.tight_layout()
plt.show()
