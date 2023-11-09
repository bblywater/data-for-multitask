import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# File paths for the data sources
file_paths = ['serial_345/v25/exet.csv', 'serial_345/v25-2/exet.csv' , 'serial_345/v25-5/exet.csv']  # Add paths as needed

# Number of variables (assumed to be the same for each data source)
num_variables = 3  # Adjust this if your datasets have a different number of variables

horizontal_lines = [0.3, 0.4, 0.5]

# Number of variables (assumed to be the same for each data source)
num_variables = 3  # Adjust this if your datasets have a different number of variables

# Create subplots with an additional column for the average values (n+1 columns)
fig, axs = plt.subplots(num_variables, len(file_paths) + 1, figsize=(15, 15), sharey=True)

horizontal_lines = [0.3, 0.4, 0.5]

# Linear regression model
model = LinearRegression()

# Loop through each file path (for columns) and each subplot axis (for rows)
for row in range(num_variables):
    row_data = []  # List to store data for calculating the row average
    
    for col, file_path in enumerate(file_paths):
        ax = axs[row, col]
        data = pd.read_csv(file_path, header=None)
        # Remove outliers
        data = data.applymap(lambda x: 0 if x > 1 else x)
        
        # Filter out NaN values
        valid_data = data.iloc[:, row].dropna()
        row_data.append(valid_data)  # Append data for this variable
        X = np.array(valid_data.index).reshape(-1, 1)
        y = valid_data.values

        # Plot data
        ax.plot(valid_data.index, valid_data, label=f'Data source {col + 1}', linestyle='-')

        # Add horizontal line
        ax.axhline(y=horizontal_lines[row], color='r', linestyle='--', label=f'Horizontal line: {horizontal_lines[row]}')

        # Regression
        if len(valid_data) > 1:  # Need at least 2 points to fit a line
            model.fit(X, y)
            y_pred = model.predict(X)
            ax.plot(X, y_pred, color='green', label='Regression line')

        ax.legend()
        ax.set_ylabel('Value')
        ax.set_xlabel('Ordinal Number')

        # Set titles for the first row
        if row == 0:
            ax.set_title(f'v25_{col + 1}')  # Title for each data source

    # Calculate and plot the average data for the n+1st column
    avg_ax = axs[row, -1]  # The last column for each row
    average_values = pd.concat(row_data, axis=1).mean(axis=1)  # Calculate average across the datasets for this variable
    avg_ax.plot(average_values.index, average_values, label=f'Average', linestyle='-', color='blue')

    # Add the same horizontal line for the average plot
    avg_ax.axhline(y=horizontal_lines[row], color='r', linestyle='--', label=f'Horizontal line: {horizontal_lines[row]}')

    avg_ax.legend()
    avg_ax.set_xlabel('Ordinal Number')
    
    # Add a title for the average column
    if row == 0:
        avg_ax.set_title('Average')

plt.tight_layout()
plt.show()