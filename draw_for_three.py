import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

file_paths = ['serial_345/v25/exet.csv', 'serial_345/v25-2/exet.csv' , 'serial_345/v25-5/exet.csv'] 

horizontal_lines = [0.3, 0.4, 0.5]

num_variables = 3  

fig, axs = plt.subplots(num_variables, len(file_paths) + 1, figsize=(15, 15), sharey=True)

horizontal_lines = [0.3, 0.4, 0.5]

model = LinearRegression()

for row in range(num_variables):
    row_data = [] 
    
    for col, file_path in enumerate(file_paths):
        ax = axs[row, col]
        data = pd.read_csv(file_path, header=None)

        data = data.applymap(lambda x: 0 if x > 1 else x)
        
        valid_data = data.iloc[:, row].dropna()
        row_data.append(valid_data)
        X = np.array(valid_data.index).reshape(-1, 1)
        y = valid_data.values

        ax.plot(valid_data.index, valid_data, label=f'Data source {col + 1}', linestyle='-')

        ax.axhline(y=horizontal_lines[row], color='r', linestyle='--', label=f'Horizontal line: {horizontal_lines[row]}')

        if len(valid_data) > 1:
            model.fit(X, y)
            y_pred = model.predict(X)
            ax.plot(X, y_pred, color='green', label='Regression line')

        ax.legend()
        ax.set_ylabel('Value')
        ax.set_xlabel('Ordinal Number')

        if row == 0:
            ax.set_title(f'v25_{col + 1}')

    avg_ax = axs[row, -1]
    average_values = pd.concat(row_data, axis=1).mean(axis=1)
    avg_ax.plot(average_values.index, average_values, label=f'Average', linestyle='-', color='blue')

    avg_ax.axhline(y=horizontal_lines[row], color='r', linestyle='--', label=f'Horizontal line: {horizontal_lines[row]}')

    avg_ax.legend()
    avg_ax.set_xlabel('Ordinal Number')
    
    if row == 0:
        avg_ax.set_title('Average')

plt.tight_layout()
plt.show()