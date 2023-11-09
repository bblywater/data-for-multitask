import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# data = pd.read_csv('serial_345/v25/test_exet_x.csv', header=None)
data = pd.read_csv('serial_345/v25/exet.csv', header=None)
data = data.applymap(lambda x: 0 if x > 1 else x)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
horizontal_lines = [0.3, 0.4, 0.5]

model = LinearRegression()

for i, ax in enumerate(axs):
    valid_data = data[i].dropna()
    X = np.array(valid_data.index).reshape(-1, 1)
    y = valid_data.values
    
    ax.plot(data[i].index, data[i], label=f'Column {i + 1}', linestyle='-')
  
    
    ax.axhline(y=horizontal_lines[i], color='r', linestyle='--', label=f'Horizontal line: {horizontal_lines[i]}')
    
    if len(valid_data) > 1:
        model.fit(X, y)
        y_pred = model.predict(X)
        ax.plot(X, y_pred, color='green', label='Regression line')
    
    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('Ordinal Number')

plt.tight_layout()
plt.show()
