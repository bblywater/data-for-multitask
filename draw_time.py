import pandas as pd
import matplotlib.pyplot as plt

data_path = 'serial_345/v25/10_20_2.csv'
data = pd.read_csv(data_path, header=None)

data.columns = ["REWARD", "TIME IN 0", "TIME IN 1"]

window_size_15 = 1

data_smoothed = data.rolling(window=window_size_15, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(14, 6))


print(data_smoothed["REWARD"])

ax1.plot(data_smoothed["REWARD"], label='Smoothed REWARD', color='blue')
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('Smoothed REWARD', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Smoothed Sequential Changes in REWARD, TIME IN 0, and TIME IN 1')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(data_smoothed["TIME IN 0"], label='Smoothed TIME IN 0', color='green', linestyle='--')
ax2.plot(data_smoothed["TIME IN 1"], label='Smoothed TIME IN 1', color='orange', linestyle='--')
ax2.set_ylabel('Smoothed TIME IN 0 and TIME IN 1', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
###################