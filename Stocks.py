import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import zscore

dataset = pd.read_csv("datasæt/stocks/IBM_2006-01-01_to_2018-01-01.csv")
dataset["Date"] = pd.to_datetime(dataset["Date"])
daily_changes = dataset[["Open", "Close", "High", "Low"]].diff()
dataset = dataset.iloc[1:]
daily_changes = daily_changes.iloc[1:]
#daily_changes.drop(dataset[dataset['Date'] > pd.Timestamp("2007-01-22")].index, inplace=True)
#dataset.drop(dataset[dataset['Date'] > pd.Timestamp("2007-01-22")].index, inplace=True)
'''

# Apply Z-score for each numeric column
z_scores = np.abs(zscore(daily_changes[["Open", "Close", "High", "Low"]]))
threshold = 3  # standard: anything >3 std devs from mean

outliers = (z_scores > threshold)
dataset_outliers = daily_changes[outliers.any(axis=1)]

print(dataset_outliers)
'''

daily_changes_moving_avg = daily_changes.rolling(window=30).mean()

fig : Figure
ax : dict[tuple[int, int], Axes]
fig, ax = plt.subplots(4, 2)
ax[0, 0].plot(dataset["Date"], daily_changes["Open"], color="blue")
ax[0, 0].set_xlabel('date')
ax[0, 0].set_ylabel('Open value')
ax[0, 0].grid(True)

ax[1, 0].plot(dataset["Date"], daily_changes["Close"], color="orange")
ax[1, 0].set_xlabel('date')
ax[1, 0].set_ylabel('Close value')
ax[1, 0].grid(True)

ax[2, 0].plot(dataset["Date"], daily_changes["High"], color="green")
ax[2, 0].set_xlabel('date')
ax[2, 0].set_ylabel('High value')
ax[2, 0].grid(True)

ax[3, 0].plot(dataset["Date"], daily_changes["Low"], color="red")
ax[3, 0].set_xlabel('date')
ax[3, 0].set_ylabel('Low value')
ax[3, 0].grid(True)

#plot after
ax[0, 1].plot(dataset["Date"], daily_changes_moving_avg["Open"], color="blue")
ax[0, 1].set_xlabel('date')
ax[0, 1].set_ylabel('Open value')
ax[0, 1].grid(True)

ax[1, 1].plot(dataset["Date"], daily_changes_moving_avg["Close"], color="orange")
ax[1, 1].set_xlabel('date')
ax[1, 1].set_ylabel('Close value')
ax[1, 1].grid(True)

ax[2, 1].plot(dataset["Date"], daily_changes_moving_avg["High"], color="green")
ax[2, 1].set_xlabel('date')
ax[2, 1].set_ylabel('High value')
ax[2, 1].grid(True)

ax[3, 1].plot(dataset["Date"], daily_changes_moving_avg["Low"], color="red")
ax[3, 1].set_xlabel('date')
ax[3, 1].set_ylabel('Low value')
ax[3, 1].grid(True)

fig.autofmt_xdate()

plt.show()