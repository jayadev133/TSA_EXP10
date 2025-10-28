# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 28/10/2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
data = pd.read_excel('/content/Coffe_sales.xlsx')
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.dropna(subset=['date'], inplace=True)
data = data.sort_values(by='date')
daily_data = data.groupby('date')['money'].sum().reset_index()
daily_data.set_index('date', inplace=True)
print(daily_data.head())

plt.plot(daily_data.index, daily_data['money'])
plt.xlabel('Date')
plt.ylabel('money')
plt.title('COFFEE SALES Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


plot_acf(daily_data['money'].dropna())
plt.title("Autocorrelation Function (ACF) for Money")
plt.show()

plot_pacf(daily_data['money'].dropna())
plt.title("Partial Autocorrelation Function (PACF) for Money")
plt.show()


# === Train-Test Split ===
train_size = int(len(daily_data) * 0.8)
train, test = daily_data['money'][:train_size], daily_data['money'][train_size:]

# === Build and fit SARIMA model ===
# Fix: Correct syntax, use daily_data['money'], and specify seasonal_order
sarima_model = SARIMAX(train, order=(1, 1, 1))
sarima_result = sarima_model.fit(disp=False) # disp=False to reduce output during fitting

# === Forecast ===
# Fix: Correct the end index for prediction
predictions = sarima_result.predict(start=len(train), end=len(daily_data) - 1, dynamic=False)


# === Evaluate performance ===
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# === Plot predictions vs actuals ===
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Money') # Fix: Change ylabel to Money
plt.title('SARIMA Model Predictions for Money') # Fix: Change title
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:
<img width="169" height="133" alt="image" src="https://github.com/user-attachments/assets/820a3876-15c6-44e9-8030-ecd51c75b47c" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/b84ce44a-e79a-4e46-bb83-d19c3786770d" />

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/f6b73226-890f-4b29-a4ec-316d7d047d9c" />
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/981a4519-649b-43b7-a153-e226ee86aadc" />
<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/7c9ee6f6-31db-4c47-b6a6-505d6e66286f" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
