from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df_all = pd.read_csv(r"continuous dataset.csv", parse_dates = ['datetime'], index_col = ['datetime'])
print(df_all.head())
print("There are %0.0f" %df_all.shape[0] + " repeated measures and %0.0f" %df_all.shape[1] +" variables in the dataset" )
df = df_all['nat_demand'].resample('D').sum()
     
df.shape
df.plot()
df_train = df[1:1500]
df_test = df[1500:2003] 

decomp = sm.tsa.seasonal_decompose(df_train,model = 'additive')
fig = decomp.plot()
fig.set_figwidth(20)

adf = adfuller(df_train)
print('adfuller test P-Value: ', adf[1])

fig, axes = plt.subplots(2, 2, figsize = (20,15))
axes[0,0].plot(df_train)
axes[0,0].set_title('Original Series')
plot_acf(df, ax=axes[0,1])
axes[1,0].plot(df_train.diff())
axes[1,0].set_title('First Differencing Series')
plot_acf(df_train.diff().dropna(), ax=axes[1,1])
plt.show()


fig, axes = plt.subplots(1, 2, figsize = (20,5))
axes[0].plot(df_train); axes[0].set_title('Original')
#axes[1].set(ylim=(0,5))
plot_pacf(df_train, ax=axes[1])

plt.show()


fig, axes = plt.subplots(1, 2, figsize = (20,5))
axes[0].plot(df_train); axes[0].set_title('Original')
plot_acf(df_train, ax=axes[1])

plt.show()

fig, axes = plt.subplots(1, 2, figsize = (20,5))
axes[0].plot(df_train, label= 'Original')
axes[0].plot(df_train.diff(1), label= 'Usual Differencing')             
axes[0].set_title('Trend Differencing')
axes[0].legend(loc='center left', fontsize=10)
axes[1].plot(df_train, label= 'Original')
axes[1].plot(df_train.diff(52), label= 'Seasonal Differencing')             
axes[1].set_title('Sesonal Differencing')
axes[1].legend(loc='center left', fontsize=10)
plt.show()

df_exo = df_all.resample('D').sum().iloc[:,1:]        # Adding exogenous variable into our ARIMA model 
exo_train = df_exo[1:1500]
exo_test = df_exo[1500:2003]
df_exo.shape
exo_test.shape
exo_train.shape

arimax= auto_arima(df_train,trace=True, X = exo_train
                   , error_action='ignore', test = 'adf', approximation=False
                   , start_p=0,start_q=0,max_p=10,max_q=10,m=1, D=0,      # D is the seasonal difference m is time step 
                   suppress_warnings=True,stepwise=True,seasonal=False)
arimax.summary()

arimax.plot_diagnostics(figsize=(20,10))
plt.show()
print(df_test)

pred_x = arimax.predict(n_periods = len(df_test),X = exo_test) # set number of periods
df_train.plot(legend = True,label = 'Train', figsize=(15,6)) 
df_test.plot(legend = True,label = 'Test')
pred_x.plot(legend = True,label = 'ARIMA_Pred')

print("The Root Mean Squared Error is: "+ str(round((np.sqrt(mean_squared_error(df_test,pred_x))/48049*100),2))+"%")

import joblib
joblib.dump(arimax,"arimax.cls")

