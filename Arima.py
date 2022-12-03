import numpy as np
import pandas as pd
#from pandas import DataFrame, datetime, read_csv

df = pd.read_csv('ARIMA-dataset.csv')
df.dropna(inplace = True)
to_drop = ['lat','lon']
df.drop(to_drop, inplace=True, axis=1)

# df.info()
# df.plot()

# df = np.log(df)
# df.plot()

msk = (df.index < len(df)-30)#اکنون بیایید سری های زمانی را نیز به مجموعه های آموزشی و آزمایشی تقسیم کنیم. ما 30 نقطه داده آخر را به عنوان مجموعه تست و داده های قبلی را به عنوان مجموعه آموزشی تنظیم می کنیم
df_train = df[msk].copy()
df_test = df[~msk].copy()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf#برای رسمACF,PACF
acf_original = plot_acf(df_train)
pacf_original = plot_pacf(df_train)
#>>از تکه کد بالا نتیجه میگیریم که سری زمانی ما ثابت نیست


df_train_diff = df_train.diff().dropna()
df_train_diff.plot()
acf_diff = plot_acf(df_train_diff)
pacf_diff = plot_pacf(df_train_diff)#با استفاده از تکه کد زیر میبینیم که  سری زمانی ما ثابت میشود یعنی ثابتش میکنیم

#اکنون زمان تصمیم گیری در مورد دو پارامتر دیگر 
# p و q 
# است. آنها مدل هایی را که ما از ARIMA
#استفاده خواهیم کرد تعیین می کنند



#مدل آریمایی با 
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())


# قبل از استفاده از این مدل برای پیش‌بینی سری‌های زمانی، باید مطمئن شویم 
# که مدل ما اطلاعات کافی را از داده‌ها گرفته است. ما می توانیم این را با نگاه کردن به باقی مانده ها بررسی کنیم. 
# اگر مدل خوب است، باقی مانده های آن باید مانند نویز سفید به نظر برسند.
import matplotlib.pyplot as plt
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()




forecast_test = model_fit.forecast(len(df_test))
df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)
df.plot()

# print(df.to_string())
#dev_acc_d نام ستونی است که حذف میشود و به آن نیازی نیست

print(df.to_string())

# model = ARIMA(df, order=(1,0,1))
# model_fitted_ = model.fit()
# print(model_fitted.summary())
# model_fit.get_prediction(start='2022-03-19T00:00:00.000')
# model_fit.get_forecast('2022-04-01T00:00:00.000')
