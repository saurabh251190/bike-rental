#!/usr/bin/env python
# coding: utf-8

# In[55]:


import os
import pandas as pd
import numpy as np


# In[56]:


os.chdir("C:/Users/Saurabh Gautam/Desktop/project")
os.getcwd()


# In[57]:


bike=pd.read_csv("C:/Users/Saurabh Gautam/Desktop/day.csv", sep=",")


# In[58]:


bike.shape #(731,16)


# In[59]:


bike.dtypes #data types


# In[118]:


df=bike 


# In[119]:


df.head() #first five rows


# In[20]:


#Data understanding creating new columns


# In[120]:


df['real_season'] = bike['season'].replace([1,2,3,4],["Spring","Summer","Fall","Winter"])
df['real_yr'] = bike['yr'].replace([0,1],["2011","2012"])
df['real_holiday'] = bike['holiday'].replace([0,1],["Working day","Holiday"])
df['real_weathersit'] = bike['weathersit'].replace([1,2,3,4],["Clear","Cloudy/Mist","Rain/Snow/Fog","Heavy Rain/Snow/Fog"])


# In[121]:


df.dtypes


# In[122]:


df['weathersit'] = df['weathersit'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['yr'] = df['yr'].astype('category')
df['season'] = df['season'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['mnth'] = df['mnth'].astype('category')
df['real_season'] = df['real_season'].astype('category')
df['real_yr'] = df['real_yr'].astype('category')
df['real_holiday'] = df['real_holiday'].astype('category')
df['real_weathersit'] = df['real_weathersit'].astype('category')


# In[123]:


df.dtypes


# In[124]:


print(df.workingday.value_counts())
print(df.weekday.value_counts())


# In[125]:


#Check if there are missing values
df.isnull().sum() #no missing value


# In[111]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[126]:


#Check the bar graph of categorical Data using factorplot
sns.set_style("whitegrid")
sns.factorplot(data=df, x='real_season', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='real_weathersit', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='workingday', kind= 'count',size=4,aspect=2)


# In[127]:


plt.hist(data=df, x='hum', bins='auto', label='Temperature')
plt.xlabel('Humidity')
plt.title("Humidity Distribution")


# In[129]:


#Check for outliers in data using boxplot
sns.boxplot(data=df[['temp','atemp','windspeed','hum']])
fig=plt.gcf()
fig.set_size_inches(8,8)

df


# In[115]:


#remove outliers in windspeed

q75, q25 = np.percentile(df['windspeed'], [75 ,25])
print(q75,q25)
iqr = q75 - q25
print(iqr)
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min)
print(max)

df = df.drop(df[df.iloc[:,13] < min].index)
df = df.drop(df[df.iloc[:,13] > max].index)


# In[130]:


df.head()


# In[90]:


q75, q25 = np.percentile(df['hum'], [75 ,25])
print(q75,q25)
iqr = q75 - q25
print(iqr)
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min)
print(max)

df = df.drop(df[df.iloc[:,12] < min].index)
df = df.drop(df[df.iloc[:,12] > max].index)


# In[131]:


df = df.drop(columns=['holiday','instant','season','yr','mnth','dteday','atemp','casual','registered','workingday','weathersit'
                      ,'real_season','real_yr','real_holiday','real_weathersit'])


# In[132]:


df=df.drop(columns=['weekday'])


# In[133]:


df.head(10)


# In[135]:


#Import Libraries for decision tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[141]:


train,test = train_test_split(df, test_size = 0.2, random_state = 123)


# In[144]:


dt_model = DecisionTreeRegressor(random_state=123).fit(train.iloc[:,0:3], train.iloc[:,3])


# In[147]:


dt_predictions = dt_model.predict(test.iloc[:,0:3])


# In[148]:


df_dt = pd.DataFrame({'actual': test.iloc[:,3], 'pred': dt_predictions})
df_dt.head()


# In[150]:


#Function for Mean Absolute Percentage Error
def MAPE(y_actual,y_pred):
    mape = np.mean(np.abs((y_actual - y_pred)/y_actual))
    return mape


# In[152]:


#Calculate MAPE for decision tree
MAPE(test.iloc[:,3],dt_predictions)


# In[154]:


#Import library for RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


# In[157]:


rf_model = RandomForestRegressor(n_estimators=500,random_state=123).fit(train.iloc[:,0:3], train.iloc[:,3])


# In[159]:


rf_predictions = rf_model.predict(test.iloc[:,0:3])


# In[161]:


df_rf = pd.DataFrame({'actual': test.iloc[:,3], 'pred': rf_predictions})
df_rf.head()


# In[163]:


MAPE(test.iloc[:,3],rf_predictions)


# In[166]:


import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# In[168]:


lr_model = sm.OLS(train.iloc[:,3].astype(float), train.iloc[:,0:3].astype(float)).fit()


# In[170]:


lr_predictions = lr_model.predict(test.iloc[:,0:3])


# In[172]:


df_lr = pd.DataFrame({'actual': test.iloc[:,3], 'pred': lr_predictions})
df_lr.head()


# In[173]:


MAPE(test.iloc[:,3],lr_predictions)

