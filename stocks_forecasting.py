#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import mplcursors


# In[2]:


df = yf.download('RELIANCE.NS')


# In[3]:


fig = plt.figure(figsize=(16,8))
plot = fig.add_subplot(111)

plt.title('historical price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)

mplcursors.cursor(hover=True)
plt.show()


# In[4]:


data = df.filter(['Close'])


# In[5]:


current_data = np.array(data).reshape(-1,1).tolist() 


# In[6]:


df = np.array(data).reshape(-1,1)


# In[7]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(np.array(df).reshape(-1,1))


# In[8]:


train_data = scaled_df[0: , :]

x_train = []
y_train = []

for i in range(90, len(train_data)):
    x_train.append(train_data[i-90:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 90:
        print(x_train)
        print(y_train)
        print()


# In[9]:


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[10]:


model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences = False))   
model.add(Dense(25))
model.add(Dense(1))


# In[11]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[12]:


model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[13]:


test_data = scaled_df[ -90: , :].tolist()
#test_data = [e[0] for e in test_data ]
x_test = []
y_test = []
for i in range(90 , 120):
    #print(i)
    x_test = (test_data[i-90:i ])
    #print(len(x_test))
    x_test = np.asarray(x_test)
    pred_data = model.predict(x_test.reshape(1 , x_test.shape[0], 1).tolist()) 
    
    y_test.append(pred_data[0][0])
    test_data.append(pred_data)
print(y_test)


# In[14]:


pred_next_30 = scaler.inverse_transform(np.asarray(y_test).reshape(-1,1))
pred_next_30


# In[15]:


train = current_data[6000:]
train.extend(pred_next_30.tolist())


plt.figure(figsize=(16, 8))
plt.title('model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.plot(train)
plt.legend(['train'], loc = 'lower right')
plt.show()


# ## Immediate day Prediction 

# In[16]:


next_day = data[-90:].values


# In[17]:


next_day_scaled = scaler.transform(next_day)


# In[18]:


next_day_pred = []
next_day_pred.append(next_day_scaled)
next_day_pred = np.array(next_day_pred)
next_day_pred = np.reshape(next_day_pred, (next_day_pred.shape[0], next_day_pred.shape[1], 1))


# In[19]:


predicted_price = model.predict(next_day_pred)
predicted_price = scaler.inverse_transform(predicted_price)
print(predicted_price)


# In[ ]:




