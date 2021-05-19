# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import yfinance as yf
import pandas as pd

df = yf.download('TATASTEEL.NS')

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')



import mplcursors
fig = plt.figure(figsize=(16,8))
plot = fig.add_subplot(111)

plt.title('historical price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)

mplcursors.cursor(hover=True)
plt.show()