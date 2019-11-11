#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import xlrd
from sklearn.impute import SimpleImputer


# In[75]:


pop_2010 = pd.read_excel(r"C:\Users\sdpou\Desktop\sdp\Prescription_Population_Combined (1).xlsx")["April 1, 2010 - Census"]
pop_2010 = pop_2010.fillna(pop_2010.mean())
pop_2010 = pop_2010.to_numpy()

pop_2015 = pd.read_excel(r"C:\Users\sdpou\Desktop\sdp\Prescription_Population_Combined (1).xlsx")["Population Estimate (as of July 1) - 2015"]
pop_2015 = pop_2015.fillna(pop_2015.mean())
pop_2015 = pop_2015.to_numpy()

rx_2010 = pd.read_excel(r"C:\Users\sdpou\Desktop\sdp\Prescription_Population_Combined (1).xlsx")["MME per cap 2010"]
rx_2010 = rx_2010.fillna(rx_2010.mean())
rx_2010 = rx_2010.to_numpy()

rx_2015 = pd.read_excel(r"C:\Users\sdpou\Desktop\sdp\Prescription_Population_Combined (1).xlsx")["MME per cap 2015"]
rx_2015 = rx_2015.fillna(rx_2015.mean())
rx_2015 = rx_2015.to_numpy()


# In[76]:


##pop_2010
##pop_2015
##rx_2010
##rx_2015


# In[77]:


##data = pd.read_excel(r"C:\Users\sdpou\Desktop\sdp\Prescription_Population_Combined (1).xlsx")


# In[78]:


linearReg2010 = LinearRegression()
linearReg2015 = LinearRegression()


# In[79]:


pop_2010 = pop_2010.reshape(-1, 1)
rx_2010 = rx_2010.reshape(-1,1)
pop_2015 = pop_2015.reshape(-1, 1)
rx_2015 = rx_2015.reshape(-1, 1)


# In[80]:


linearReg2010.fit(pop_2010, rx_2010)


# In[81]:


plt.scatter(pop_2010, rx_2010)
slope_2010 = linearReg2010.predict(pop_2010)
plt.plot(pop_2010, slope_2010, color = 'red')
plt.xlabel("Population")
plt.ylabel("Prescriptions")
plt.title("2010")
plt.show()


# In[82]:


linearReg2015.fit(pop_2015, rx_2015)


# In[83]:


plt.scatter(pop_2015, rx_2015)
plt.scatter(pop_2015, rx_2015)
slope_2015 = linearReg2015.predict(pop_2015)
plt.plot(pop_2015, slope_2015, color = 'purple')
plt.xlabel("Population")
plt.ylabel("Prescriptions")
plt.title("2015")
plt.show()


# In[ ]:





# In[ ]:




