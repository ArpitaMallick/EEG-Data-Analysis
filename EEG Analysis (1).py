#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import sweetviz as sv


# In[52]:


df = pd.read_csv("EEG_data.csv")
subs=pd.read_csv("datasets_106_24522_demographic_info.csv")


# In[53]:


print(df.columns.values)
print(df.shape)
print(subs.columns.values)


# In[54]:


subs.head()


# In[55]:


df.rename(columns={'SubjectID':'subject ID'},inplace=True)

print(df.columns.values)


# In[56]:


df['user-definedlabeln'].value_counts()


# In[57]:


print(subs.head)


# In[58]:


for col in list(subs.columns):
    if subs[col].dtype == 'object':
        dums=pd.get_dummies(subs[col])
        subs = pd.concat([dums,subs], axis=1, join='outer')
        subs = subs.drop(col, 1)

print(subs.head())


# In[59]:


subs.head()


# In[60]:


merged=df.merge(subs, on='subject ID')
print(merged.head())


# In[61]:


merged['predefinedlabel'].value_counts()


# In[62]:


merged.columns


# In[63]:


seed=7
def set_aside_test_data(d):
    label=d.pop("user-definedlabeln") 
    X_train,X_test,y_train,y_test = train_test_split(d,label,test_size=0.2,random_state=seed)
    return X_train,X_test,y_train,y_test
    
X_train,X_test,y_train,y_test = set_aside_test_data(merged)


# In[64]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# # ANN

# In[65]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[68]:


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[69]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[71]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# # XG Boost

# In[73]:


import xgboost
from sklearn.metrics import accuracy_score


# In[75]:


model = xgboost.XGBClassifier()
model.fit(X_train, y_train)


# In[76]:


y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:




