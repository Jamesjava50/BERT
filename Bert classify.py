#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
df = pd.read_csv("Testy.csv")
df.head(5)
df['Topic'].value_counts()
df_Appointments = df[df['Topic']=='Appointments']
df_public_holidays = df[df['Topic']=='public_holidays']
df_Job_vacancies = df[df['Topic']=='Job_vacancies']
df_Tender = df[df['Topic']=='Tender']
df_balanced = pd.concat([df_Appointments, df_public_holidays, df_Tender, df_Job_vacancies])
df_balanced['Topic'].value_counts()


# In[49]:


df_balanced['Appointments']=df_balanced['Topic'].apply(lambda x: 1 if x=='Appointments' else 0)
df_balanced['Tender']=df_balanced['Topic'].apply(lambda x: 2 if x=='Tender' else 0)
df_balanced['public_holidays']=df_balanced['Topic'].apply(lambda x: 3 if x=='public_holidays' else 0)
df_balanced['Job_vacancies']=df_balanced['Topic'].apply(lambda x: 4 if x=='Job_vacancies' else 0)
df_balanced.sample(5)


# In[8]:


df.shape


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Task'],df_balanced['Appointments'], stratify=df_balanced['Appointments'])
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Task'],df_balanced['Tender'], stratify=df_balanced['Tender'])
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Task'],df_balanced['public_holidays'], stratify=df_balanced['public_holidays'])
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Task'],df_balanced['Job_vacancies'], stratify=df_balanced['Job_vacancies'])


# In[15]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[16]:


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)


# In[17]:


l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='softmax', name="output")(l)


# In[18]:


model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.summary()


# In[52]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
 loss='categorical_crossentropy',
 metrics=METRICS)


# In[41]:


model.fit(X_train, y_train, epochs=10)
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


# In[47]:


import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0,)
y_predicted = np.where(y_predicted > 1, 2, 1)
y_predicted = np.where(y_predicted > 1.5, 3, 2)
y_predicted = np.where(y_predicted > 2, 4, 3)
y_predicted


# In[51]:


sample_dataset = ["IN EXERCISE of the authority conferred in me by the Constitution of Kenya, the County Governments Act and the Public Appointments (County Assemblies Approval) Act, I, appoint"]
model.predict(sample_dataset)


# In[ ]:




