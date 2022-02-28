#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
df = pd.read_csv("Testy.csv")
df.head(10)


# In[5]:


df['Topic'].value_counts()
df_Appointments = df[df['Topic']=='Appointments']
df_public_holidays = df[df['Topic']=='public_holidays']
df_Job_vacancies = df[df['Topic']=='Job_vacancies']
df_Tender = df[df['Topic']=='Tender']
df_balanced = pd.concat([df_Appointments, df_public_holidays, df_Tender, df_Job_vacancies])
df_balanced['Topic'].value_counts()


# In[6]:


df_balanced['Appointments']=df_balanced['Topic'].apply(lambda x: 1 if x=='Appointments' else 2 if x=='Tender' else 3 if x=='public_holidays' else 0) 
df_balanced.sample(20)


# In[7]:


df.shape


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Task'],df_balanced['Appointments'], stratify=df_balanced['Appointments'])


# In[12]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[44]:


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)


# In[ ]:


l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='softmax', name="output")(l)


# In[35]:


model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.summary()


# In[16]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
 loss='categorical_crossentropy',
 metrics=METRICS)


# In[17]:


model.fit(X_train, y_train, epochs=10)
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


# In[20]:


y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


# In[21]:


import numpy as np

y_predicted = np.where(y_predicted == 1, 2, 3)
y_predicted


# In[23]:


sample_dataset = ["NOTICE given under Disposal of Uncollected Goods Act (Cap.38) of the laws of Kenya from Messrs. Ray Securior Group Limited, Nairobi and of P.O. Box 3719–00100, Nairobi to Messrs. Keysian Auctioneers of P.O. Box 2788–00200, Nairobi and Peter Mitema of P.O. Box 4614–00100, Nairobi who brought and stored Motor Vehicle Reg No. KBS 752J, Toyota Town Ace, that the same shall be sold by private treaty or by public auction after expiry of thirty (30) days from the date of publication of this notice without any further reference to yourselves or the owners unless the total outstanding storage and other incidentals amounting to KSh. 460,520.00 as at 31st December, 2018 which amount continue to attract storage until full payment is  received."]
model.predict(sample_dataset)


# In[ ]:




