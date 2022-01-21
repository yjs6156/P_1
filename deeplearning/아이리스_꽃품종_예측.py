
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

df=pd.read_csv('C:/Users/yjs61/Desktop/python/dataset/iris.csv',
names = ["sepal_length","sepal_width","petal_length","petal_width","species"])

sns.pairplot(df,hue='species')
plt.show()

dataset=df.values
X=dataset[:,0:4].astype(float)
Y_obj =dataset[:,4]

e=LabelEncoder()
e.fit(Y_obj)
Y=e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

model=Sequential()
model.add(Dense(16,input_dim=4,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,Y_encoded,epochs=50,batch_size=1)

print("\n Accuracy: %.4f" %(model.evaluate(X,Y_encoded)[1]))