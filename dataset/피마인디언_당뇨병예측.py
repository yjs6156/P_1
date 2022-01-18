from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy
import tensorflow as tf

numpy.random.seed(3)
tf.random.set_seed(3)

dataset = numpy.loadtxt("C:/Users/yjs61/Desktop/python/dataset/pima-indians-diabetes.csv",delimiter=",")
X=dataset[:,0:8]
Y=dataset[:,8]

model =Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,Y,epochs=200,batch_size=10)
print("\n Accuarcy: %.4f" %(model.evaluate(X,Y)[1]))