import math
from socket import SIO_KEEPALIVE_VALS
from socketserver import ThreadingTCPServer
from statistics import mode
from keras import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

dataframe=read_csv('C:/Users/yjs61/Desktop/git/deeplearning/corona_daily.csv',usecols=[3],engine='python',skipfooter=3)

print(dataframe)
dataset=dataframe.values
dataset=dataset.astype('float32')

scaler=MinMaxScaler(feature_range =(0,1))
dataset=scaler.fit_transform(dataset)
train_data,test_data = train_test_split(dataset,test_size =0.2,shuffle=False)
print(len(train_data),len(test_data))

def create_dataset(dataset,look_back):
    x_data=[]
    y_data=[]

    for i in range(len(dataset)-look_back-1):
        data=dataset[i:(i+look_back),0]
        x_data.append(data)
        y_data.append(dataset[i+look_back, 0])
    return np.array(x_data),np.array(y_data)

look_back=3
x_train ,y_train = create_dataset(train_data ,look_back)
x_test,y_test = create_dataset(test_data,look_back)

print(x_train.shape , y_train.shape)
print(x_test.shape, y_test.shape)

X_train =np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
X_test = np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))
print(X_train.shape)
print(X_test.shape)

model =Sequential()
model.add(SimpleRNN(3,input_shape=(1,look_back)))
model.add(Dense(1,activation="linear"))
model.compile(loss='mse',optimizer='adam')
model.summary()

model.fit(X_train,y_train,epochs=100,batch_size=1,verbose=1)

trainPredict=model.predict(X_train)
testPredict =model.predict(X_test)
TrainPredict =scaler.inverse_transform(trainPredict)
Y_train = scaler.inverse_transform([y_train])
TestPredict = scaler.inverse_transform(testPredict)
Y_test = scaler.inverse_transform([y_test])

trainScore= math.sqrt(mean_squared_error(Y_train[0],TrainPredict[:,0]))
print('Train Score : %.2f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(Y_test[0],TestPredict[:,0]))
print('Test Score : %.2f RMSE'%(testScore))

trainPredictPlot= np.empty_like(dataset)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(TrainPredict)+look_back,:]=TrainPredict
testPredictPlot=np.empty_like(dataset)
testPredictPlot[:,:] =np.nan
testPredictPlot[len(TrainPredict)+(look_back+1)*2:len(dataset) , :]=TestPredict
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

