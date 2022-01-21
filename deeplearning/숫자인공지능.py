import keras
from tensorflow.keras.models import Sequential #케라스의 모델도구 시퀀셜모델 함수 시퀀셜모델을 불러오는 명령어
from tensorflow.keras.layers import Dense, Activation##dense 각레이어 뉴런개수 설정 activation 활성화함수
from tensorflow.keras.utils import to_categorical#0부터 9사이 숫자이미지 구별하는 인공지능,원-핫 인코딩 구현할수있는 함수
from tensorflow.keras.datasets import mnist#mnist 데이터셋 불러오는 명령어
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_math_ops import mod

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print("x_train shpae",x_train.shape)
print("y_train shape,",y_train.shape)
print("x_test shape",x_test.shape)
print("y_test shape",y_test.shape)


X_train=x_train.reshape(60000, 784)
X_test=x_test.reshape(10000, 784)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train /=255
X_test /=255
print("X Traning matrix shape",X_train.shape)
print("X Testing matrix shape",X_test.shape)

Y_train=to_categorical(y_train,10)
Y_test=to_categorical(y_test,10)
print("Y_Training matrix shape",Y_train.shape)
print("Y_Testing matrix shape",Y_test.shape)
#인공지능을 만들기위해 데이터를 내가 만들기 원하는 방향으로 변환하는 과정

model = Sequential()#딥러닝 모델 쉽게 개발할 수 있도록 도와줌
model.add(Dense(512, input_shape=(784,)))#모델에 층을추가 add, 층이 어떤 형태인지 설정하기위해 Dense함수 사용 첫번째 은닉충 노드 512개
model.add(Activation('relu'))#활성화 함수 relu함수로 설정
model.add(Dense(256))#다음충 두번째 은닉충은 256개 노드구성 
model.add(Activation('relu'))#두번째 은닉층도 활성화 함수 relu로 구성
model.add(Dense(10))#마지막 3번째층 10개노드 구성 그이유는 최종 결괏값이 0부터9까지 중 하나로 결정되기 떄문
model.add(Activation('softmax'))#각 노드 전달된 값의 총합이 1이 되도록 소프트맥스 함수 사용 
model.summary()#summary 함수는 모델 구성 살펴보는 함수

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])#함수 사용규칙 3가지 오차값계산 방법,오차 줄이는 방법(옵티마이저 사용{adam})) 3번째 학습결과 어떻게 확인할지
model.fit(X_train,Y_train,batch_size=128,epochs=10,verbose=1)#케라스는 학습시키기위해 맞춘다라는 의미의 fit함수 제공,규칙 첫번쟤 입력할 데이터 정하기 두번째 배치 사이즈 정하기 세번째 에포크 정하기
#배치 사이즈는 인공지능 모델이 한번에 학습하는 데이터의 수, 에포크는 모든 데이터를 1번 학습하는 것을 의미

score=model.evaluate(X_test,Y_test) #테스트 정확도 평가 evaluate 함수에 데이터 넣으면 두가지 결과 첫번째 오차값 오차값은 0~1사이 0이면 오차없음 1이면 오차가 아주 크다는 의미 두번째 정확도(accuarcy) 1에 가까울 수록 정답을 많이 맞춘것을 의미
print('Test score: ',score[0])#score 변수에는 오차값과 정확도 들어있음 오차값 출력위해 score 변수 첫번째 항목인 점수 출력
print('Test accuaracy: ',score[1])#score 변수 두번째 항목인 정확도 출력 최종오차 0.08 정확도 0.98나옴

predicted_classes = np.argmax(model.predict(X_test),axis=1)
correct_indices =np.nonzero(predicted_classes == y_test)[0]#실제값과 예측값이 일치하는 값을 찾아내여 correct_indices변수에 저장하는 과정
incorrect_indices =np.nonzero(predicted_classes !=y_test)[0]

plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    correct = correct_indices[i]
    plt.imshow(X_test[correct].reshape(28,28),cmap='gray')
    plt.title("Predicted {},Class {}".format(predicted_classes[correct],y_test[correct]))
    plt.tight_layout()
    plt.show()


    #변경완료
    
